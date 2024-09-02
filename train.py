import torch
import numpy as np

import time
import sys
import os

from config.GPTconfig import MinGPTConfig, GPT2Config, ToyGPTConfig
from GPT import GPT

from utils import interpolate_weights, visualize_interpolation


# ----------------- Model Configuration ----------------- #
model_config = MinGPTConfig()
dataset = 'fineweb'

# Optimizer Configuration
# LR
min_lr = 6e-5
max_lr = 6e-4

max_iters = 3000

betas = (0.9, 0.95)
eps = 1e-8

grad_clip = 1.0

# Scheduler Configuration
warmup_steps = 1200
max_steps = 2500

device = 'cpu'
device_type = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using {device}")

colab = False


exec(open('config/configurator.py').read())
dir_path = ''
if colab:
    dir_path = '/content/drive/My Drive/lmc-transformers'

batch_size = model_config.batch_size
block_size = model_config.block_size

print(f"Model config: {model_config}")

# ----------------- Data Loading ----------------- #
dataset_dir = f'datasets/{dataset}'
exec(open(os.path.join(dataset_dir, 'prepare.py')).read())

train_data = np.memmap(os.path.join(dataset_dir, 'train.npy'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(dataset_dir, 'val.npy'), dtype=np.uint16, mode='r')

def get_batch(mode):
    if mode == 'train':
        data = train_data
    else:
        data = val_data
    
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(dtype=np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(dtype=np.int64)) for i in idx])

    return x, y

def evaluate(model, eval_iter='all', mode='train'):
    model.eval()
    if mode == 'train':
        data = train_data
    else:
        data = val_data
    total_loss = 0
    if eval_iter == 'all':
        count = 0
        for i in range(0, len(data) - block_size, block_size):
            x = torch.from_numpy(data[i:i+block_size].astype(dtype=np.int64)).unsqueeze(0)
            y = torch.from_numpy(data[i+1:i+block_size+1].astype(dtype=np.int64)).unsqueeze(0)

            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                _, loss = model(x, y)

            total_loss += loss.detach().item()
            count += 1
        print(f"Count: {count}")
        total_loss /= count

    else:
        assert type(eval_iter) == int, "eval_iter must be 'all' or integer"
        for _ in range(eval_iter):
            x, y = get_batch(mode)
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                _, loss = model(x, y)

            total_loss += loss.detach().item()
        total_loss /= eval_iter
    
    return total_loss

def get_lr(i):
    if i < warmup_steps:
        return min_lr + (max_lr - min_lr) * i / warmup_steps
    elif i > max_steps:
        return min_lr
    
    ratio = (i - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + np.cos(np.pi * ratio))

    return min_lr + (max_lr - min_lr) * coeff


# ----------------- Model Architecture ----------------- #
model1 = GPT(model_config)
model2 = GPT(model_config)
baseline = GPT(model_config)

print(f"Number of parameters: {sum(p.numel() for p in model1.parameters())}")


model2.load_state_dict(model1.state_dict()) # Ensure they have the same initial parameters

optimizer1 = torch.optim.Adam(model1.parameters(), lr=min_lr, betas=betas, eps=eps)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=min_lr, betas=betas, eps=eps)

model_dir = os.path.join(dir_path, f'models/{dataset}/{model_config.__class__.__name__}')
print(model_dir)
if not os.path.exists(os.path.join(model_dir, 'model1')):
    os.makedirs(os.path.join(model_dir, 'model1'))
    os.makedirs(os.path.join(model_dir, 'model2'))

# ----------------- Training ----------------- #
def train(model, optimizer, model_name='model1'):
    model.to(device)

    # print(f"Initial Loss: {evaluate(model, 100)}")
    # sys.exit(0)

    for i in range(max_iters):
        x, y = get_batch('train')
        x, y = x.to(device), y.to(device)

        t0 = time.time()

        optimizer.zero_grad()

        logits, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(i)

        
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        
        if i % 500 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, model_name, f'iteration={i}.checkpoint.pth.tar'))

        print(f"Step {i:5} | Loss: {loss:10.6f} | Time: {(t1-t0)*1e3:10.2f} ms")


train(model1, optimizer1, 'model1')
train(model2, optimizer2, 'model2')


# ----------------- Linear Interpolation ----------------- #
res = 30
alphas = torch.linspace(0, 1, res)

error_rates = torch.zeros((2, res))
eval_iter = 100
for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, baseline, alpha, device=device)
    err = evaluate(interpolated_model, eval_iter=eval_iter, mode='val')
    error_rates[0, i] = err
    err = evaluate(interpolated_model, eval_iter=eval_iter, mode='train')
    error_rates[1, i] = err

visualize_interpolation(alphas, error_rates, dir_path, f'{dataset}_{model_config.__class__.__name__}')
