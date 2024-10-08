from config.GPTconfig import MinGPTConfig, GPT2Config, ToyGPTConfig
from GPT import GPT

import torch
import numpy as np

import os
import sys

from utils import interpolate_weights, visualize_interpolation

model_config = MinGPTConfig()
dataset = 'fineweb'

device = 'cpu'
device_type = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using {device}")

colab = False

model_iters = 9500 

# Interpolation
eval_iters = 30
res = 30

extra = ''

# Take command-line configurations
exec(open('config/configurator.py').read())

dir_path = ''
if colab:
    dir_path = '/content/drive/My Drive/lmc-transformers'

model1 = GPT(model_config)
model2 = GPT(model_config)
baseline = GPT(model_config)

print(f"Model config: {model_config}")

# Interpolate
checkpoint = torch.load(os.path.join(dir_path, 'models', dataset, model_config.__class__.__name__, 'model1', f'iteration={model_iters}.checkpoint.pth.tar'), map_location=torch.device(device), weights_only=True)
model1.load_state_dict(checkpoint['model'])
checkpoint = torch.load(os.path.join(dir_path, 'models', dataset, model_config.__class__.__name__, 'model2', f'iteration={model_iters}.checkpoint.pth.tar'), map_location=torch.device(device), weights_only=True)
model2.load_state_dict(checkpoint['model'])

batch_size = model_config.batch_size
block_size = model_config.block_size


# Sampling (Optional)
# import tiktoken

# prompt = "No, sir, nor I mean it not."
# num_copies = 2
# max_len = 30
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode(prompt)

# model1.to(device)
# next_tokens = model1.generate(tokens, max_len, num_copies, device)

# for i in range(num_copies):
#     tokens = next_tokens[i, :max_len].tolist()
#     text = enc.decode(tokens)
#     print(text)
# sys.exit(0)

# Dataset

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
        num_batches = (len(data) - block_size) // (block_size * batch_size)
        for batch_idx in range(num_batches):
            # Prepare the batch data
            batch_x = []
            batch_y = []
            for i in range(batch_size):
                start_idx = batch_idx * block_size * batch_size + i * block_size
                end_idx = start_idx + block_size
                
                x = data[start_idx:end_idx].astype(dtype=np.int64)
                y = data[start_idx + 1:end_idx + 1].astype(dtype=np.int64)
                
                batch_x.append(x)
                batch_y.append(y)
            
            # Convert to tensors
            batch_x = torch.from_numpy(np.array(batch_x)).to(device)
            batch_y = torch.from_numpy(np.array(batch_y)).to(device)

            with torch.no_grad():
                _, loss = model(batch_x, batch_y)
            
            # Accumulate loss
            total_loss += loss.detach().item()
        print(f"Count: {num_batches}")
        total_loss /= num_batches

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


alphas = np.linspace(0, 1, res)

error_rates = np.zeros((2, res))
for i, alpha in enumerate(alphas):
    interpolated_model = interpolate_weights(model1, model2, baseline, alpha, device=device)
    err = evaluate(interpolated_model, eval_iter=eval_iters, mode='val')
    error_rates[0, i] = err
    err = evaluate(interpolated_model, eval_iter=eval_iters, mode='train')
    error_rates[1, i] = err
    print(f"Iteration {i}")

visualize_interpolation(alphas, error_rates, dir_path, f'{dataset}_{model_config.__class__.__name__}_{extra}')




