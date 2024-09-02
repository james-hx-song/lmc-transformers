import torch
import time
import copy
import random
import sys

from config.GPTconfig import MinGPTConfig, GPT2Config, ToyGPTConfig
from GPT import GPT


# ----------------- Model Configuration ----------------- #
model_config = ToyGPTConfig()
dataset = 'fineweb'

# Optimizer Configuration
# LR
min_lr = 6e-5
max_lr = 6e-4

max_iters = 24000

betas = (0.9, 0.95)
eps = 1e-8

grad_clip = 1.0

device = 'cpu'
device_type = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using {device}")


exec(open('config/configurator.py').read())

batch_size = model_config.batch_size
block_size = model_config.block_size

print(f"Model config: {model_config}")

# ----------------- Data Loading ----------------- #


# ----------------- Model Architecture ----------------- #
model1 = GPT(model_config)
model2 = GPT(model_config)

print(f"Number of parameters: {sum(p.numel() for p in model1.parameters())}")


model2.load_state_dict(model1.state_dict()) # Ensure they have the same initial parameters

optimizer1 = torch.optim.Adam(model1.parameters(), lr=min_lr, betas=betas, eps=eps)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=min_lr, betas=betas, eps=eps)






