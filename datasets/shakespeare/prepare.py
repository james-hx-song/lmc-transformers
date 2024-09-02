import tiktoken
import requests
import numpy as np
import os

print("Downloading Shakespeare dataset")
input_file_path = os.path.join(os.path.dirname(__file__), 'datasets/shakespeare/', 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

enc = tiktoken.get_encoding('gpt2')

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_tokens = enc.encode(train_data)
val_tokens = enc.encode(val_data)

train_tokens = np.array(train_tokens, dtype=np.uint16)
val_tokens = np.array(val_tokens, dtype=np.uint16)

print(f"Train tokens length: {len(train_tokens)}")
print(f"Val tokens length: {len(val_tokens)}")

train_tokens.tofile(os.path.join(os.path.dirname(__file__), 'datasets/shakespeare/', 'train.npy'))
val_tokens.tofile(os.path.join(os.path.dirname(__file__), 'datasets/shakespeare/', 'val.npy'))




