
from dataclasses import dataclass
@dataclass
class GPT2Config:
    vocab_size: int = 50257 # (Radford et al. 2020), GPT 2 Tokenizer
    n_embed: int = 768
    block_size: int = 1024
    batch_size: int = 512
    n_layer: int = 12
    n_head: int = 12

@dataclass
class MinGPTConfig: 
    vocab_size: int = 50257
    n_embed: int = 384
    block_size: int = 256
    batch_size: int = 32
    n_layer: int = 6
    n_head: int = 6

@dataclass
class ToyGPTConfig:
    vocab_size: int = 50257
    n_embed: int = 32
    block_size: int = 8
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4





