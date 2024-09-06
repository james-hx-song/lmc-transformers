
from dataclasses import dataclass
@dataclass
class GPT2Config: # 124 M
    vocab_size: int = 50257 # (Radford et al. 2020), GPT 2 Tokenizer
    n_embed: int = 768
    block_size: int = 1024
    # batch_size: int = 512
    batch_size: int = 8
    n_layer: int = 12
    n_head: int = 12

@dataclass
class MinGPTConfig: # 30 M
    vocab_size: int = 50257
    n_embed: int = 384
    block_size: int = 256
    batch_size: int = 32
    n_layer: int = 6
    n_head: int = 6

@dataclass
class CompactGPTConfig: # 12 M
    vocab_size: int = 50257
    n_embed: int = 192
    block_size: int = 64
    batch_size: int = 32
    n_layer: int = 6
    n_head: int = 6

@dataclass
class ToyGPTConfig: # 1.65 M
    vocab_size: int = 50257
    n_embed: int = 32
    block_size: int = 8
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4

@dataclass
class TinyGPTConfig: # 204 K
    vocab_size: int = 65
    n_embed: int = 64
    block_size: int = 8
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4





