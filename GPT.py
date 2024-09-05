from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import config.GPTconfig as GPTconfig
import inspect

# ----------------- Model Architecture ----------------- #
# We follow the naming convention of GPT-2 (GPT2LMHeadModel) from HuggingFace
# Largest model we are training is 124 M GPT-2 model


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, config.n_embed*3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x) # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=2) # (B, T, C) x 3

        # Now we want to split the heads, so # (B, nh, T, hs = C/nh)
        q = q.view(B, T, self.n_head, C// self.n_head).transpose(1, 2) # (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V

        # head_size = q.size(-1)
        # attn_score = (q @ k.transpose(-2, -1)) * (head_size ** (-0.5)) # (B, nh, T, T)
        # attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # attn_prob = F.softmax(attn_score, dim=-1)
        # attn_vec = attn_prob @ v # (B, nh, T, hs)
        
        # Flash attention
        attn_vec = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Concatenate the heads
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        out = self.c_proj(attn_vec) # (B, T, C)

        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed*4) # Both Vaswani eta al. and GPT-2 use 4x
        self.c_proj = nn.Linear(config.n_embed*4, config.n_embed)

        # GPT-2 uses GELU; Vaswani et al. uses ReLU
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Vaswani et al. use LayerNorm(x + Attention(LayerNorm(x)))
        x = x + self.attn(self.ln_1(x)) # Residual Connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed), #  Token Embeddings
                wpe = nn.Embedding(config.block_size, config.n_embed), #  Positional Embeddings
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer Blocks
                ln_f = nn.LayerNorm(config.n_embed) # Layer Norm
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # Weight Sharing Scheme (Vaswani et al. 2017):
        # "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax
        # linear transformation, similar to [30]". 
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module == self.transformer.wte:
                torch.nn.init.normal_(module.weight, std=0.02)
            elif module == self.transformer.wpe:
                torch.nn.init.normal_(module.weight, std=0.01)
            

    def forward(self, x, y=None):
        # Token + Position
        x = self.transformer.wte(x) + self.transformer.wpe(torch.arange(x.size(1), device=x.device))

        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
        
        # According to GPT 2 repo, they add layer norm at the end
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        # The model name must be one of the following:
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768), # 124 M Parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024), # 350 M Parameters
            'gpt2-large': dict(n_layers=36, n_head=20, n_embed=1280), # 774 M Parameters
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embed=1600) # 1.558 B Parameters
        }[model_name]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Now that we have all the params, we load it into dataclass
        config = GPTconfig.GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()


        # Ignore the ones ending with attn.mask
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')]


        # for k in sd_keys:
        #     print(k)
        # Load the Pretrained model
        from transformers import GPT2LMHeadModel
        model_pre = GPT2LMHeadModel.from_pretrained(model_name)
        sd_pre = model_pre.state_dict()
        # print(sd_pre.keys())
        sd_keys_pre = sd_pre.keys()
        sd_keys_pre = [k for k in sd_keys_pre if not k.endswith('.attn.bias')]
        sd_keys_pre = [k for k in sd_keys_pre if not k.endswith('.attn.masked_bias')]

        # Manually checking dimensions, some keys are transposed for whatever reason
        transposed_keys = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys) == len(sd_keys_pre), f"{len(sd_keys)} != {len(sd_keys_pre)}"
        for k in sd_keys_pre:
            if any(k.endswith(tk) for tk in transposed_keys):
                assert sd_pre[k].T.shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_pre[k].T)
            else:
                # vanilla copy
                assert sd_pre[k].shape == sd[k].shape, f"Key: {k}, {sd_pre[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_pre[k])

        return model
    
    def generate(self, prefix_tokens, max_len, num_copies, device):
        prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0)
        prefix_tokens = prefix_tokens.repeat(num_copies, 1)

        x = prefix_tokens.to(device)
        while x.size(1) < max_len:
            with torch.no_grad():
                print(x.shape)
                logits, _ = self(x)
                print(logits.shape)

                logits = logits[:, -1, :] # (B, vocab_size) gets last token
                
                proba = F.softmax(logits, dim=-1)

                # Sample from the distribution
                topk_probs, topk_idx = torch.topk(proba, 50, dim=-1)
                idx = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_idx, 1, idx)
                x = torch.cat((x, xcol), dim=1)

        return x

if __name__ == '__main__':

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTconfig.TinyGPTConfig())
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # prompt = "Hello, I am a Large Language Model. "
    # num_copies = 2
    # max_len = 30
    # device = 'mps'

    # model = GPT(GPTConfig())
    # model.to(device)

    # import tiktoken
    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode(prompt)
    
    # next_tokens = model.generate(tokens, max_len, num_copies, device)

    # for i in range(num_copies):
    #     tokens = next_tokens[i, :max_len].tolist()
    #     text = enc.decode(tokens)
    #     print(text)
    