import sys
from ast import literal_eval
from config.GPTconfig import MinGPTConfig, GPT2Config, ToyGPTConfig, CompactGPTConfig, TinyGPTConfig

for arg in sys.argv[1:]:
    if '=' not in arg:
        raise ValueError(f"Argument {arg} is not in the form of key=value")
    
    assert arg.startswith('--')
    key, val = arg[2:].split('=')

    if key in globals():

        if key == 'model_config':
            if 'MinGPT' in val:
                attempt = MinGPTConfig()
            elif 'GPT2' in val:
                attempt = GPT2Config()
            elif 'ToyGPT' in val:
                attempt = ToyGPTConfig()
            elif 'CompactGPT' in val:
                attempt = CompactGPTConfig()
            elif 'TinyGPT' in val:
                attempt = TinyGPTConfig()
            else:
                raise ValueError(f"Model config {val} not found")
        else:
            try:
                attempt = literal_eval(val)
            except:
                attempt = val
            
            assert type(attempt) == type(globals()[key]) or (key == 'eval_iters' and type(attempt) == str) or (key == 'sim_batch_size' and type(attempt) == int), f"Argument of {key} ({val}) is not of type {type(globals()[key])}"

        globals()[key] = attempt

        print(f"Set {key} to {val}")

    else:
        raise ValueError(f"Argument {key} is not a valid key")

