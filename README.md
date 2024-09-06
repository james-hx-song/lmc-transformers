# Linear Mode Connectivity and Transformer

This is a fun little experiment on whether Linear Mode Connectivity occurs in transformers. 

Despite the highly complex loss landscapes that often occur in neural network training, there has been interesting phenomena and patterns that are shown empirically. Linear Mode Connectivity (LMC) is one of those phenomena due to the observation that different solutions reached by Stochastic Gradient Descent can be connected with a linear path. In this work, we study whether LMC occurs in Transformers. We provide empirical evidence across decoder-only transformers with varying sizes and model configurations, which follows GPT-2  (Radford et al.(2019), Brown et al. (2020)). We demonstrate empirically that transformers are stable during training and unstable during initialization, which means linear mode connectivity occurs when we train or fine-tune a pretrained model. 

For an in depth read of the results, please read the pdf [here](.github/LMC_Transformers.pdf). 

## How to run the code

There are many improvements that could be made to this repo. First, since this is only a simple experimental investigation, we only used the tiny shakespeare dataset. Fineweb can most definitely be used to train the models.  

For a basic run of the programs, ```train.py``` is the file you should be looking for. An example would be something like
```bash
python3 train.py --model_config=GPT2 --dataset=shakespeare --max_iters=1000 --warmup_steps=200 --max_steps=1000 
```

If you are training on a GPU and the batch size don't fit, you can customize the batch size, but we still use gradient accumulation to "simulate" the actual batch size.
```bash
python3 train.py --model_config=GPT2 --dataset=shakespeare --max_iters=1000 --warmup_steps=200 --max_steps=1000 --sim_batch_size=8
```
I also have fine tuning setup in this repo. However, it currently only supports 124 M GPT-2. If you want to finetune, simply set ```fine_tune=True```.
```bash
python3 train.py --model_config=GPT2 --dataset=shakespeare --max_iters=1000 --warmup_steps=200 --max_steps=1000 --sim_batch_size=8 --fine_tune=True
```

If you are running on colab, there is an option to save your models to a folder on your google drive. First, you would have to mount your drive. Then, simply set the colab flag to be true: ```--colab=True```
```bash
python3 train.py --dataset=shakespeare --model_config=CompactGPT --colab=True --max_iters=10000 --warmup_steps=1000 --max_steps=10000
```

If you are doing interpolation on models that are already trained, here is an example:
```bash
python3 interpolate.py --model_config=GPT2 --model_iters=200 --eval_iters=all --dataset=shakespeare
```

Notice that ```eval_iters``` can be set to either "all" or an integer. If it is set to "all", it will evaluate the model on the entire test dataset. Otherwise, it will sample from the test dataset ```eval_iters``` number of times, with batch size specified in the model configurations. 
