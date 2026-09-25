import os
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/qwen3-0.6b-base"

# Don't let PyTorch use more threads than the VPS actually has.
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("Loading model...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.eval()

print(f"Loaded in {time.time() - start:.2f}s")

parameter_count = sum(p.numel() for p in model.parameters())

print(f"Parameters: {parameter_count:,}")
print(f"Parameters (B): {parameter_count / 1e9:.3f}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Device: {next(model.parameters()).device}")
