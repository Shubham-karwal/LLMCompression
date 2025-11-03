import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os

# Helper to calculate model size (in MB)
def model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)  # FP32 uses 4 bytes per parameter
    return size_mb

# Measure inference speed safely
def measure_inference_time(model, tokenizer, text):
    device = next(model.parameters()).device  # Match model device automatically
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=40)
    end = time.time()
    return end - start


def print_gpu_info():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory Allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))
        print("Memory Cached (MB):", round(torch.cuda.memory_reserved(0)/1024**2, 2))
    else:
        print("⚠️ No GPU detected. Running on CPU (slower).")



model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FP32 Model
print("\n🔹 Loading FP32 model...")
model_fp32 = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fp32.to(device)
size_fp32 = model_size(model_fp32)
print(f"Model size (FP32): {size_fp32:.2f} MB")

# 8-bit Quantized Model
print("\n🔹 Loading 8-bit quantized model...")
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)
print_gpu_info()

# 4-bit Quantized Model
print("\n🔹 Loading 4-bit quantized model...")
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
print_gpu_info()


test_text = "Artificial Intelligence is transforming the world because"
print("\nRunning inference test...")

t_fp32 = measure_inference_time(model_fp32, tokenizer, test_text)
t_8bit = measure_inference_time(model_8bit, tokenizer, test_text)
t_4bit = measure_inference_time(model_4bit, tokenizer, test_text)

print(f"\n🧮 Inference Time (FP32): {t_fp32:.2f}s")
print(f"🧮 Inference Time (8-bit): {t_8bit:.2f}s")
print(f"🧮 Inference Time (4-bit): {t_4bit:.2f}s")

print("\n✅ Quantization comparison complete!")
