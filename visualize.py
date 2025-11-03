print("\n✅ Quantization comparison complete!")

import matplotlib.pyplot as plt

precisions = ['FP32', 'INT8', 'INT4']
model_sizes = [330, 160, 90]
inference_times = [3.5, 2.2, 1.6]

plt.figure(figsize=(6,4))
plt.bar(precisions, model_sizes)
plt.title("Model Size vs Quantization Level")
plt.xlabel("Precision Type")
plt.ylabel("Model Size (MB)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(precisions, inference_times, color='orange')
plt.title("Inference Time vs Quantization Level")
plt.xlabel("Precision Type")
plt.ylabel("Inference Time (seconds)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
