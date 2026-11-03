"""
Create a tiny random OLMoE checkpoint using the REAL Hugging Face implementation.

    pip install torch transformers
    python tools/make_tiny_hf.py tiny_olmoe

This is the cheap first gate against real Hugging Face code: no 14 GB download, runs in seconds.
Then:
    python tools/convert_hf.py tiny_olmoe tiny.rbt2 --dtype f32
    python tools/check_vs_hf.py tiny_olmoe tiny.rbt2
"""
import os
import sys

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

out = sys.argv[1] if len(sys.argv) > 1 else "tiny_olmoe"

cfg = OlmoeConfig(
    vocab_size=512, hidden_size=128, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=8, num_key_value_heads=8, num_experts=8, num_experts_per_tok=2,
    max_position_embeddings=256, norm_topk_prob=False, tie_word_embeddings=False, rms_norm_eps=1e-5,
)
torch.manual_seed(0)
model = OlmoeForCausalLM(cfg).float()

# Default init leaves norms at 1 and weights tiny, which hides bugs. Make everything lively.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.ndim == 1:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        elif "embed_tokens" in name:
            p.normal_(0.0, 0.5)
        else:
            p.normal_(0.0, p.shape[1] ** -0.5)

model.save_pretrained(out, safe_serialization=True)
print(f"saved tiny OLMoE to {out}/")

# Show the tensor names that were actually written, so a naming mismatch is easy to spot.
from safetensors import safe_open  # noqa: E402  (installed with transformers)

for fn in sorted(os.listdir(out)):
    if fn.endswith(".safetensors"):
        with safe_open(os.path.join(out, fn), framework="pt") as f:
            names = sorted(f.keys())
        print(f"{fn}: {len(names)} tensors, e.g.")
        for n in [n for n in names if "layers.0." in n][:8]:
            print("   ", n)
