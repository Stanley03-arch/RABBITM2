"""Print a capacity/memory plan for a .rbt2 model (read-only, no inference).

Inspired by Colibri-style planning: estimate footprint, expert size, and cache guidance.
Reads only the file header and computed layout.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import rbt2  # noqa: E402


def mib(n):
    return n / (1024 * 1024)


def gib(n):
    return n / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help=".rbt2 model file")
    ap.add_argument("--ram-gib", type=float, default=8.0, help="assumed system RAM for guidance")
    ap.add_argument("--ctx", type=int, default=512, help="context length for rough KV estimate")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        header = f.read(rbt2.HEADER_BYTES)
    if len(header) != rbt2.HEADER_BYTES:
        sys.exit("truncated header")
    c = rbt2.unpack_header(header)
    L = rbt2.layout(c)
    file_size = os.path.getsize(args.model)
    total_experts = c.n_layers * c.n_experts
    expert_bytes = L["expert_stride"]
    kv_dim = c.n_kv_heads * c.head_dim
    kv_bytes = 2 * c.n_layers * args.ctx * kv_dim * 4  # k+v, f32
    active_expert_bytes = c.n_layers * c.top_k * expert_bytes

    print(f"model: {args.model}")
    print(f"file size: {gib(file_size):.2f} GiB ({file_size:,} bytes)  layout total: {gib(L['total']):.2f} GiB")
    print(f"arch: layers={c.n_layers} dim={c.dim} heads={c.n_heads}/{c.n_kv_heads} head_dim={c.head_dim}")
    print(f"moe: experts/layer={c.n_experts} top_k={c.top_k} expert_hidden={c.expert_hidden} total experts={total_experts}")
    print(f"dtype: {'bf16' if c.dtype == rbt2.DTYPE_BF16 else 'f32'}  qk_norm={bool(c.qk_norm)} norm_topk={bool(c.norm_topk)} max_seq={c.max_seq}")
    print(f"expert blob: {mib(expert_bytes):.2f} MiB each (aligned)")
    print(f"per token (all selected experts, if uncached): ~{mib(active_expert_bytes):.1f} MiB read across layers")
    print(f"rough KV cache at ctx={args.ctx}: ~{mib(kv_bytes):.1f} MiB (f32, excludes activations/runtime)")
    ram = args.ram_gib * (1024 ** 3)
    reserve = 1.5 * (1024 ** 3)
    cache_budget = max(0, ram - reserve - kv_bytes)
    slots = int(cache_budget // expert_bytes) if expert_bytes else 0
    print(f"guidance (--ram-gib {args.ram_gib:g}): reserve ~1.5 GiB + KV => expert-cache payload budget ~{mib(cache_budget):.0f} MiB ({slots} expert slots)")
    print("notes: mmap default uses OS page cache; --expert-cache-mib budgets payload only, not total RAM; Windows/runtime not fully validated.")


if __name__ == "__main__":
    main()
