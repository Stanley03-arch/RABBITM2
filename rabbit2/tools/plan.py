"""Print a capacity/memory plan for a .rbt2 model (read-only, no inference).

Inspired by Colibri-style planning: estimate footprint, expert size, and cache guidance.
Reads only the file header and computed layout.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import rbt2  # noqa: E402


def mib(n):
    return n / (1024 * 1024)


def gib(n):
    return n / (1024 ** 3)


def compute_plan(model, ram_gib, ctx):
    with open(model, "rb") as f:
        header = f.read(rbt2.HEADER_BYTES)
    if len(header) != rbt2.HEADER_BYTES:
        raise ValueError("truncated header")
    c = rbt2.unpack_header(header)
    L = rbt2.layout(c)
    file_size = os.path.getsize(model)
    total_experts = c.n_layers * c.n_experts
    expert_bytes = L["expert_stride"]
    kv_dim = c.n_kv_heads * c.head_dim
    kv_bytes = 2 * c.n_layers * ctx * kv_dim * 4
    active_expert_bytes = c.n_layers * c.top_k * expert_bytes
    ram = ram_gib * (1024 ** 3)
    reserve = 1.5 * (1024 ** 3)
    cache_budget = max(0, ram - reserve - kv_bytes)
    slots = int(cache_budget // expert_bytes) if expert_bytes else 0
    return {
        "model": model,
        "file_size_bytes": file_size,
        "layout_total_bytes": L["total"],
        "arch": {"n_layers": c.n_layers, "dim": c.dim, "n_heads": c.n_heads, "n_kv_heads": c.n_kv_heads,
                 "head_dim": c.head_dim, "vocab": c.vocab, "max_seq": c.max_seq},
        "moe": {"n_experts": c.n_experts, "top_k": c.top_k, "expert_hidden": c.expert_hidden, "total_experts": total_experts},
        "dtype": "bf16" if c.dtype == rbt2.DTYPE_BF16 else "f32",
        "qk_norm": bool(c.qk_norm), "norm_topk": bool(c.norm_topk),
        "expert_blob_bytes": expert_bytes,
        "active_expert_read_bytes_per_token": active_expert_bytes,
        "kv_cache_bytes_estimate": kv_bytes,
        "ctx": ctx,
        "guidance": {"ram_gib": ram_gib, "reserve_gib": 1.5, "expert_cache_payload_bytes": int(cache_budget),
                     "expert_cache_payload_mib": round(mib(cache_budget), 1), "expert_cache_slots": slots},
        "notes": ["mmap uses OS page cache", "--expert-cache-mib budgets payload only", "Windows/runtime not fully validated"],
    }


def print_plan(p):
    print(f"model: {p['model']}")
    print(f"file size: {gib(p['file_size_bytes']):.2f} GiB ({p['file_size_bytes']:,} bytes)  layout total: {gib(p['layout_total_bytes']):.2f} GiB")
    a = p["arch"]
    print(f"arch: layers={a['n_layers']} dim={a['dim']} heads={a['n_heads']}/{a['n_kv_heads']} head_dim={a['head_dim']}")
    m = p["moe"]
    print(f"moe: experts/layer={m['n_experts']} top_k={m['top_k']} expert_hidden={m['expert_hidden']} total experts={m['total_experts']}")
    print(f"dtype: {p['dtype']}  qk_norm={p['qk_norm']} norm_topk={p['norm_topk']} max_seq={a['max_seq']}")
    print(f"expert blob: {mib(p['expert_blob_bytes']):.2f} MiB each (aligned)")
    print(f"per token (all selected experts, if uncached): ~{mib(p['active_expert_read_bytes_per_token']):.1f} MiB read across layers")
    print(f"rough KV cache at ctx={p['ctx']}: ~{mib(p['kv_cache_bytes_estimate']):.1f} MiB (f32, excludes activations/runtime)")
    g = p["guidance"]
    print(f"guidance (--ram-gib {g['ram_gib']:g}): reserve ~1.5 GiB + KV => expert-cache payload budget ~{g['expert_cache_payload_mib']:.0f} MiB ({g['expert_cache_slots']} expert slots)")
    for n in p["notes"]:
        print(f"note: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help=".rbt2 model file")
    ap.add_argument("--ram-gib", type=float, default=8.0, help="assumed system RAM for guidance")
    ap.add_argument("--ctx", type=int, default=512, help="context length for rough KV estimate")
    ap.add_argument("--json", action="store_true", dest="as_json", help="output machine-readable JSON")
    args = ap.parse_args()
    try:
        plan = compute_plan(args.model, args.ram_gib, args.ctx)
    except Exception as e:
        sys.exit(f"error: {e}")
    if args.as_json:
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_plan(plan)


if __name__ == "__main__":
    main()
