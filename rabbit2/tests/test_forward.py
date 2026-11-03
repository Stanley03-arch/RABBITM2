"""
Compares the C engine against the NumPy reference on random tiny MoE models.

    python tests/test_forward.py

Checks, for several architecture shapes (f32 and bf16 weights, QK-norm on/off, top-k renorm on/off):
  * expert routing matches exactly
  * logits match within tolerance (fp32 engine vs float64 reference)
  * greedy generation (the --gen path, incremental KV cache) matches the reference
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import rbt2  # noqa: E402
import reference as ref  # noqa: E402

ATOL = 1e-4


def find_or_build_binary():
    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    path = os.path.join(ROOT, "build", exe)
    src = os.path.join(ROOT, "src", "rabbit2.c")
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src):
        return path
    cc = next((c for c in ("gcc", "clang", "cc") if shutil.which(c)), None)
    if cc is None:
        sys.exit("No C compiler found. Install gcc/clang (MSYS2 MinGW-w64 on Windows) or build manually.")
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    cmd = [cc, "-O2", "-std=c99", "-Wall", "-Wextra", "-o", path, src] + ([] if os.name == "nt" else ["-lm"])
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return path


def run_case(binary, cfg, seed, n_tokens, tmp):
    rng = np.random.default_rng(1000 + seed)
    tokens = rng.integers(0, cfg.vocab, size=n_tokens).tolist()

    weights = ref.make_weights(cfg, seed)
    model_path = os.path.join(tmp, "m.rbt2")
    ref.write_model(model_path, cfg, weights)
    want_logits, want_routing = ref.forward(cfg, weights, tokens)

    out_l, out_r = os.path.join(tmp, "logits.bin"), os.path.join(tmp, "routing.bin")
    subprocess.run([binary, model_path, "--dump", out_l, out_r, *map(str, tokens)], check=True,
                   stderr=subprocess.DEVNULL)
    got_logits = np.fromfile(out_l, dtype="<f4").reshape(n_tokens, cfg.vocab)
    got_routing = np.fromfile(out_r, dtype="<i4").reshape(n_tokens, cfg.n_layers, cfg.top_k)

    routing_ok = np.array_equal(got_routing, want_routing)
    max_diff = float(np.max(np.abs(got_logits - want_logits)))
    argmax_ok = np.array_equal(got_logits.argmax(1), want_logits.argmax(1))

    # greedy generation through the incremental (--gen) path
    n_new = min(6, cfg.max_seq - n_tokens)
    res = subprocess.run([binary, model_path, "--gen", str(n_new), *map(str, tokens)], check=True,
                         capture_output=True, text=True)
    got_gen = [int(t) for t in res.stdout.split()]
    want_gen = ref.greedy(cfg, weights, tokens, n_new)
    gen_ok = got_gen == want_gen

    ok = routing_ok and max_diff < ATOL and argmax_ok and gen_ok
    print(f"  seed={seed}  routing={'match' if routing_ok else 'DIFFER'}  max|dlogit|={max_diff:.2e}  "
          f"argmax={'match' if argmax_ok else 'DIFFER'}  gen={'match' if gen_ok else 'DIFFER'}  "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def main():
    binary = find_or_build_binary()
    C = rbt2.Config
    shapes = {
        "f32, GQA 4q/2kv, 8 experts top-2, renorm": C(),
        "bf16 weights, GQA, renorm": C(dtype=rbt2.DTYPE_BF16),
        "f32, QK-norm, top-k NOT renormalised (OLMoE style)": C(qk_norm=1, norm_topk=0),
        "bf16 + QK-norm + no renorm, MHA 4q/4kv, top-3 of 6 (OLMoE-like)": C(
            vocab=100, dim=48, n_layers=3, n_heads=4, n_kv_heads=4, head_dim=12, n_experts=6, top_k=3,
            expert_hidden=40, max_seq=32, dtype=rbt2.DTYPE_BF16, qk_norm=1, norm_topk=0),
        "odd shape: 6q/3kv, odd hidden size, f32": C(
            vocab=70, dim=36, n_layers=2, n_heads=6, n_kv_heads=3, head_dim=6, n_experts=5, top_k=2,
            expert_hidden=25, max_seq=32),
        "top-1 routing, MHA": C(vocab=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8,
                                n_experts=4, top_k=1, expert_hidden=64, max_seq=32),
    }
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, cfg in shapes.items():
            print(name)
            for seed in range(3):
                all_ok &= run_case(binary, cfg, seed, n_tokens=min(12, cfg.max_seq - 6), tmp=tmp)
    print("\nALL PASSED" if all_ok else "\nFAILURES")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
