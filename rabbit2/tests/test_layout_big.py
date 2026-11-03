"""
Checks that the C engine and ref/rbt2.py compute identical byte offsets for a full-size OLMoE-1B-7B
file (about 13.8 GB, so offsets go far beyond 4 GiB where 32-bit arithmetic bugs would show up).

    python tests/test_layout_big.py

Creates a SPARSE file containing only the header (no 14 GB of data is written), asks the engine to
parse it with --info, and compares every offset. Skipped automatically if the disk is too full.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import rbt2  # noqa: E402
from test_forward import find_or_build_binary  # noqa: E402

OLMOE = rbt2.Config(vocab=50304, dim=2048, n_layers=16, n_heads=16, n_kv_heads=16, head_dim=128,
                    n_experts=64, top_k=8, expert_hidden=1024, max_seq=4096, dtype=rbt2.DTYPE_BF16,
                    qk_norm=1, norm_topk=0, rope_theta=10000.0, rms_eps=1e-5)


def main():
    binary = find_or_build_binary()
    L = rbt2.layout(OLMOE)
    need = L["total"]
    with tempfile.TemporaryDirectory() as tmp:
        # On Linux the file is genuinely sparse. On Windows/NTFS the space may be reserved, so be careful.
        # Set RABBIT2_SPARSE_OK=1 to skip the disk check if you know your filesystem handles sparse files.
        if not os.environ.get("RABBIT2_SPARSE_OK") and shutil.disk_usage(tmp).free < 1.2 * need:
            print(f"skipped: need ~{1.2 * need / 1e9:.0f} GB free in the temp folder "
                  "(set RABBIT2_SPARSE_OK=1 to override)")
            return
        path = os.path.join(tmp, "olmoe_sparse.rbt2")
        rbt2.Rbt2Writer(path, OLMOE).close()  # header + sparse tail
        res = subprocess.run([binary, path, "--info"], capture_output=True, text=True)
        if res.returncode != 0:
            print("engine rejected the file:", res.stderr)
            sys.exit(1)

    kv = dict(tok.split("=") for line in res.stdout.splitlines() for tok in line.split())
    want = {
        "tok_emb": L["tok_emb"], "layer0_experts": L["layers"][0]["experts"],
        "last_layer_experts": L["layers"][-1]["experts"], "expert_stride": L["expert_stride"],
        "lm_head": L["lm_head"],
    }
    ok = True
    print(f"model file size: {need / 1e9:.2f} GB  (largest offset {need:,} > 2^32 = {2**32:,})")
    for k, v in want.items():
        good = int(kv[k]) == v
        ok &= good
        print(f"  {k:20s} python={v:>14,}  c={int(kv[k]):>14,}  {'ok' if good else 'MISMATCH'}")
    end_ok = int(kv["layout_end"]) == L["total"]
    ok &= end_ok
    print(f"  {'layout_end':20s} python={L['total']:>14,}  c={int(kv['layout_end']):>14,}  {'ok' if end_ok else 'MISMATCH'}")
    cfg_ok = (int(kv["n_experts"]) == 64 and int(kv["top_k"]) == 8 and int(kv["dim"]) == 2048
              and int(kv["dtype"]) == 1 and int(kv["qk_norm"]) == 1 and int(kv["norm_topk"]) == 0)
    ok &= cfg_ok
    print(f"  header fields parsed correctly: {'ok' if cfg_ok else 'MISMATCH'}")
    print("\nALL PASSED" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
