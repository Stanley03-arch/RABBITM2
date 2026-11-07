"""Check local RABBIT2 setup: Python deps, engine binary, and basic model sanity."""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ok(msg):
    print(f"[ok]   {msg}")


def warn(msg):
    print(f"[warn] {msg}")


def fail(msg):
    print(f"[fail] {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="optional .rbt2 model to inspect")
    ap.add_argument("--bin", help="optional engine binary path")
    args = ap.parse_args()

    py = sys.version.split()[0]
    ok(f"Python {py}")
    try:
        import numpy  # noqa: F401
        ok(f"numpy {numpy.__version__}")
    except Exception as e:
        fail(f"numpy missing/broken: {e}")
    for mod in ("transformers", "torch", "huggingface_hub"):
        try:
            m = __import__(mod)
            ok(f"{mod} {getattr(m, '__version__', 'installed')}")
        except Exception:
            warn(f"{mod} not installed (optional; needed for tokenizer/HF conversion/validation)")

    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    binary = args.bin or os.path.join(ROOT, "build", exe)
    if os.path.exists(binary):
        ok(f"engine binary: {binary}")
        try:
            r = subprocess.run([binary, "--help"], capture_output=True, text=True)
            # engine prints usage to stderr on bad args; --help not a real flag but should not crash system
            out = (r.stdout or "") + (r.stderr or "")
            if "usage" in out.lower() or r.returncode in (0, 1, 2):
                ok("engine responds")
            else:
                warn(f"engine returned rc={r.returncode}")
        except Exception as e:
            warn(f"could not run engine: {e}")
    else:
        warn(f"engine not built at {binary} (build with gcc/clang; see README)")

    cc = next((c for c in ("gcc", "clang", "cc", "cl") if shutil.which(c)), None)
    if cc:
        ok(f"C compiler found: {cc}")
    else:
        warn("no C compiler found on PATH (MSYS2 MinGW-w64 gcc recommended on Windows)")

    if args.model:
        if not os.path.exists(args.model):
            fail(f"model not found: {args.model}")
        else:
            ok(f"model file exists: {args.model} ({os.path.getsize(args.model):,} bytes)")
            if os.path.exists(binary):
                try:
                    out = subprocess.run([binary, args.model, "--info"], capture_output=True, text=True, check=True)
                    ok("model header parsed by engine")
                    for line in out.stdout.strip().splitlines():
                        print("       " + line)
                except Exception as e:
                    fail(f"engine could not read model: {e}")
            else:
                try:
                    sys.path.insert(0, os.path.join(ROOT, "ref"))
                    import rbt2
                    with open(args.model, "rb") as f:
                        c = rbt2.unpack_header(f.read(rbt2.HEADER_BYTES))
                    ok(f"header parsed (Python): layers={c.n_layers} dim={c.dim} vocab={c.vocab} experts={c.n_experts} top_k={c.top_k} dtype={c.dtype}")
                except Exception as e:
                    fail(f"could not parse model header: {e}")


if __name__ == "__main__":
    main()
