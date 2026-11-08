"""Check local RABBIT2 setup: Python deps, engine binary, and basic model sanity."""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _record(items, level, msg):
    items.append({"level": level, "message": msg})
    tag = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[fail]"}.get(level, "[info]")
    print(f"{tag} {msg}")


def check(args):
    items = []
    py = sys.version.split()[0]
    _record(items, "ok", f"Python {py}")
    try:
        import numpy  # noqa: F401
        _record(items, "ok", f"numpy {numpy.__version__}")
    except Exception as e:
        _record(items, "fail", f"numpy missing/broken: {e}")
    for mod in ("transformers", "torch", "huggingface_hub"):
        try:
            m = __import__(mod)
            _record(items, "ok", f"{mod} {getattr(m, '__version__', 'installed')}")
        except Exception:
            _record(items, "warn", f"{mod} not installed (optional; needed for tokenizer/HF conversion/validation)")

    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    binary = args.bin or os.path.join(ROOT, "build", exe)
    if os.path.exists(binary):
        _record(items, "ok", f"engine binary: {binary}")
        try:
            r = subprocess.run([binary, "--help"], capture_output=True, text=True)
            out = (r.stdout or "") + (r.stderr or "")
            if "usage" in out.lower() or r.returncode in (0, 1, 2):
                _record(items, "ok", "engine responds")
            else:
                _record(items, "warn", f"engine returned rc={r.returncode}")
        except Exception as e:
            _record(items, "warn", f"could not run engine: {e}")
    else:
        _record(items, "warn", f"engine not built at {binary} (build with gcc/clang; see README)")

    cc = next((c for c in ("gcc", "clang", "cc", "cl") if shutil.which(c)), None)
    if cc:
        _record(items, "ok", f"C compiler found: {cc}")
    else:
        _record(items, "warn", "no C compiler found on PATH (MSYS2 MinGW-w64 gcc recommended on Windows)")

    model_info = None
    if args.model:
        if not os.path.exists(args.model):
            _record(items, "fail", f"model not found: {args.model}")
        else:
            _record(items, "ok", f"model file exists: {args.model} ({os.path.getsize(args.model):,} bytes)")
            if os.path.exists(binary):
                try:
                    out = subprocess.run([binary, args.model, "--info"], capture_output=True, text=True, check=True)
                    _record(items, "ok", "model header parsed by engine")
                    model_info = out.stdout.strip()
                    if not args.as_json:
                        for line in model_info.splitlines():
                            print("       " + line)
                except Exception as e:
                    _record(items, "fail", f"engine could not read model: {e}")
            else:
                try:
                    sys.path.insert(0, os.path.join(ROOT, "ref"))
                    import rbt2
                    with open(args.model, "rb") as f:
                        c = rbt2.unpack_header(f.read(rbt2.HEADER_BYTES))
                    model_info = {"n_layers": c.n_layers, "dim": c.dim, "vocab": c.vocab, "n_experts": c.n_experts, "top_k": c.top_k, "dtype": c.dtype}
                    _record(items, "ok", f"header parsed (Python): layers={c.n_layers} dim={c.dim} vocab={c.vocab} experts={c.n_experts} top_k={c.top_k} dtype={c.dtype}")
                except Exception as e:
                    _record(items, "fail", f"could not parse model header: {e}")
    return {"checks": items, "model": model_info, "ok": not any(i["level"] == "fail" for i in items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="optional .rbt2 model to inspect")
    ap.add_argument("--bin", help="optional engine binary path")
    ap.add_argument("--json", action="store_true", dest="as_json", help="output machine-readable JSON")
    args = ap.parse_args()
    if args.as_json:
        # suppress text records in JSON mode by capturing stdout? simpler: re-run logic cleanly
        items = []
        def rec(level, msg): items.append({"level": level, "message": msg})
        py = sys.version.split()[0]
        rec("ok", f"Python {py}")
        try:
            import numpy
            rec("ok", f"numpy {numpy.__version__}")
        except Exception as e:
            rec("fail", f"numpy missing/broken: {e}")
        for mod in ("transformers", "torch", "huggingface_hub"):
            try:
                m = __import__(mod)
                rec("ok", f"{mod} {getattr(m, '__version__', 'installed')}")
            except Exception:
                rec("warn", f"{mod} not installed (optional; needed for tokenizer/HF conversion/validation)")
        exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
        binary = args.bin or os.path.join(ROOT, "build", exe)
        if os.path.exists(binary):
            rec("ok", f"engine binary: {binary}")
        else:
            rec("warn", f"engine not built at {binary}")
        cc = next((c for c in ("gcc", "clang", "cc", "cl") if shutil.which(c)), None)
        rec("ok" if cc else "warn", f"C compiler found: {cc}" if cc else "no C compiler found on PATH")
        model_info = None
        if args.model and os.path.exists(args.model):
            sys.path.insert(0, os.path.join(ROOT, "ref"))
            try:
                import rbt2
                with open(args.model, "rb") as f:
                    c = rbt2.unpack_header(f.read(rbt2.HEADER_BYTES))
                model_info = {"n_layers": c.n_layers, "dim": c.dim, "vocab": c.vocab, "n_experts": c.n_experts, "top_k": c.top_k, "dtype": c.dtype}
                rec("ok", "model header parsed")
            except Exception as e:
                rec("fail", f"could not parse model header: {e}")
        elif args.model:
            rec("fail", f"model not found: {args.model}")
        json.dump({"checks": items, "model": model_info, "ok": not any(i["level"] == "fail" for i in items)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    check(args)


if __name__ == "__main__":
    main()
