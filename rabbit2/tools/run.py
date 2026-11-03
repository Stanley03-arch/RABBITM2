"""
Generate text with a .rbt2 model (greedy decoding).

    pip install transformers
    python tools/run.py --model olmoe.rbt2 --tok allenai/OLMoE-1B-7B-0924 \
                        --prompt "The capital of France is" --n 30

--tok may be a Hugging Face id (downloads only the small tokenizer files) or a local folder.
Without a tokenizer you can pass raw ids:   python tools/run.py --model m.rbt2 --ids 5 17 42 --n 10

First run on a 14 GB model is slow because the OS has to read experts from disk; later tokens reuse
whatever the OS cache kept. Expect seconds per token on an 8 GB laptop at this stage.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", help="tokenizer: Hugging Face id or local folder")
    ap.add_argument("--prompt", help="text prompt (needs --tok)")
    ap.add_argument("--ids", type=int, nargs="*", help="raw prompt token ids (instead of --prompt)")
    ap.add_argument("--n", type=int, default=30, help="max new tokens")
    ap.add_argument("--ctx", type=int, default=0, help="context cap (default: prompt + n)")
    ap.add_argument("--bin", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    binary = args.bin or os.path.join(ROOT, "build", exe)
    if not os.path.exists(binary):
        sys.exit(f"engine not built: {binary}")

    tok, eos = None, -1
    if args.ids:
        ids = args.ids
    elif args.prompt and args.tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tok)
        ids = tok(args.prompt)["input_ids"]
        eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    else:
        sys.exit("give either --prompt with --tok, or --ids")

    ctx = args.ctx or (len(ids) + args.n)
    cmd = [binary, args.model, "--gen", str(args.n), "--ctx", str(ctx)]
    if eos >= 0:
        cmd += ["--eos", str(eos)]
    cmd += [str(i) for i in ids]

    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)
    new_ids = [int(t) for t in res.stdout.split()]

    if tok is not None:
        print(args.prompt + tok.decode(new_ids, skip_special_tokens=True))
    else:
        print("generated ids:", new_ids)


if __name__ == "__main__":
    main()
