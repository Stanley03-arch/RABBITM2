"""Interactive chat loop for a local .rbt2 model (greedy, via engine).

Requires a built engine binary and, for text prompts, transformers + tokenizer.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_gen(binary, model, ids, n, ctx, eos, extra):
    cmd = [binary, model, "--gen", str(n), "--ctx", str(ctx)] + extra
    if eos >= 0:
        cmd += ["--eos", str(eos)]
    cmd += [str(i) for i in ids]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or f"engine exited {res.returncode}").strip())
    sys.stderr.write(res.stderr)
    new_ids = [int(t) for t in res.stdout.split() if t.strip()]
    return new_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True, help="tokenizer: Hugging Face id or local folder")
    ap.add_argument("--n", type=int, default=64, help="max new tokens per reply")
    ap.add_argument("--ctx", type=int, default=0, help="total context cap (default: prompt+n, clamped to model)")
    ap.add_argument("--system", default="", help="optional text prepended before each user turn")
    ap.add_argument("--expert-cache-mib", type=int, default=None)
    ap.add_argument("--kernel", choices=["scalar", "auto", "avx2"], default=None)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--max-history-tokens", type=int, default=2048, help="trim oldest history when prompt exceeds this")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    binary = args.bin or os.path.join(ROOT, "build", exe)
    if not os.path.exists(binary):
        sys.exit(f"engine not built: {binary}")
    tok = AutoTokenizer.from_pretrained(args.tok)
    eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    extra = []
    if args.expert_cache_mib is not None:
        extra += ["--expert-cache-mib", str(args.expert_cache_mib)]
    if args.kernel is not None:
        extra += ["--kernel", args.kernel]

    history_ids = []
    print("Rabbit2 chat (greedy, stateless per turn via engine; history kept in Python). Ctrl+C to exit.")
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                print()
                break
            if not user.strip():
                continue
            text = (args.system + "\n" if args.system else "") + "User: " + user + "\nAssistant:"
            turn_ids = tok(text, add_special_tokens=bool(not history_ids))["input_ids"]
            prompt_ids = (history_ids + turn_ids)[-args.max_history_tokens:]
            ctx = args.ctx or (len(prompt_ids) + args.n)
            try:
                new_ids = run_gen(binary, args.model, prompt_ids, args.n, ctx, eos, extra)
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)
                continue
            reply = tok.decode(new_ids, skip_special_tokens=True)
            print("bot>", reply.strip())
            history_ids = prompt_ids + new_ids
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
