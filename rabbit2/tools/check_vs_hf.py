"""
Compare the RABBIT2 engine with the real Hugging Face model. This is the gate for milestone 2.

Two ways to get the Hugging Face side:

  --hf <hf_dir>   load the HF model here and run it (needs torch + transformers and enough RAM:
                  float32 needs ~4 bytes/param, so only small models; fine for the tiny test model)
  --ref <file.npz> use logits saved earlier by tools/dump_hf_logits.py on a bigger machine / Colab
                  (this is how you check the real 7B OLMoE from an 8 GB laptop)

    python tools/check_vs_hf.py --hf tiny_olmoe tiny.rbt2
    python tools/check_vs_hf.py --ref hf_ref.npz olmoe.rbt2 --atol 0.1

Expected: tiny random model, f32 .rbt2        -> max|dlogit| around 1e-5, routing 100% identical
          real OLMoE, bf16 .rbt2 vs a bf16 or fp16 reference -> differences of 1e-2..1e-1 are normal;
          what matters is argmax agreement and nearly all routing identical.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_bin(arg):
    if arg:
        return arg
    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    path = os.path.join(ROOT, "build", exe)
    if not os.path.exists(path):
        sys.exit(f"engine not built: {path}\nbuild it first (see README), or pass --bin PATH")
    return path


def engine_info(binary, model):
    out = subprocess.run([binary, model, "--info"], capture_output=True, text=True, check=True).stdout
    return {k: int(v) for line in out.splitlines() for k, v in (t.split("=") for t in line.split())}


def hf_reference(hf_dir, ids_arg, vocab):
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32)
    model.eval()
    ids = ids_arg or np.random.default_rng(0).integers(0, vocab, size=12).tolist()
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_router_logits=True)
    k = model.config.num_experts_per_tok
    routing = None
    if getattr(out, "router_logits", None):
        routing = np.stack([torch.topk(torch.softmax(r.float(), dim=-1), k, dim=-1).indices.numpy()
                            for r in out.router_logits], axis=1)  # [T, n_layers, k]
    return ids, out.logits[0].float().numpy(), routing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", metavar="DIR", help="Hugging Face checkpoint folder to run here")
    src.add_argument("--ref", metavar="NPZ", help="reference file from tools/dump_hf_logits.py")
    ap.add_argument("rbt2", help="converted model")
    ap.add_argument("--ids", type=int, nargs="*", help="token ids (only with --hf; default 12 random ids)")
    ap.add_argument("--atol", type=float, default=1e-3, help="max allowed |logit difference|")
    ap.add_argument("--bin", default=None)
    args = ap.parse_args()
    binary = find_bin(args.bin)
    info = engine_info(binary, args.rbt2)
    vocab, n_layers, k = info["vocab"], info["n_layers"], info["top_k"]

    if args.ref:
        z = np.load(args.ref)
        ids, hf_logits = z["ids"].tolist(), z["logits"].astype(np.float32)
        hf_routing = z["routing"] if "routing" in z.files and z["routing"].size else None
    else:
        ids, hf_logits, hf_routing = hf_reference(args.hf, args.ids, vocab)
    T = len(ids)
    if hf_logits.shape != (T, vocab):
        sys.exit(f"reference logits have shape {hf_logits.shape}, expected {(T, vocab)}: wrong model?")

    with tempfile.TemporaryDirectory() as tmp:
        fl, fr = os.path.join(tmp, "l.bin"), os.path.join(tmp, "r.bin")
        subprocess.run([binary, args.rbt2, "--dump", fl, fr, "--ctx", str(T), *map(str, ids)], check=True)
        c_logits = np.fromfile(fl, dtype="<f4").reshape(T, vocab)
        c_routing = np.fromfile(fr, dtype="<i4").reshape(T, n_layers, k)

    diff = np.abs(c_logits - hf_logits)
    argmax_ok = c_logits.argmax(1) == hf_logits.argmax(1)
    print(f"tokens: {T}   vocab: {vocab}   logit scale (max |HF logit|): {np.abs(hf_logits).max():.2f}")
    print("max |logit diff| per position:", " ".join(f"{d:.1e}" for d in diff.max(axis=1)))
    print(f"max |logit diff| overall: {diff.max():.3e}   mean: {diff.mean():.3e}")
    print(f"argmax agrees at {argmax_ok.sum()}/{T} positions")

    routing_ok = True
    if hf_routing is not None:
        same = (np.sort(c_routing, axis=2) == np.sort(np.asarray(hf_routing), axis=2)).all(axis=2)
        routing_ok = bool(same.all())
        print(f"expert routing identical at {same.sum()}/{same.size} (token, layer) pairs")
        if not routing_ok:
            t, l = np.argwhere(~same)[0]
            print(f"  first difference: token {t}, layer {l}: engine {sorted(c_routing[t, l].tolist())} "
                  f"vs HF {sorted(np.asarray(hf_routing)[t, l].tolist())}")
    else:
        print("(no router logits in the reference; routing not compared)")

    ok = diff.max() < args.atol and argmax_ok.all() and routing_ok
    print("\nPASS" if ok else "\nMISMATCH")
    if not ok:
        print("Hints: a diff that is large from position 0 points at weights / layout / QK-norm / routing;\n"
              "       a diff that grows with position points at RoPE or the KV cache;\n"
              "       small diffs (<= ~1e-1) against a bf16/fp16 reference can be normal: retry with a larger --atol,\n"
              "       and look mainly at argmax agreement and routing.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
