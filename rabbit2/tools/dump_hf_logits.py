"""
Run the real Hugging Face model and save logits + expert routing for one prompt.

Use this on a machine that has enough memory for the model (a bigger PC, or Google Colab with a GPU),
then copy the small .npz file to your laptop and compare with:

    python tools/check_vs_hf.py --ref hf_ref.npz olmoe.rbt2 --atol 0.1

    pip install torch transformers accelerate
    python tools/dump_hf_logits.py allenai/OLMoE-1B-7B-0924 hf_ref.npz --dtype bfloat16 --device auto

--dtype   float32 is the most precise (needs ~28 GB RAM for OLMoE-1B-7B). bfloat16 needs ~14 GB.
          If bfloat16 fails on an older GPU (e.g. Colab T4), use float16.
--device  auto = let accelerate place the model on the GPU(s); cpu = plain CPU.
The prompt is tokenised here, so the saved file holds the exact ids the engine must use.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Hugging Face id or local folder")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", type=int, nargs="*", help="use these token ids instead of --prompt")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    kw = {"device_map": "auto"} if args.device == "auto" else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kw)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kw)
    model.eval()

    if args.ids:
        ids = args.ids
    else:
        ids = AutoTokenizer.from_pretrained(args.model)(args.prompt)["input_ids"]
    print("token ids:", ids)

    with torch.no_grad():
        out = model(torch.tensor([ids], device=model.device), output_router_logits=True)
    logits = out.logits[0].float().cpu().numpy()
    k = model.config.num_experts_per_tok
    routing = np.zeros((0,), dtype=np.int32)
    if getattr(out, "router_logits", None):
        routing = np.stack([torch.topk(torch.softmax(r.float(), dim=-1), k, dim=-1).indices.cpu().numpy()
                            for r in out.router_logits], axis=1).astype(np.int32)  # [T, n_layers, k]
    np.savez(args.out, ids=np.array(ids, dtype=np.int64), logits=logits, routing=routing)
    print(f"saved {args.out}: logits {logits.shape}, routing {routing.shape}, dtype used {args.dtype}")


if __name__ == "__main__":
    main()
