"""
End-to-end test of the Hugging Face converter, without downloading anything.

    python tests/test_convert.py

For a tiny random model it:
  1. writes a fake OLMoE checkpoint in real HF layout (sharded bf16 safetensors + index + config.json)
  2. converts it to .rbt2 (bf16 and f32 output)
  3. runs the C engine and compares against the NumPy reference computed from the ORIGINAL weights
  4. checks the reader against the official `safetensors` package (if installed)
  5. checks that unsupported / broken checkpoints are rejected with a clear error

This proves the converter, the safetensors reader and the engine agree with each other. It does NOT
prove they agree with Hugging Face -- that needs tools/check_vs_hf.py on a machine with torch.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import rbt2  # noqa: E402
import reference as ref  # noqa: E402
import convert_hf  # noqa: E402
import safetensors_min as stm  # noqa: E402
from test_forward import find_or_build_binary  # noqa: E402

ATOL = 1e-4
CFG = rbt2.Config(vocab=100, dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=16, n_experts=8,
                  top_k=2, expert_hidden=32, max_seq=64, dtype=rbt2.DTYPE_BF16, qk_norm=1, norm_topk=0)


def bf16_weights(seed):
    """Random weights where EVERY tensor is exactly representable in bf16 (like a real bf16 checkpoint)."""
    w = ref.make_weights(CFG, seed)  # matrices already bf16-rounded
    rv = lambda a: rbt2.round_to_dtype(a, rbt2.DTYPE_BF16)
    w["final_norm"] = rv(w["final_norm"])
    for ly in w["layers"]:
        for k in ("attn_norm", "ffn_norm", "q_norm", "k_norm"):
            ly[k] = rv(ly[k])
    return w


def hf_tensors(w):
    t = {"model.embed_tokens.weight": w["tok_emb"], "model.norm.weight": w["final_norm"],
         "lm_head.weight": w["lm_head"]}
    for l, ly in enumerate(w["layers"]):
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = ly["attn_norm"]
        t[p + "post_attention_layernorm.weight"] = ly["ffn_norm"]
        t[p + "self_attn.q_proj.weight"] = ly["wq"]
        t[p + "self_attn.k_proj.weight"] = ly["wk"]
        t[p + "self_attn.v_proj.weight"] = ly["wv"]
        t[p + "self_attn.o_proj.weight"] = ly["wo"]
        t[p + "self_attn.q_norm.weight"] = ly["q_norm"]
        t[p + "self_attn.k_norm.weight"] = ly["k_norm"]
        t[p + "mlp.gate.weight"] = ly["router"]
        for e, (w1, w3, w2) in enumerate(ly["experts"]):
            t[p + f"mlp.experts.{e}.gate_proj.weight"] = w1
            t[p + f"mlp.experts.{e}.up_proj.weight"] = w3
            t[p + f"mlp.experts.{e}.down_proj.weight"] = w2
    return t


def hf_config(**over):
    c = {"architectures": ["OlmoeForCausalLM"], "model_type": "olmoe", "hidden_size": CFG.dim,
         "intermediate_size": CFG.expert_hidden, "num_hidden_layers": CFG.n_layers,
         "num_attention_heads": CFG.n_heads, "num_key_value_heads": CFG.n_kv_heads,
         "num_experts": CFG.n_experts, "num_experts_per_tok": CFG.top_k, "vocab_size": CFG.vocab,
         "max_position_embeddings": CFG.max_seq, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
         "norm_topk_prob": False, "clip_qkv": None, "rope_scaling": None, "attention_bias": False,
         "hidden_act": "silu", "tie_word_embeddings": False}
    c.update(over)
    return c


def write_checkpoint(d, tensors, cfg_over=None, shards=2):
    os.makedirs(d, exist_ok=True)
    names = list(tensors)
    groups = [names[i::shards] for i in range(shards)]  # interleave so shards are genuinely mixed
    weight_map = {}
    for i, g in enumerate(groups):
        fn = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        stm.write_safetensors(os.path.join(d, fn), {n: tensors[n] for n in g}, dtype="BF16")
        weight_map.update({n: fn for n in g})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(hf_config(**(cfg_over or {})), f)


def run_engine(binary, model, tokens, tmp):
    out_l, out_r = os.path.join(tmp, "l.bin"), os.path.join(tmp, "r.bin")
    subprocess.run([binary, model, "--dump", out_l, out_r, *map(str, tokens)], check=True,
                   stderr=subprocess.DEVNULL)
    n = len(tokens)
    return (np.fromfile(out_l, dtype="<f4").reshape(n, CFG.vocab),
            np.fromfile(out_r, dtype="<i4").reshape(n, CFG.n_layers, CFG.top_k))


def expect_error(label, fn, needle):
    try:
        fn()
    except convert_hf.ConvertError as e:
        ok = needle in str(e)
        print(f"  {label}: rejected{'' if ok else ' but message lacks ' + repr(needle)} -> {'ok' if ok else 'FAIL'}")
        return ok
    print(f"  {label}: NOT rejected -> FAIL")
    return False


def main():
    binary = find_or_build_binary()
    ok_all = True
    tokens = np.random.default_rng(7).integers(0, CFG.vocab, size=10).tolist()

    with tempfile.TemporaryDirectory() as tmp:
        print("conversion + engine vs reference")
        for seed in range(2):
            w = bf16_weights(seed)
            hf_dir = os.path.join(tmp, f"hf{seed}")
            write_checkpoint(hf_dir, hf_tensors(w))
            want_logits, want_routing = ref.forward(CFG, w, tokens)
            for dt in ("bf16", "f32"):
                out = os.path.join(tmp, f"m_{seed}_{dt}.rbt2")
                convert_hf.convert(hf_dir, out, dt, log=lambda *_: None)
                got_logits, got_routing = run_engine(binary, out, tokens, tmp)
                r_ok = np.array_equal(got_routing, want_routing)
                diff = float(np.max(np.abs(got_logits - want_logits)))
                ok = r_ok and diff < ATOL
                ok_all &= ok
                print(f"  seed={seed} out={dt:4s} routing={'match' if r_ok else 'DIFFER'} "
                      f"max|dlogit|={diff:.2e} -> {'ok' if ok else 'FAIL'}")

        print("tied embeddings (no lm_head.weight in checkpoint)")
        w = bf16_weights(5)
        w["lm_head"] = w["tok_emb"].copy()
        t = hf_tensors(w)
        del t["lm_head.weight"]
        d = os.path.join(tmp, "hf_tied")
        write_checkpoint(d, t, {"tie_word_embeddings": True}, shards=1)
        out = os.path.join(tmp, "tied.rbt2")
        convert_hf.convert(d, out, "bf16", log=lambda *_: None)
        got, _ = run_engine(binary, out, tokens, tmp)
        diff = float(np.max(np.abs(got - ref.forward(CFG, w, tokens)[0])))
        ok_all &= diff < ATOL
        print(f"  max|dlogit|={diff:.2e} -> {'ok' if diff < ATOL else 'FAIL'}")

        print("reader vs official safetensors package")
        try:
            from safetensors.numpy import save_file
            arrs = {"a": np.random.default_rng(1).standard_normal((5, 7)).astype(np.float32),
                    "b.c": np.arange(12, dtype=np.float32).reshape(3, 4)}
            p = os.path.join(tmp, "official.safetensors")
            save_file(arrs, p)
            sf = stm.SafetensorsFile(p)
            same = all(np.array_equal(sf.get(k), v) for k, v in arrs.items())
            ok_all &= same
            print(f"  f32 file written by safetensors, read by ours: {'identical -> ok' if same else 'DIFFER -> FAIL'}")
        except ImportError:
            print("  (safetensors package not installed; skipped. pip install safetensors to enable)")

        print("bad checkpoints are rejected")
        w = bf16_weights(0)
        full = hf_tensors(w)

        d = os.path.join(tmp, "bad_type")
        write_checkpoint(d, full, {"model_type": "llama"}, shards=1)
        ok_all &= expect_error("wrong model_type", lambda: convert_hf.convert(d, os.path.join(tmp, "x"), "bf16", lambda *_: None),
                               "only 'olmoe'")

        d = os.path.join(tmp, "bad_missing")
        t = dict(full)
        del t["model.layers.1.mlp.experts.3.up_proj.weight"]
        write_checkpoint(d, t, shards=1)
        ok_all &= expect_error("missing tensor", lambda: convert_hf.convert(d, os.path.join(tmp, "x"), "bf16", lambda *_: None),
                               "missing")

        d = os.path.join(tmp, "bad_shape")
        t = dict(full)
        # (q_proj is square here, so a transpose would be invisible to a shape check; use a non-square tensor)
        t["model.layers.0.mlp.gate.weight"] = t["model.layers.0.mlp.gate.weight"].T.copy()
        write_checkpoint(d, t, shards=1)
        ok_all &= expect_error("transposed tensor", lambda: convert_hf.convert(d, os.path.join(tmp, "x"), "bf16", lambda *_: None),
                               "shape mismatches")

        d = os.path.join(tmp, "bad_quant")
        write_checkpoint(d, full, {"quantization_config": {"quant_method": "bitsandbytes"}}, shards=1)
        ok_all &= expect_error("already-quantized checkpoint", lambda: convert_hf.convert(d, os.path.join(tmp, "x"), "bf16", lambda *_: None),
                               "already quantized")

    print("\nALL PASSED" if ok_all else "\nFAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
