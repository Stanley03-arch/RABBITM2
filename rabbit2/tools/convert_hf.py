"""
Convert a Hugging Face OLMoE checkpoint (safetensors) into a RABBIT2 .rbt2 file.

    python tools/convert_hf.py <hf_dir> <out.rbt2> [--dtype bf16|f32]

<hf_dir> must contain config.json and either model.safetensors or
model.safetensors.index.json + shards. Tensors are copied one at a time, so memory use stays small
(largest single tensor is the 50304 x 2048 embedding, ~0.4 GB in float32).

Supported: model_type "olmoe" (e.g. allenai/OLMoE-1B-7B-0924). Anything unsupported is rejected
with a clear message rather than converted into silent garbage.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ref"))
sys.path.insert(0, HERE)
import rbt2  # noqa: E402
from safetensors_min import SafetensorsFile  # noqa: E402


class ConvertError(Exception):
    pass


def config_from_hf(hf, dtype):
    problems = []
    if hf.get("model_type") != "olmoe":
        problems.append(f"model_type is {hf.get('model_type')!r}; only 'olmoe' is supported in this milestone")
    if hf.get("clip_qkv") is not None:
        problems.append("clip_qkv is set (not supported)")
    if hf.get("rope_scaling") is not None:
        problems.append("rope_scaling is set (not supported)")
    rope_params = hf.get("rope_parameters") or {}  # newer transformers versions store RoPE settings here
    if rope_params.get("rope_type", "default") != "default":
        problems.append(f"rope_type={rope_params.get('rope_type')!r} (only default RoPE supported)")
    if hf.get("attention_bias"):
        problems.append("attention_bias=true (not supported)")
    if hf.get("hidden_act", "silu") != "silu":
        problems.append(f"hidden_act={hf.get('hidden_act')!r} (only silu supported)")
    if hf.get("quantization_config"):
        problems.append("checkpoint is already quantized; convert the original bf16/fp32 weights instead")
    heads = hf.get("num_attention_heads", 0)
    if not heads or hf.get("hidden_size", 0) % heads:
        problems.append("hidden_size is not divisible by num_attention_heads")
    if problems:
        raise ConvertError("cannot convert:\n  - " + "\n  - ".join(problems))

    return rbt2.Config(
        vocab=hf["vocab_size"], dim=hf["hidden_size"], n_layers=hf["num_hidden_layers"],
        n_heads=heads, n_kv_heads=hf.get("num_key_value_heads", heads),
        head_dim=hf["hidden_size"] // heads,
        n_experts=hf["num_experts"], top_k=hf["num_experts_per_tok"],
        expert_hidden=hf["intermediate_size"], max_seq=hf["max_position_embeddings"],
        dtype=dtype, qk_norm=1, norm_topk=int(bool(hf.get("norm_topk_prob", False))),
        rope_theta=float(hf.get("rope_theta", rope_params.get("rope_theta", 10000.0))), rms_eps=float(hf.get("rms_norm_eps", 1e-5)),
    )


def expected_tensors(cfg):
    """HF tensor name -> expected shape, for every tensor the converter reads (except a tied lm_head)."""
    q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    t = {"model.embed_tokens.weight": (cfg.vocab, cfg.dim), "model.norm.weight": (cfg.dim,),
         "lm_head.weight": (cfg.vocab, cfg.dim)}
    for l in range(cfg.n_layers):
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = (cfg.dim,)
        t[p + "post_attention_layernorm.weight"] = (cfg.dim,)
        t[p + "self_attn.q_proj.weight"] = (q_dim, cfg.dim)
        t[p + "self_attn.k_proj.weight"] = (kv_dim, cfg.dim)
        t[p + "self_attn.v_proj.weight"] = (kv_dim, cfg.dim)
        t[p + "self_attn.o_proj.weight"] = (cfg.dim, q_dim)
        t[p + "self_attn.q_norm.weight"] = (q_dim,)
        t[p + "self_attn.k_norm.weight"] = (kv_dim,)
        t[p + "mlp.gate.weight"] = (cfg.n_experts, cfg.dim)
        for e in range(cfg.n_experts):
            t[p + f"mlp.experts.{e}.gate_proj.weight"] = (cfg.expert_hidden, cfg.dim)
            t[p + f"mlp.experts.{e}.up_proj.weight"] = (cfg.expert_hidden, cfg.dim)
            t[p + f"mlp.experts.{e}.down_proj.weight"] = (cfg.dim, cfg.expert_hidden)
    return t


def open_checkpoint(hf_dir):
    """Return {tensor name: SafetensorsFile}."""
    idx = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {s: SafetensorsFile(os.path.join(hf_dir, s)) for s in sorted(set(weight_map.values()))}
        return {name: shards[s] for name, s in weight_map.items()}
    single = os.path.join(hf_dir, "model.safetensors")
    if os.path.exists(single):
        sf = SafetensorsFile(single)
        return {k: sf for k in sf.keys()}
    raise ConvertError(f"no model.safetensors or model.safetensors.index.json in {hf_dir}")


def convert(hf_dir, out_path, dtype_name="bf16", log=print):
    with open(os.path.join(hf_dir, "config.json")) as f:
        hf = json.load(f)
    dtype = rbt2.DTYPE_BF16 if dtype_name == "bf16" else rbt2.DTYPE_F32
    cfg = config_from_hf(hf, dtype)
    tensors = open_checkpoint(hf_dir)

    want = expected_tensors(cfg)
    tied = "lm_head.weight" not in tensors
    if tied:
        log("note: no lm_head.weight in checkpoint; using embed_tokens as lm_head (tied embeddings)")
    missing = [n for n in want if n not in tensors and not (tied and n == "lm_head.weight")]
    if missing:
        sample = sorted(n for n in tensors if "experts" not in n)[:12]
        raise ConvertError(
            f"{len(missing)} expected tensors are missing, e.g. {missing[:5]}\n"
            f"tensor names in this checkpoint look like: {sample}\n"
            "(a re-saved checkpoint with fused expert weights is not supported yet; use the original files)")
    bad = [(n, tensors[n].shape(n), s) for n, s in want.items()
           if n in tensors and tensors[n].shape(n) != s]
    if bad:
        raise ConvertError("shape mismatches (name, found, expected): " + repr(bad[:5]))

    L = rbt2.layout(cfg)
    gb = L["total"] / 1e9
    log(f"{cfg.n_layers} layers, {cfg.n_experts} experts/layer (top-{cfg.top_k}), dim {cfg.dim}, "
        f"{'bf16' if dtype == rbt2.DTYPE_BF16 else 'f32'} -> {gb:.2f} GB")

    get = lambda n: tensors[n].get(n)
    w = rbt2.Rbt2Writer(out_path, cfg)
    try:
        w.put_mat(L["tok_emb"], get("model.embed_tokens.weight"))
        for l, lo in enumerate(L["layers"]):
            p = f"model.layers.{l}."
            w.put_vec(lo["attn_norm"], get(p + "input_layernorm.weight"))
            w.put_mat(lo["wq"], get(p + "self_attn.q_proj.weight"))
            w.put_mat(lo["wk"], get(p + "self_attn.k_proj.weight"))
            w.put_mat(lo["wv"], get(p + "self_attn.v_proj.weight"))
            w.put_mat(lo["wo"], get(p + "self_attn.o_proj.weight"))
            w.put_vec(lo["q_norm"], get(p + "self_attn.q_norm.weight"))
            w.put_vec(lo["k_norm"], get(p + "self_attn.k_norm.weight"))
            w.put_vec(lo["ffn_norm"], get(p + "post_attention_layernorm.weight"))
            w.put_mat(lo["router"], get(p + "mlp.gate.weight"))
            for e in range(cfg.n_experts):
                q = p + f"mlp.experts.{e}."
                # w1 = gate_proj, w3 = up_proj, w2 = down_proj   (out = down(silu(gate(x)) * up(x)))
                w.put_expert(l, e, get(q + "gate_proj.weight"), get(q + "up_proj.weight"),
                             get(q + "down_proj.weight"))
            log(f"  layer {l + 1}/{cfg.n_layers} done")
        w.put_vec(L["final_norm"], get("model.norm.weight"))
        w.put_mat(L["lm_head"], get("model.embed_tokens.weight" if tied else "lm_head.weight"))
    finally:
        w.close()
    log(f"wrote {out_path}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    args = ap.parse_args()
    try:
        convert(args.hf_dir, args.out, args.dtype)
    except ConvertError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
