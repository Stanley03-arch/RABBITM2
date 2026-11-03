"""
RABBIT2 reference implementation (NumPy, float64 math). The "golden" model the C engine must match.

Architecture (Llama / Mixtral / OLMoE family):
  token embedding
  per layer:  RMSNorm -> attention (GQA, RoPE, causal) -> residual
              RMSNorm -> MoE FFN (softmax router, top-k, SwiGLU experts) -> residual
  final RMSNorm -> lm_head

Switches (from Config):
  qk_norm    RMSNorm over the full q and k projections, before RoPE           (OLMoE: on)
  norm_topk  renormalise the chosen router weights to sum to 1               (OLMoE: off, Mixtral: on)
"""
import numpy as np

import rbt2
from rbt2 import Config  # noqa: F401  (re-exported for convenience)


def make_weights(cfg, seed=0):
    """Random weights, already rounded to what cfg.dtype can represent (so engine and reference agree)."""
    rng = np.random.default_rng(seed)
    rnd = lambda a: rbt2.round_to_dtype(a, cfg.dtype)

    def mat(rows, cols):
        return rnd(rng.standard_normal((rows, cols)) / np.sqrt(cols))

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    w = {"tok_emb": rnd(0.5 * rng.standard_normal((cfg.vocab, cfg.dim))), "layers": []}
    for _ in range(cfg.n_layers):
        ly = {
            "attn_norm": norm(cfg.dim),
            "wq": mat(q_dim, cfg.dim), "wk": mat(kv_dim, cfg.dim), "wv": mat(kv_dim, cfg.dim),
            "wo": mat(cfg.dim, q_dim),
            "ffn_norm": norm(cfg.dim),
            "router": mat(cfg.n_experts, cfg.dim),
            "experts": [(mat(cfg.expert_hidden, cfg.dim),    # w1 (gate)
                         mat(cfg.expert_hidden, cfg.dim),    # w3 (up)
                         mat(cfg.dim, cfg.expert_hidden))    # w2 (down)
                        for _ in range(cfg.n_experts)],
        }
        if cfg.qk_norm:
            ly["q_norm"], ly["k_norm"] = norm(q_dim), norm(kv_dim)
        w["layers"].append(ly)
    w["final_norm"] = norm(cfg.dim)
    w["lm_head"] = mat(cfg.vocab, cfg.dim)
    return w


def write_model(path, cfg, w):
    wr = rbt2.Rbt2Writer(path, cfg)
    L = wr.L
    wr.put_mat(L["tok_emb"], w["tok_emb"])
    for li, (lw, lo) in enumerate(zip(w["layers"], L["layers"])):
        wr.put_vec(lo["attn_norm"], lw["attn_norm"])
        for name in ("wq", "wk", "wv", "wo"):
            wr.put_mat(lo[name], lw[name])
        if cfg.qk_norm:
            wr.put_vec(lo["q_norm"], lw["q_norm"])
            wr.put_vec(lo["k_norm"], lw["k_norm"])
        wr.put_vec(lo["ffn_norm"], lw["ffn_norm"])
        wr.put_mat(lo["router"], lw["router"])
        for e, (w1, w3, w2) in enumerate(lw["experts"]):
            wr.put_expert(li, e, w1, w3, w2)
    wr.put_vec(L["final_norm"], w["final_norm"])
    wr.put_mat(L["lm_head"], w["lm_head"])
    wr.close()


# ----------------------------------------------------------------------------- math

def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, theta):
    """x: [T, heads, head_dim]. Rotates pairs (i, i+half) -- the HF 'rotate_half' convention."""
    T, _, hd = x.shape
    half = hd // 2
    freqs = theta ** (-2.0 * np.arange(half) / hd)
    ang = np.arange(T)[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def forward(cfg, w, tokens):
    """Returns (logits [T, vocab], routing [T, n_layers, top_k])."""
    T, hd = len(tokens), cfg.head_dim
    group = cfg.n_heads // cfg.n_kv_heads
    f64 = lambda a: np.asarray(a, dtype=np.float64)

    x = f64(w["tok_emb"][tokens])
    routing = np.zeros((T, cfg.n_layers, cfg.top_k), dtype=np.int32)
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)

    for li, ly in enumerate(w["layers"]):
        # --- attention
        xb = rmsnorm(x, f64(ly["attn_norm"]), cfg.rms_eps)
        q = xb @ f64(ly["wq"]).T
        k = xb @ f64(ly["wk"]).T
        v = xb @ f64(ly["wv"]).T
        if cfg.qk_norm:
            q = rmsnorm(q, f64(ly["q_norm"]), cfg.rms_eps)
            k = rmsnorm(k, f64(ly["k_norm"]), cfg.rms_eps)
        q = rope(q.reshape(T, cfg.n_heads, hd), cfg.rope_theta)
        k = rope(k.reshape(T, cfg.n_kv_heads, hd), cfg.rope_theta)
        v = v.reshape(T, cfg.n_kv_heads, hd)
        att = np.zeros((T, cfg.n_heads, hd))
        for h in range(cfg.n_heads):
            kvh = h // group
            scores = q[:, h, :] @ k[:, kvh, :].T / np.sqrt(hd)
            scores[causal] = -np.inf
            att[:, h, :] = softmax(scores) @ v[:, kvh, :]
        x = x + att.reshape(T, -1) @ f64(ly["wo"]).T

        # --- MoE feed-forward
        xb = rmsnorm(x, f64(ly["ffn_norm"]), cfg.rms_eps)
        probs = softmax(xb @ f64(ly["router"]).T)
        moe = np.zeros_like(x)
        for t in range(T):
            top = np.argsort(-probs[t], kind="stable")[: cfg.top_k]
            routing[t, li] = top
            wts = probs[t, top]
            if cfg.norm_topk:
                wts = wts / wts.sum()
            for e, wt in zip(top, wts):
                w1, w3, w2 = (f64(a) for a in ly["experts"][e])
                h = silu(w1 @ xb[t]) * (w3 @ xb[t])
                moe[t] += wt * (w2 @ h)
        x = x + moe

    x = rmsnorm(x, f64(w["final_norm"]), cfg.rms_eps)
    return x @ f64(w["lm_head"]).T, routing


def greedy(cfg, w, prompt, n_new, eos=-1):
    """Greedy continuation by re-running the full sequence each step (slow, obviously correct)."""
    toks = list(prompt)
    for _ in range(n_new):
        if len(toks) >= cfg.max_seq:
            break
        logits, _ = forward(cfg, w, toks)
        nxt = int(np.argmax(logits[-1]))
        toks.append(nxt)
        if nxt == eos:
            break
    return toks[len(prompt):]
