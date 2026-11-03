"""
RABBIT2 file format (.rbt2, version 2): config, byte layout, header, bf16 helpers, writer.

Shared by the NumPy reference, the Hugging Face converter and the tests.
The C engine mirrors layout() in src/rabbit2.c -- if you change one, change the other.

Rules
  * matrices ([out, in], row-major) are stored in cfg.dtype  (0 = float32, 1 = bfloat16)
  * vectors (norm weights) are ALWAYS float32 (bf16 -> f32 is exact, so nothing is lost)
  * header and every expert blob start on 4 KiB; every other tensor starts on 64 bytes
  * one expert = w1[hidden,dim] | w3[hidden,dim] | w2[dim,hidden], contiguous, padded to 4 KiB
"""
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"RBT2"
VERSION = 2
ALIGN = 4096   # header + expert blobs (direct-I/O friendly)
TALIGN = 64    # all other tensors
DTYPE_F32, DTYPE_BF16 = 0, 1
ELEM = {DTYPE_F32: 4, DTYPE_BF16: 2}
HEADER_FMT = "<4sI13i2f"
HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 68


@dataclass
class Config:
    vocab: int = 256
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 16
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 128
    max_seq: int = 64
    dtype: int = DTYPE_F32
    qk_norm: int = 0      # 1 = RMSNorm over the whole q and k projection, before RoPE (OLMoE)
    norm_topk: int = 1    # 1 = renormalise the selected router weights (Mixtral); 0 = keep raw (OLMoE)
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5


FIELD_ORDER = ["vocab", "dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "n_experts", "top_k",
               "expert_hidden", "max_seq", "dtype", "qk_norm", "norm_topk", "rope_theta", "rms_eps"]


def pack_header(cfg):
    return struct.pack(HEADER_FMT, MAGIC, VERSION, *[getattr(cfg, f) for f in FIELD_ORDER])


def unpack_header(buf):
    magic, version, *vals = struct.unpack(HEADER_FMT, buf[:HEADER_BYTES])
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a version-2 .rbt2 file")
    return Config(**dict(zip(FIELD_ORDER, vals)))


def align_up(x, a):
    return (x + a - 1) // a * a


def layout(cfg):
    """Byte offset of every tensor. MUST mirror layout() in src/rabbit2.c exactly."""
    E, F = ELEM[cfg.dtype], 4
    q_dim, kv_dim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    off = ALIGN
    out = {"layers": []}

    def take(nbytes):
        nonlocal off
        off = align_up(off, TALIGN)
        start = off
        off += nbytes
        return start

    out["tok_emb"] = take(cfg.vocab * cfg.dim * E)
    out["expert_stride"] = align_up(3 * cfg.expert_hidden * cfg.dim * E, ALIGN)
    for _ in range(cfg.n_layers):
        ly = {}
        ly["attn_norm"] = take(cfg.dim * F)
        ly["wq"] = take(q_dim * cfg.dim * E)
        ly["wk"] = take(kv_dim * cfg.dim * E)
        ly["wv"] = take(kv_dim * cfg.dim * E)
        ly["wo"] = take(cfg.dim * q_dim * E)
        if cfg.qk_norm:
            ly["q_norm"] = take(q_dim * F)
            ly["k_norm"] = take(kv_dim * F)
        ly["ffn_norm"] = take(cfg.dim * F)
        ly["router"] = take(cfg.n_experts * cfg.dim * E)
        off = align_up(off, ALIGN)
        ly["experts"] = off
        off += cfg.n_experts * out["expert_stride"]
        out["layers"].append(ly)
    out["final_norm"] = take(cfg.dim * F)
    out["lm_head"] = take(cfg.vocab * cfg.dim * E)
    out["total"] = off
    return out


# ----------------------------------------------------------------------------- bf16

def f32_to_bf16(a):
    """float32 -> bfloat16 bit patterns (uint16), round-to-nearest-even."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    out[np.isnan(a)] = 0x7FC0
    return out


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def round_to_dtype(a, dtype):
    """Return float32 values exactly representable in `dtype` (what the engine will actually see)."""
    a = np.asarray(a, dtype=np.float32)
    return bf16_to_f32(f32_to_bf16(a)) if dtype == DTYPE_BF16 else a.copy()


def encode_matrix(a, dtype):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return f32_to_bf16(a).astype("<u2").tobytes() if dtype == DTYPE_BF16 else a.astype("<f4").tobytes()


# ----------------------------------------------------------------------------- writer

class Rbt2Writer:
    """Seek-and-write a .rbt2 file tensor by tensor (never holds the whole model in RAM)."""

    def __init__(self, path, cfg):
        self.cfg, self.L = cfg, layout(cfg)
        self.f = open(path, "wb")
        self.f.truncate(align_up(self.L["total"], ALIGN))
        self.f.write(pack_header(cfg))

    def _put(self, off, data):
        self.f.seek(off)
        self.f.write(data)

    def put_vec(self, off, a):
        self._put(off, np.ascontiguousarray(a, dtype="<f4").tobytes())

    def put_mat(self, off, a):
        self._put(off, encode_matrix(a, self.cfg.dtype))

    def put_expert(self, layer, e, w1, w3, w2):
        off = self.L["layers"][layer]["experts"] + e * self.L["expert_stride"]
        self._put(off, encode_matrix(w1, self.cfg.dtype)
                  + encode_matrix(w3, self.cfg.dtype)
                  + encode_matrix(w2, self.cfg.dtype))

    def close(self):
        self.f.close()
