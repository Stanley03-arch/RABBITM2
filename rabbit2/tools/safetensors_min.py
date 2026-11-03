"""
Minimal safetensors reader/writer (pure Python + NumPy, no torch needed).

Why: the official numpy loader cannot read bfloat16, and real checkpoints are bf16.
Format: 8-byte little-endian header length N, N bytes of JSON, then raw tensor data.
Each tensor is read on demand with a seek, so a multi-GB shard is never loaded whole.
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
import rbt2  # noqa: E402

_DTYPES = {"F32": 4, "F16": 2, "BF16": 2}


class SafetensorsFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        self.data_start = 8 + n
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}

    def keys(self):
        return self.meta.keys()

    def shape(self, name):
        return tuple(self.meta[name]["shape"])

    def get(self, name):
        """Return the tensor as a float32 array."""
        info = self.meta[name]
        dt = info["dtype"]
        if dt not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {dt} (supported: {sorted(_DTYPES)})")
        begin, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
        if dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            arr = rbt2.bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
        return arr.reshape(info["shape"])


def write_safetensors(path, tensors, dtype="BF16"):
    """tensors: dict name -> float32 array. dtype: 'BF16' or 'F32'."""
    header, chunks, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        data = rbt2.f32_to_bf16(a).astype("<u2").tobytes() if dtype == "BF16" else a.astype("<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(data)]}
        chunks.append(data)
        off += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for c in chunks:
            f.write(c)
