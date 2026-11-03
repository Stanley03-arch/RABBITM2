# RABBIT2

A from-scratch inference engine for Mixture-of-Experts language models, built so that models far
bigger than RAM can run on an ordinary laptop by keeping the weights on disk.

**Status: Milestone 2 built.** The engine can load a real MoE model (OLMoE-1B-7B) from a converted
file and generate text. It is verified against its own NumPy reference. The final gate, matching the
real Hugging Face implementation, still has to be run by you (see below). No quantization or explicit
disk streaming yet; those are M3 and M4.

## What is and isn't verified

| Check | Status |
|---|---|
| C engine vs NumPy reference (f32 + bf16, GQA/MHA, QK-norm, top-k renorm on/off, KV cache, generation) | **passes**, 18 cases, logits within ~2e-6 |
| Converter + safetensors reader + engine, end to end on a fake sharded bf16 OLMoE checkpoint | **passes** |
| Bad checkpoints (wrong model type, missing tensor, wrong shape, pre-quantized) are rejected | **passes** |
| C and Python agree on every byte offset at full OLMoE size (13.84 GB, offsets > 4 GiB) | **passes** |
| `check_vs_hf.py --ref` catches corrupted logits and routing | **passes** |
| Windows build | compiles with zero warnings (MinGW-w64 cross-compile), **never run on Windows** |
| Matches real Hugging Face OLMoE code | **NOT yet verified. This is your gate (A, then B).** |
| `make_tiny_hf.py`, `dump_hf_logits.py`, `run.py` with a real tokenizer | written, syntax-checked, **never run** (need torch/transformers) |

Why the NumPy reference alone isn't enough: it encodes my understanding of the OLMoE architecture.
I checked the details that matter (QK-norm over the full projection before RoPE; no renormalisation of
top-k weights; `mlp.gate`, `gate_proj/up_proj/down_proj` tensor names) against the public config
files and vLLM's implementation, but only a comparison with Hugging Face's own code proves it.

## Setup

```
pip install numpy            # needed for tests and the converter
```
Build (any C99 compiler; on Windows install MSYS2 + MinGW-w64 gcc, 64-bit):
```
mkdir build
gcc -O2 -std=c99 -o build/rabbit2 src/rabbit2.c -lm        # Linux/macOS
gcc -O2 -std=c99 -o build\rabbit2.exe src\rabbit2.c         # Windows
```
Self-tests (no downloads; the tests also build the engine if it's missing):
```
python tests/test_forward.py
python tests/test_convert.py
python tests/test_layout_big.py      # needs ~17 GB free temp space on Windows (sparse file on Linux)
```
Each should end with `ALL PASSED`. **First thing to do: run these on Windows and report any failure.**

## Gate A: tiny model through real Hugging Face code (about 5 minutes)

```
pip install torch transformers
python tools/make_tiny_hf.py tiny_olmoe
python tools/convert_hf.py tiny_olmoe tiny.rbt2 --dtype f32
python tools/check_vs_hf.py --hf tiny_olmoe tiny.rbt2
```
Expect `PASS` with max logit diff around 1e-5 and routing 100% identical. If conversion fails with
"tensors are missing", paste the error: the printed tensor names tell us how this transformers version
saves experts.

## Gate B: the real OLMoE-1B-7B

1. Download the original weights (about 13.8 GB; you need ~28 GB free during conversion):
   `pip install huggingface_hub`, then `hf download allenai/OLMoE-1B-7B-0924 --local-dir olmoe_hf`
   (older versions: `huggingface-cli download ...`).
2. Convert (about 13.8 GB output, bf16): `python tools/convert_hf.py olmoe_hf olmoe.rbt2`
3. Sanity run, the quick check that works on your 8 GB laptop:
   ```
   pip install transformers
   python tools/run.py --model olmoe.rbt2 --tok olmoe_hf --prompt "The capital of France is" --n 20
   ```
   Output should be coherent English. A wrong RoPE, QK-norm or routing detail usually produces garbage.
4. Exact check. Hugging Face needs more RAM than your laptop has, so produce the reference elsewhere,
   for example Google Colab with a GPU:
   ```
   !pip -q install transformers accelerate
   !git clone https://github.com/Stanley03-arch/RABBIT2.git
   !python RABBIT2/tools/dump_hf_logits.py allenai/OLMoE-1B-7B-0924 hf_ref.npz --dtype float16
   ```
   Download `hf_ref.npz` (a few MB) and on your laptop run:
   `python tools/check_vs_hf.py --ref hf_ref.npz olmoe.rbt2 --atol 0.1`
   Argmax should agree at (nearly) every position and routing at (nearly) every layer. Small logit
   differences are expected when the reference runs in fp16/bf16.

## What speed to expect

Measured here with a zero-weight file of the exact OLMoE shape (so no disk reads), one 2.8 GHz server core:
about **2.5 tokens/s of pure compute**. Real runs add disk reads: a token touches 8 experts in each of 16
layers, up to ~1.6 GB if nothing is cached. With 8 GB RAM the OS keeps some experts cached and your SSD
speed sets the rest. Treat roughly 0.3 to 1 token/s as a plausible range on your laptop, to be measured.
The prompt phase is slower per token on the first run (cold cache).

## Layout

| Path | What |
|---|---|
| `src/rabbit2.c` | The engine: f32/bf16 weights, memory-mapped file, QK-norm, KV cache, greedy generation |
| `ref/rbt2.py` | `.rbt2` format v2: config, byte layout, bf16 helpers, streaming writer |
| `ref/reference.py` | NumPy golden model and random test-weight generator |
| `tools/convert_hf.py` | Hugging Face OLMoE safetensors to `.rbt2` (never holds the model in RAM) |
| `tools/safetensors_min.py` | Pure-Python safetensors reader/writer with bf16 support |
| `tools/make_tiny_hf.py` | Tiny random OLMoE built with the real HF code (Gate A) |
| `tools/check_vs_hf.py` | Compare engine to HF live (`--hf`) or to a saved reference (`--ref`) |
| `tools/dump_hf_logits.py` | Save HF reference logits and routing on a bigger machine |
| `tools/run.py` | Tokenise, generate, decode |
| `tests/` | The self-tests listed above |

## File format v2 (.rbt2)

Header: magic `RBT2`, version 2, then 13 int32 (`vocab dim n_layers n_heads n_kv_heads head_dim
n_experts top_k expert_hidden max_seq dtype qk_norm norm_topk`) and 2 float32 (`rope_theta rms_eps`).
Matrices are `[out, in]` row-major in the file dtype (f32 or bf16); norm vectors are always f32.
Each expert is one contiguous blob (`w1 | w3 | w2`, gate/up/down) padded to 4 KiB, so fetching an
expert from disk later is a single aligned read. All offsets are computed by `layout()` in both
`ref/rbt2.py` and `src/rabbit2.c`, and a test checks they agree.

## Design decisions worth keeping

- **Reference first.** Every engine change must keep matching the NumPy model.
- **One seam for experts.** All expert reads go through `get_expert()`. M4 replaces that one function.
- **Memory-mapping is a stepping stone.** It lets a 14 GB model run in 8 GB RAM now, but the OS decides
  what stays cached. M4 replaces it with explicit I/O so RABBIT2 controls the cache and prefetching.
- **No fast-math.** Keeps results reproducible while validating.

## Roadmap

1. **M1 (done):** tiny MoE forward pass vs NumPy reference.
2. **M2 (built; gates A and B pending):** real OLMoE through a converter, mmap loading, generation.
3. **M3, quantization:** int8 then group-wise int4/2-bit experts, AVX2 matvec. *Gate:* perplexity within an
   agreed margin of bf16, plus a measured speedup.
4. **M4, disk streaming:** explicit unbuffered I/O (Windows: `FILE_FLAG_NO_BUFFERING` + overlapped I/O or IoRing;
   Linux: `O_DIRECT`), fixed RAM budget, LRU expert cache. *Gate:* same outputs as M3 with RAM capped below model size.
5. **M5, prefetch and overlap:** read the next layer's experts while computing; profile I/O wait vs compute.
6. **M6, usability:** multithreading, sampling, chat loop.
7. **M7, scale:** ~30B, then ~300B, then an external NVMe for 1T+.

## Known limits

- Only OLMoE is supported by the converter. Other architectures (Qwen MoE, DeepSeek, Mixtral) need their own
  mapping and sometimes new features (shared experts, MLA attention).
- Checkpoints re-saved with fused expert tensors are rejected; use the original files.
- A transposed *square* matrix (e.g. OLMoE's q_proj) can't be caught by shape checks; only the logit comparison can.
- Needs a 64-bit build (a 32-bit process cannot map a multi-GB file).
- Near-tied router scores can route differently between precisions. That is expected, not a bug.
- Speed is not optimised yet. Single thread, scalar math.
