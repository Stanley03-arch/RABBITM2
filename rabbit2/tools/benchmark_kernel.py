"""Compare scalar and auto-selected CPU kernels using local model inference.

This measures whole-process prompt/dump time on repeated token IDs. It is not
an internet, cold-disk, chat-generation, or 3T-model benchmark.
"""
import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import rbt2


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def benchmark(binary, model, tokens, repeats, directory):
    with open(model, 'rb') as f:
        header = f.read(rbt2.HEADER_BYTES)
    if len(header) != rbt2.HEADER_BYTES:
        raise ValueError('truncated model header')
    cfg = rbt2.unpack_header(header)
    if cfg.vocab < 1 or cfg.max_seq < tokens:
        raise ValueError('invalid vocab or token count exceeds model max_seq')
    prompt = [str(1 if cfg.vocab > 1 else 0)] * tokens
    results = {'scalar': [], 'auto': []}
    outputs = {}
    kernels = {}

    def run(kernel, measured):
        logits = directory / (kernel + '-logits.bin')
        routing = directory / (kernel + '-routing.bin')
        cmd = [str(binary), str(model), '--dump', str(logits), str(routing), '--ctx', str(tokens),
               '--kernel', kernel, *prompt]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - start
        for line in proc.stderr.splitlines():
            if line.startswith('matvec kernel:'):
                kernels[kernel] = line.split(':', 1)[1].strip()
        if measured:
            results[kernel].append(elapsed)
            outputs[kernel] = (np.fromfile(logits, dtype='<f4'), routing.read_bytes())

    # Warm each path once; alternate order for measured runs.
    for kernel in ('scalar', 'auto'):
        run(kernel, False)
    for i in range(repeats):
        for kernel in (('scalar', 'auto') if i % 2 == 0 else ('auto', 'scalar')):
            run(kernel, True)
    s, a = outputs['scalar'][0], outputs['auto'][0]
    if s.shape != a.shape or s.size != tokens * cfg.vocab:
        raise RuntimeError('unexpected logits output size')
    scalar_time, auto_time = (statistics.median(results[k]) for k in ('scalar', 'auto'))
    print(f"Selected auto kernel: {kernels.get('auto', 'unknown')}")
    print(f'Scalar median: {scalar_time:.4f} s; auto median: {auto_time:.4f} s; '
          f'ratio scalar/auto: {scalar_time / auto_time:.2f}x')
    print(f'Max absolute logit difference: {np.max(np.abs(s - a)):.6g}; '
          f'routing identical: {outputs["scalar"][1] == outputs["auto"][1]}')
    print('Whole-process prompt/dump wall time; repeated token IDs, warm OS cache, no generated tokens.')
    print('Not a cold-disk, remote streaming, or 3T model speed measurement.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', help='existing local .rbt2 file')
    group.add_argument('--synthetic', action='store_true', help='make a small random test model temporarily')
    p.add_argument('--bin', default=str(ROOT / 'build' / ('rabbit2.exe' if sys.platform == 'win32' else 'rabbit2')))
    p.add_argument('--tokens', type=positive, default=16)
    p.add_argument('--repeats', type=positive, default=3)
    args = p.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix='rbt-kernel-bench-') as td:
            directory = Path(td)
            if args.synthetic:
                import reference
                cfg = rbt2.Config(vocab=256, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
                                  head_dim=32, n_experts=4, top_k=2, expert_hidden=256,
                                  max_seq=max(args.tokens, 64), dtype=rbt2.DTYPE_BF16)
                model = directory / 'synthetic.rbt2'
                reference.write_model(str(model), cfg, reference.make_weights(cfg, 42))
                print('Synthetic random model; results do not predict real model speed.')
            else:
                model = Path(args.model)
            benchmark(Path(args.bin).resolve(), model, args.tokens, args.repeats, directory)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        p.exit(1, f'Benchmark failed: {exc}\n')


if __name__ == '__main__':
    main()
