"""Replay expert ranges through the prototype cache; does NOT run inference."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time

from expert_cache import ExpertCache, RangeSource

MIB = 1024 * 1024


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return n


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def replay(source, ranges, tokens, directory, memory, disk, max_range, verify=None):
    cache = ExpertCache(source, directory, memory, disk, max_range)
    start = time.perf_counter()
    for offset, length in ranges:
        data = cache.get(offset, length)
        if verify is not None and data != verify.read(offset, length):
            raise RuntimeError('range does not match the local reference')
    elapsed = time.perf_counter() - start
    s = cache.snapshot()
    print(f"Requests: {s['requests']}  |  trace tokens: {tokens}")
    print(f"RAM hits: {s['memory_hits']}  |  disk hits: {s['disk_hits']}  |  source fetches: {s['source_fetches']}")
    print(f"Fetched: {s['source_bytes'] / MIB:.2f} MiB  |  per trace token: {s['source_bytes'] / max(tokens, 1) / MIB:.2f} MiB")
    print(f"Cache payload: RAM {s['memory_payload_bytes'] / MIB:.2f} MiB, disk {s['disk_payload_bytes'] / MIB:.2f} MiB")
    print(f"Evictions: RAM {s['memory_evictions']}, disk {s['disk_evictions']}")
    print(f'Replay elapsed: {elapsed:.3f} s (includes verification)' if verify else f'Replay elapsed: {elapsed:.3f} s')
    print('Replay measures expert reads only, NOT model inference speed or total system memory.')
    return s


def demo(args):
    # Small, deterministic fixture served over loopback; not a model or an internet speed test.
    block = 64 * 1024
    chunks = [bytes([i]) * block for i in range(8)]
    data = b''.join(chunks)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                unit, spec = self.headers['Range'].split('=', 1)
                start, end = map(int, spec.split('-', 1))
                if unit != 'bytes' or start < 0 or end < start or end >= len(data):
                    raise ValueError()
            except (KeyError, ValueError):
                self.send_error(416)
                return
            payload = data[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='rbt-cache-demo-') as tmp:
            fixture = Path(tmp) / 'reference.bin'
            fixture.write_bytes(data)
            source = RangeSource(f'http://127.0.0.1:{server.server_port}/weights')
            trace = [0, 1, 0, 2, 0, 1, 3, 0, 2, 4, 0, 1]
            print('Synthetic loopback demo: 12 requests; verifying every range against fixture.')
            replay(source, [(i * block, block) for i in trace], 3, Path(tmp) / 'cache',
                   2 * block, 4 * block, block, RangeSource(fixture))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_trace(args):
    # rbt2 imports numpy; only real model replay requires the project dependency.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ref'))
    import rbt2
    with open(args.header, 'rb') as f:
        header = f.read(rbt2.HEADER_BYTES)
    if len(header) != rbt2.HEADER_BYTES:
        raise ValueError('model header is truncated')
    cfg = rbt2.unpack_header(header)
    if (min(cfg.vocab, cfg.n_layers, cfg.n_experts, cfg.top_k, cfg.dim, cfg.expert_hidden,
            cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq) <= 0
            or cfg.top_k > cfg.n_experts or cfg.dtype not in rbt2.ELEM):
        raise ValueError('invalid model header')
    layout = rbt2.layout(cfg)
    stride = layout['expert_stride']
    if stride > args.max_range_mib * MIB:
        raise ValueError(f'expert size {stride} exceeds --max-range-mib')
    raw = Path(args.routing).read_bytes()
    record = cfg.n_layers * cfg.top_k
    if not raw or len(raw) % (4 * record):
        raise ValueError('routing file size does not match header')
    values = struct.iter_unpack('<i', raw)
    ranges = []
    for i, (expert,) in enumerate(values):
        if expert < 0 or expert >= cfg.n_experts:
            raise ValueError(f'invalid expert index {expert} at routing entry {i}')
        layer = (i // cfg.top_k) % cfg.n_layers
        ranges.append((layout['layers'][layer]['experts'] + expert * stride, stride))
    tokens = len(raw) // (4 * record)
    source = RangeSource(args.source)
    print(f'Expert size: {stride / MIB:.2f} MiB; {cfg.n_layers} layers, top-{cfg.top_k}.')
    print('Source must contain the exact same immutable .rbt2 bytes as the header and routing trace.')
    with tempfile.TemporaryDirectory(prefix='rbt-cache-') as temp:
        directory = args.cache_dir or temp
        replay(source, ranges, tokens, directory, args.ram_mib * MIB, args.disk_mib * MIB,
               args.max_range_mib * MIB, RangeSource(args.verify_file) if args.verify_file else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('demo', help='small synthetic loopback HTTP test; no downloads')
    trace = sub.add_parser('replay', help='replay actual --dump routing against an immutable .rbt2 source')
    trace.add_argument('--header', required=True, help='local .rbt2 file or its first 68 bytes')
    trace.add_argument('--routing', required=True, help='routing.bin from rabbit2 --dump')
    trace.add_argument('--source', required=True, help='local .rbt2 path or immutable HTTP(S) URL')
    trace.add_argument('--verify-file', help='compare fetched bytes to this local .rbt2 (slower)')
    trace.add_argument('--ram-mib', type=nonnegative, default=64)
    trace.add_argument('--disk-mib', type=nonnegative, default=256)
    trace.add_argument('--max-range-mib', type=positive, default=32)
    trace.add_argument('--cache-dir', help='dedicated cache directory; default temporary and deleted on exit')
    args = p.parse_args()
    try:
        demo(args) if args.mode == 'demo' else run_trace(args)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, f'Benchmark failed: {exc}\n')


if __name__ == '__main__':
    main()
