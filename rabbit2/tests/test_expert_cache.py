"""Tests for the prototype cache and replay CLI, including a local HTTP server."""
import contextlib
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from expert_cache import ExpertCache, RangeError, RangeSource  # noqa: E402

DATA = bytes(range(256)) * 8


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ignore':
            self.send_response(200)
            self.end_headers()
            return
        a, b = map(int, self.headers['Range'][6:].split('-'))
        self.send_response(206)
        self.send_header('Content-Range', f'bytes {a if self.path != "/wrong" else a + 1}-{b}/{len(DATA)}')
        self.end_headers()
        self.wfile.write(DATA[a:b + 1])

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def server():
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    t = threading.Thread(target=http.serve_forever, daemon=True)
    t.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown()
        http.server_close()
        t.join()


class CacheTests(unittest.TestCase):
    def test_lru_and_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            src = temp / 'weights'
            src.write_bytes(DATA)
            directory = temp / 'cache'
            c = ExpertCache(RangeSource(src), directory, 16, 32, 16)
            for offset in [0, 0, 16, 0, 32, 16]:
                self.assertEqual(c.get(offset, 16), DATA[offset:offset + 16])
                self.assertLessEqual(c.snapshot()['disk_payload_bytes'], 32)
                self.assertLessEqual(c.snapshot()['memory_payload_bytes'], 16)
            s = c.snapshot()
            self.assertEqual(s['memory_hits'], 1)
            self.assertEqual(s['disk_hits'], 1)
            self.assertEqual(s['source_fetches'], 4)
            self.assertGreater(s['disk_evictions'], 0)
            c2 = ExpertCache(RangeSource(src), directory, 0, 32, 16)
            self.assertEqual(c2.get(16, 16), DATA[16:32])
            self.assertEqual(c2.snapshot()['disk_hits'], 1)
            # A smaller limit trims old entries, but leaves unrelated files alone.
            (directory / 'keep.txt').write_text('keep')
            c3 = ExpertCache(RangeSource(src), directory, 0, 0, 16)
            self.assertEqual(c3.snapshot()['disk_payload_bytes'], 0)
            self.assertTrue((directory / 'keep.txt').exists())
            self.assertEqual(c3.get(16, 16), DATA[16:32])
            self.assertEqual(c3.snapshot()['disk_payload_bytes'], 0)

    def test_http_range_and_fail_closed(self):
        with server() as url:
            self.assertEqual(RangeSource(url + '/ok').read(10, 20), DATA[10:30])
            with self.assertRaises(RangeError):
                RangeSource(url + '/ignore').read(10, 20)
            with self.assertRaises(RangeError):
                RangeSource(url + '/wrong').read(10, 20)
            with self.assertRaises(RangeError):
                RangeSource(url + '/ok').read(len(DATA) - 1, 10)

    def test_short_file_and_range_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'w'
            file.write_bytes(DATA)
            cache = ExpertCache(RangeSource(file), Path(temp) / 'c', 64, 64, 32)
            for offset, length in [(-1, 10), (0, 0), (0, 33)]:
                with self.assertRaises(ValueError):
                    cache.get(offset, length)
            with self.assertRaises(RangeError):
                cache.get(len(DATA) - 1, 10)
            self.assertEqual(cache.snapshot()['source_fetches'], 0)

    def test_demo_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_cache.py'), 'demo'],
                                capture_output=True, text=True, check=True)
        self.assertIn('Requests: 12', result.stdout)
        self.assertIn('NOT model inference', result.stdout)

    def test_engine_routing_replay(self):
        if not any(shutil.which(c) for c in ('gcc', 'clang', 'cc')):
            self.skipTest('C compiler not available')
        sys.path.insert(0, str(ROOT / 'ref'))
        import rbt2
        import reference
        from test_forward import find_or_build_binary
        cfg = rbt2.Config(vocab=16, dim=8, n_layers=2, n_heads=2, n_kv_heads=2,
                          head_dim=4, n_experts=2, top_k=1, expert_hidden=8, max_seq=4)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model, logits, routing = temp / 'model.rbt2', temp / 'logits.bin', temp / 'routing.bin'
            reference.write_model(str(model), cfg, reference.make_weights(cfg, 2))
            subprocess.run([find_or_build_binary(), str(model), '--dump', str(logits),
                            str(routing), '1', '2'], check=True, capture_output=True)
            out = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_cache.py'), 'replay',
                                  '--header', str(model), '--routing', str(routing), '--source', str(model),
                                  '--verify-file', str(model)], capture_output=True, text=True, check=True).stdout
            self.assertIn('Requests: 4  |  trace tokens: 2', out)

    def test_real_trace_cli(self):
        sys.path.insert(0, str(ROOT / 'ref'))
        import rbt2
        import reference
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            cfg = rbt2.Config(vocab=16, dim=8, n_layers=2, n_heads=2, n_kv_heads=2,
                              head_dim=4, n_experts=2, top_k=1, expert_hidden=8, max_seq=4)
            model, routing = temp / 'model.rbt2', temp / 'routing.bin'
            reference.write_model(str(model), cfg, reference.make_weights(cfg, 1))
            # Two tokens, two layers, one expert per layer.
            import struct
            routing.write_bytes(struct.pack('<4i', 0, 1, 0, 1))
            cmd = [sys.executable, str(ROOT / 'tools/benchmark_cache.py'), 'replay',
                   '--header', str(model), '--routing', str(routing), '--source', str(model),
                   '--verify-file', str(model), '--ram-mib', '1', '--disk-mib', '1']
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            self.assertIn('Requests: 4  |  trace tokens: 2', out)
            self.assertIn('RAM hits: 2', out)
            routing.write_bytes(struct.pack('<4i', 0, 99, 0, 1))
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
