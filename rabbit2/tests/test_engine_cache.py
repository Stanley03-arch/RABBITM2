"""Test the optional local expert-cache inference path against mmap inference."""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import rbt2
import reference
from test_forward import find_or_build_binary


class EngineCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = find_or_build_binary()

    def test_equivalence_and_eviction(self):
        # F32 expert blobs are 768 KiB; a 1 MiB budget holds one, forcing evictions.
        for dtype in (rbt2.DTYPE_F32, rbt2.DTYPE_BF16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as td:
                p = Path(td)
                cfg = rbt2.Config(vocab=32, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
                                  head_dim=32, n_experts=3, top_k=2, expert_hidden=512,
                                  max_seq=16, dtype=dtype, qk_norm=1, norm_topk=0)
                model = p / 'model.rbt2'
                reference.write_model(str(model), cfg, reference.make_weights(cfg, 9))
                tokens = ['1', '2', '3', '4']

                def run_dump(cache_mib=None):
                    logits, routing = p / 'logits.bin', p / 'routing.bin'
                    cmd = [self.binary, str(model), '--dump', str(logits), str(routing)]
                    if cache_mib:
                        cmd += ['--expert-cache-mib', str(cache_mib)]
                    result = subprocess.run(cmd + tokens, capture_output=True, text=True, check=True)
                    return logits.read_bytes(), routing.read_bytes(), result.stderr

                ref_l, ref_r, _ = run_dump()
                small_l, small_r, small_log = run_dump(1)
                big_l, big_r, big_log = run_dump(8)
                self.assertEqual((small_l, small_r), (ref_l, ref_r))
                self.assertEqual((big_l, big_r), (ref_l, ref_r))
                s = {k: int(v) for k, v in re.findall(r'(hits|misses|evictions|bytes_read|resident_bytes|capacity_bytes)=(\d+)', small_log)}
                self.assertGreater(s['evictions'], 0)
                self.assertLessEqual(s['resident_bytes'], 1024 * 1024)
                self.assertEqual(s['bytes_read'], s['misses'] * rbt2.layout(cfg)['expert_stride'])
                self.assertGreater(int(re.search(r'hits=(\d+)', big_log)[1]), 0)

                def generate(extra):
                    cmd = [self.binary, str(model), '--gen', '4', '--ctx', '12'] + extra + tokens
                    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
                self.assertEqual(generate([]), generate(['--expert-cache-mib', '1']))
                wrapper = subprocess.run([sys.executable, str(ROOT / 'tools/run.py'), '--bin', self.binary,
                                          '--model', str(model), '--ids', '1', '2', '--n', '2',
                                          '--expert-cache-mib', '1'], capture_output=True, text=True, check=True)
                self.assertIn('expert cache:', wrapper.stderr)

    def test_bad_budget_and_small_cache(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'm.rbt2'
            cfg = rbt2.Config(vocab=8, dim=128, n_layers=1, n_heads=4, n_kv_heads=2,
                              head_dim=32, n_experts=1, top_k=1, expert_hidden=1024, max_seq=4)
            writer = rbt2.Rbt2Writer(path, cfg)
            writer.close()
            for value in ('0', '-1', 'garbage', '18446744073709551616'):
                result = subprocess.run([self.binary, str(path), '--gen', '1', '--expert-cache-mib', value, '1'],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--expert-cache-mib', result.stderr)
            result = subprocess.run([self.binary, str(path), '--gen', '1', '--expert-cache-mib', '1', '1'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('budget too small', result.stderr)

    @unittest.skipIf(os.name == 'nt', 'sparse-file creation test runs only on POSIX')
    def test_expert_read_beyond_4gib(self):
        # Sparse zero-weight model: layer 1 expert begins beyond 4 GiB, but inference is tiny.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sparse.rbt2'
            cfg = rbt2.Config(vocab=8, dim=2, n_layers=2, n_heads=1, n_kv_heads=1,
                              head_dim=2, n_experts=1048576, top_k=1, expert_hidden=1, max_seq=2)
            layout = rbt2.layout(cfg)
            self.assertGreater(layout['layers'][1]['experts'], 2**32)
            with path.open('wb') as f:
                f.write(rbt2.pack_header(cfg))
                f.truncate(layout['total'])
            p = Path(td)
            cmd = [self.binary, str(path), '--dump', str(p / 'logits'), str(p / 'routing'),
                   '--expert-cache-mib', '1', '1']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn('misses=2', result.stderr)
            self.assertEqual(struct.unpack('<2i', (p / 'routing').read_bytes()), (0, 0))


if __name__ == '__main__':
    unittest.main()
