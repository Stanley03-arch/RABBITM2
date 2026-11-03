"""Check the opt-in AVX2 kernel against scalar and NumPy for f32/bf16 models."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import rbt2
import reference
from test_forward import find_or_build_binary


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = find_or_build_binary()

    def test_kernel_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.rbt2'
            cfg = rbt2.Config()
            reference.write_model(str(path), cfg, reference.make_weights(cfg, 1))
            result = subprocess.run([self.binary, str(path), '--info', '--kernel', 'auto'],
                                    capture_output=True, text=True, check=True)
            self.assertRegex(result.stderr, r'matvec kernel: (scalar|avx2)')
            result = subprocess.run([self.binary, str(path), '--info', '--kernel', 'invalid'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            result = subprocess.run([self.binary, str(path), '--info', '--kernel', 'avx2'],
                                    capture_output=True, text=True)
            if 'matvec kernel: avx2' in result.stderr:
                self.assertEqual(result.returncode, 0)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unavailable', result.stderr)

    def test_benchmark_smoke(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_kernel.py'),
                                 '--synthetic', '--tokens', '4', '--repeats', '1',
                                 '--bin', self.binary], capture_output=True, text=True, check=True)
        self.assertIn('Selected auto kernel:', result.stdout)
        self.assertIn('routing identical: True', result.stdout)
        self.assertIn('Not a cold-disk', result.stdout)

    def test_auto_matches_reference(self):
        # Exercise vector and scalar tails, including dimensions smaller than 16.
        for dtype in (rbt2.DTYPE_F32, rbt2.DTYPE_BF16):
            for dim, heads, hidden in [(8, 2, 7), (36, 6, 25), (64, 4, 64)]:
                with self.subTest(dtype=dtype, dim=dim), tempfile.TemporaryDirectory() as td:
                    p = Path(td)
                    cfg = rbt2.Config(vocab=48, dim=dim, n_layers=2, n_heads=heads,
                                      n_kv_heads=heads, head_dim=dim // heads, n_experts=4,
                                      top_k=2, expert_hidden=hidden, max_seq=16, dtype=dtype,
                                      qk_norm=1, norm_topk=0)
                    weights = reference.make_weights(cfg, 23)
                    model = p / 'm.rbt2'
                    reference.write_model(str(model), cfg, weights)
                    tokens = [1, 2, 3, 4, 5]
                    expected_logits, expected_routing = reference.forward(cfg, weights, tokens)
                    def dump(kernel, cache=False):
                        logits, routing = p / 'logits.bin', p / 'routing.bin'
                        cmd = [self.binary, str(model), '--dump', str(logits), str(routing), '--kernel', kernel]
                        if cache:
                            cmd += ['--expert-cache-mib', '1']
                        result = subprocess.run(cmd + list(map(str, tokens)), capture_output=True, text=True, check=True)
                        return (np.fromfile(logits, dtype='<f4').reshape(len(tokens), cfg.vocab),
                                np.fromfile(routing, dtype='<i4').reshape(len(tokens), cfg.n_layers, cfg.top_k),
                                result.stderr)
                    scalar, scalar_routes, _ = dump('scalar')
                    vector, vector_routes, log = dump('auto', cache=True)
                    self.assertIn('matvec kernel:', log)
                    np.testing.assert_allclose(vector, expected_logits, atol=1e-4, rtol=0)
                    np.testing.assert_allclose(vector, scalar, atol=1e-4, rtol=0)
                    np.testing.assert_array_equal(vector_routes, expected_routing)
                    np.testing.assert_array_equal(vector_routes, scalar_routes)
                    def generate(kernel):
                        return subprocess.run([self.binary, str(model), '--gen', '4', '--kernel', kernel,
                                               *map(str, tokens)], check=True, capture_output=True, text=True).stdout
                    self.assertEqual(generate('scalar'), generate('auto'))


if __name__ == '__main__':
    unittest.main()
