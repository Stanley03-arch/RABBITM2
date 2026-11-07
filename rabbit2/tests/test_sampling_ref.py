"""Tests for NumPy reference sampling helpers (no C engine required)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import reference as ref  # noqa: E402
import rbt2  # noqa: E402


class SamplingRefTests(unittest.TestCase):
    def test_greedy_temperature_zero(self):
        logits = np.array([1.0, 3.0, 2.0], dtype=np.float64)
        rng = np.random.default_rng(0)
        self.assertEqual(ref.sample_next(logits, temperature=0.0, rng=rng), 1)

    def test_top_k_limits_candidates(self):
        logits = np.array([10.0, 9.0, -5.0, -6.0], dtype=np.float64)
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for seed in range(200):
            tok = ref.sample_next(logits, temperature=1.0, top_k=2, rng=np.random.default_rng(seed))
            counts[tok] += 1
        self.assertEqual(counts[2], 0)
        self.assertEqual(counts[3], 0)
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)

    def test_top_p_nucleus(self):
        logits = np.array([5.0, 4.9, -10.0], dtype=np.float64)
        counts = {0: 0, 1: 0, 2: 0}
        for seed in range(200):
            tok = ref.sample_next(logits, temperature=1.0, top_p=0.6, rng=np.random.default_rng(seed))
            counts[tok] += 1
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[2], 0)

    def test_deterministic_seed(self):
        logits = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        a = ref.sample_next(logits, temperature=1.0, rng=np.random.default_rng(123))
        b = ref.sample_next(logits, temperature=1.0, rng=np.random.default_rng(123))
        self.assertEqual(a, b)

    def test_generate_matches_greedy_default(self):
        cfg = rbt2.Config(vocab=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=2,
                          head_dim=8, n_experts=2, top_k=1, expert_hidden=16, max_seq=16)
        w = ref.make_weights(cfg, 7)
        prompt = [1, 2, 3]
        self.assertEqual(ref.generate(cfg, w, prompt, 4), ref.greedy(cfg, w, prompt, 4))


if __name__ == '__main__':
    unittest.main()
