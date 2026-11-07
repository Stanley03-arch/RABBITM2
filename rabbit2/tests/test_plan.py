"""Tests for tools/plan.py (no engine required)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import rbt2
import reference


class PlanTests(unittest.TestCase):
    def test_plan_reads_header(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 't.rbt2'
            cfg = rbt2.Config(vocab=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=2,
                              head_dim=8, n_experts=4, top_k=1, expert_hidden=16, max_seq=8)
            reference.write_model(str(p), cfg, reference.make_weights(cfg, 0))
            r = subprocess.run([sys.executable, str(ROOT / 'tools/plan.py'), str(p), '--ram-gib', '4'],
                               capture_output=True, text=True, check=True)
            self.assertIn('layers=1', r.stdout)
            self.assertIn('experts/layer=4', r.stdout)
            self.assertIn('guidance', r.stdout)


if __name__ == '__main__':
    unittest.main()
