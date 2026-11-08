"""Tests for tools/doctor.py JSON mode (no engine required)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref'))
import rbt2
import reference


class DoctorTests(unittest.TestCase):
    def test_doctor_json(self):
        r = subprocess.run([sys.executable, str(ROOT / 'tools/doctor.py'), '--json'],
                           capture_output=True, text=True, check=False)
        obj = json.loads(r.stdout)
        self.assertIn('checks', obj)
        self.assertIn('ok', obj)

    def test_doctor_json_model(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 't.rbt2'
            cfg = rbt2.Config(vocab=16, dim=8, n_layers=1, n_heads=1, n_kv_heads=1,
                              head_dim=8, n_experts=2, top_k=1, expert_hidden=8, max_seq=4)
            reference.write_model(str(p), cfg, reference.make_weights(cfg, 0))
            r = subprocess.run([sys.executable, str(ROOT / 'tools/doctor.py'), '--json', '--model', str(p)],
                               capture_output=True, text=True, check=False)
            obj = json.loads(r.stdout)
            self.assertEqual(obj['model']['n_layers'], 1)


if __name__ == '__main__':
    unittest.main()
