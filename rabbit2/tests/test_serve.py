"""Tests for serve.py pure helpers (no engine/tokenizer/model required)."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import serve  # noqa: E402


class ServeTests(unittest.TestCase):
    def test_messages_to_prompt(self):
        p = serve.messages_to_prompt([
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
        ])
        self.assertIn("System: Be brief", p)
        self.assertIn("User: Hi", p)
        self.assertTrue(p.rstrip().endswith("Assistant:"))

    def test_messages_content_parts(self):
        p = serve.messages_to_prompt([{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}])
        self.assertIn("User: AB", p)

    def test_run_engine_success(self):
        fake = mock.Mock(returncode=0, stdout="5 6\n", stderr="timing\n")
        with mock.patch('serve.subprocess.run', return_value=fake) as run:
            ids, err = serve.run_engine('bin', 'm.rbt2', [1, 2], 3, 16, 99, ['--kernel', 'auto'])
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[:7], ['bin', 'm.rbt2', '--gen', '3', '--ctx', '16', '--kernel'])
        self.assertIn('--eos', cmd)
        self.assertEqual(ids, [5, 6])
        self.assertEqual(err, 'timing\n')

    def test_run_engine_failure(self):
        fake = mock.Mock(returncode=1, stdout='', stderr='bad token')
        with mock.patch('serve.subprocess.run', return_value=fake):
            with self.assertRaises(serve.EngineError):
                serve.run_engine('bin', 'm.rbt2', [1], 1, 8, -1, [])


if __name__ == '__main__':
    unittest.main()
