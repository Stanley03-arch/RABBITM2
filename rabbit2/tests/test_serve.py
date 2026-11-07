"""Tests for serve.py helpers (no engine/tokenizer/model required)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import serve  # noqa: E402


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


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
        self.assertIn('--gen', cmd)
        self.assertIn('--eos', cmd)
        self.assertEqual(ids, [5, 6])
        self.assertEqual(err, 'timing\n')

    def test_run_engine_failure(self):
        fake = mock.Mock(returncode=1, stdout='', stderr='bad token')
        with mock.patch('serve.subprocess.run', return_value=fake):
            with self.assertRaises(serve.EngineError):
                serve.run_engine('bin', 'm.rbt2', [1], 1, 8, -1, [])

    def test_chat_stream_chunks(self):
        chunks = list(serve.iter_chat_chunks('m', FakeTok(), [0, 1, 2]))
        self.assertEqual(chunks[0]['choices'][0]['delta'].get('role'), 'assistant')
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
        self.assertEqual(text, 'abc')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_completion_stream_chunks(self):
        chunks = list(serve.iter_completion_chunks('m', FakeTok(), [0, 1]))
        text = ''.join(c['choices'][0].get('text', '') for c in chunks)
        self.assertEqual(text, 'ab')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_engine_cmd_sampling(self):
        cmd = serve.engine_cmd('b', 'm', [1], 2, 8, -1, ['--temperature', '0.8', '--seed', '7'])
        self.assertEqual(cmd[:6], ['b', 'm', '--gen', '2', '--ctx', '8'])
        self.assertNotIn('--eos', cmd)
        self.assertIn('--temperature', cmd)


if __name__ == '__main__':
    unittest.main()
