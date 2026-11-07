"""Minimal OpenAI-compatible local HTTP server for a .rbt2 model.

Uses the local Rabbit2 engine for greedy generation (one engine process per request).
This is a usability layer, not a production server. No auth, no streaming, no sampling.

    python tools/serve.py --model model.rbt2 --tok <tokenizer> --host 127.0.0.1 --port 8000

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions  (messages -> greedy completion)
  POST /v1/completions       (prompt -> greedy completion)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EngineError(RuntimeError):
    pass


def run_engine(binary, model, ids, n, ctx, eos, extra):
    cmd = [binary, model, "--gen", str(n), "--ctx", str(ctx)] + list(extra)
    if eos >= 0:
        cmd += ["--eos", str(eos)]
    cmd += [str(i) for i in ids]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise EngineError((res.stderr or res.stdout or f"engine exited {res.returncode}").strip())
    return [int(t) for t in res.stdout.split() if t.strip()], res.stderr


def messages_to_prompt(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{role.capitalize()}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    server_version = "Rabbit2Serve/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        cfg = self.server.cfg
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": cfg["model_id"], "object": "model", "owned_by": "local"}]})
        if self.path in ("/", "/health"):
            return self._json(200, {"status": "ok", "model": cfg["model_id"]})
        return self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        try:
            payload = self._read_json()
        except Exception as e:
            return self._json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request_error"}})
        if self.path.rstrip("/") == "/v1/chat/completions":
            return self._chat(payload)
        if self.path.rstrip("/") == "/v1/completions":
            return self._completion(payload)
        return self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _generate(self, prompt, max_tokens):
        cfg = self.server.cfg
        tok = cfg["tokenizer"]
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        ids = ids[-cfg["max_history_tokens"]:]
        ctx = min(cfg["ctx"] or (len(ids) + max_tokens), cfg["max_seq"] or (len(ids) + max_tokens))
        if len(ids) >= ctx:
            ids = ids[-max(1, ctx - 1):]
        new_ids, timing = run_engine(cfg["binary"], cfg["model"], ids, max_tokens, ctx, cfg["eos"], cfg["extra"])
        text = tok.decode(new_ids, skip_special_tokens=True)
        return text, len(ids), len(new_ids), timing

    def _chat(self, payload):
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": {"message": "messages must be a non-empty list", "type": "invalid_request_error"}})
        if payload.get("stream"):
            return self._json(400, {"error": {"message": "stream=true is not supported", "type": "invalid_request_error"}})
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or self.server.cfg["n"])
        try:
            text, ptok, ctok, timing = self._generate(messages_to_prompt(messages), max_tokens)
        except EngineError as e:
            return self._json(500, {"error": {"message": str(e), "type": "server_error"}})
        now = int(time.time())
        self._json(200, {
            "id": f"chatcmpl-rabbit2-{now}",
            "object": "chat.completion",
            "created": now,
            "model": self.server.cfg["model_id"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok},
            "timing_stderr": timing.strip(),
        })

    def _completion(self, payload):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            return self._json(400, {"error": {"message": "prompt must be a string", "type": "invalid_request_error"}})
        if payload.get("stream"):
            return self._json(400, {"error": {"message": "stream=true is not supported", "type": "invalid_request_error"}})
        max_tokens = int(payload.get("max_tokens") or self.server.cfg["n"])
        try:
            text, ptok, ctok, timing = self._generate(prompt, max_tokens)
        except EngineError as e:
            return self._json(500, {"error": {"message": str(e), "type": "server_error"}})
        now = int(time.time())
        self._json(200, {
            "id": f"cmpl-rabbit2-{now}",
            "object": "text_completion",
            "created": now,
            "model": self.server.cfg["model_id"],
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok},
            "timing_stderr": timing.strip(),
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True, help="tokenizer id or local folder")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=0, help="optional hard max context cap")
    ap.add_argument("--max-history-tokens", type=int, default=2048)
    ap.add_argument("--model-id", default="rabbit2-local")
    ap.add_argument("--expert-cache-mib", type=int, default=None)
    ap.add_argument("--kernel", choices=["scalar", "auto", "avx2"], default=None)
    ap.add_argument("--bin", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    exe = "rabbit2.exe" if os.name == "nt" else "rabbit2"
    binary = args.bin or os.path.join(ROOT, "build", exe)
    if not os.path.exists(binary):
        sys.exit(f"engine not built: {binary}")
    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    extra = []
    if args.expert_cache_mib is not None:
        extra += ["--expert-cache-mib", str(args.expert_cache_mib)]
    if args.kernel is not None:
        extra += ["--kernel", args.kernel]

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.cfg = {
        "binary": binary, "model": args.model, "tokenizer": tokenizer, "eos": eos,
        "n": args.n, "ctx": args.ctx, "max_seq": args.max_seq, "extra": extra,
        "model_id": args.model_id, "max_history_tokens": args.max_history_tokens,
    }
    print(f"serving {args.model_id} on http://{args.host}:{args.port} (greedy, non-streaming)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
