"""Minimal OpenAI-compatible local HTTP server for a .rbt2 model.

Runs the local Rabbit2 engine per request (greedy by default; sampling flags supported).
Usability layer, not production: no auth, local trusted use. Supports SSE streaming
(`stream=true`) by reading engine token ids as they are produced.

    python tools/serve.py --model model.rbt2 --tok <tokenizer> --host 127.0.0.1 --port 8000

Endpoints:
  GET  /, /health, /v1/models
  POST /v1/chat/completions
  POST /v1/completions
"""
import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EngineError(RuntimeError):
    pass


def engine_cmd(binary, model, ids, n, ctx, eos, extra):
    cmd = [binary, model, "--gen", str(n), "--ctx", str(ctx)] + list(extra)
    if eos >= 0:
        cmd += ["--eos", str(eos)]
    cmd += [str(i) for i in ids]
    return cmd


def run_engine(binary, model, ids, n, ctx, eos, extra):
    cmd = engine_cmd(binary, model, ids, n, ctx, eos, extra)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise EngineError((res.stderr or res.stdout or f"engine exited {res.returncode}").strip())
    return [int(t) for t in res.stdout.split() if t.strip()], res.stderr


def stream_engine(binary, model, ids, n, ctx, eos, extra):
    cmd = engine_cmd(binary, model, ids, n, ctx, eos, extra)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield int(line)
        except ValueError:
            continue
    proc.wait()
    stderr = (proc.stderr.read() if proc.stderr else "") or ""
    if proc.returncode != 0:
        raise EngineError(stderr.strip() or f"engine exited {proc.returncode}")
    return stderr


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


def _decode_delta(tokenizer, ids, token):
    before = tokenizer.decode(ids, skip_special_tokens=True)
    ids.append(int(token))
    after = tokenizer.decode(ids, skip_special_tokens=True)
    return after[len(before):] if after.startswith(before) else after


def iter_chat_chunks(model_id, tokenizer, token_iter):
    rid = f"chatcmpl-rabbit2-{int(time.time())}"
    created = int(time.time())
    ids = []
    yield {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
           "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    for tok in token_iter:
        delta = _decode_delta(tokenizer, ids, tok)
        if delta:
            yield {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                   "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
    yield {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def iter_completion_chunks(model_id, tokenizer, token_iter):
    rid = f"cmpl-rabbit2-{int(time.time())}"
    created = int(time.time())
    ids = []
    for tok in token_iter:
        text = _decode_delta(tokenizer, ids, tok)
        if text:
            yield {"id": rid, "object": "text_completion", "created": created, "model": model_id,
                   "choices": [{"index": 0, "text": text, "finish_reason": None}]}
    yield {"id": rid, "object": "text_completion", "created": created, "model": model_id,
           "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}


class Handler(BaseHTTPRequestHandler):
    server_version = "Rabbit2Serve/0.2"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for obj in chunks:
                self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def _prepare(self, prompt, max_tokens):
        cfg = self.server.cfg
        tok = cfg["tokenizer"]
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        ids = ids[-cfg["max_history_tokens"]:]
        ctx = min(cfg["ctx"] or (len(ids) + max_tokens), cfg["max_seq"] or (len(ids) + max_tokens))
        if len(ids) >= ctx:
            ids = ids[-max(1, ctx - 1):]
        return ids, ctx, len(ids)

    def do_GET(self):
        cfg = self.server.cfg
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": cfg["model_id"], "object": "model", "owned_by": "local"}]})
        if self.path in ("/", "/health"):
            return self._json(200, {"status": "ok", "model": cfg["model_id"], "streaming": True})
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
        ids, ctx, ptok = self._prepare(prompt, max_tokens)
        new_ids, timing = run_engine(cfg["binary"], cfg["model"], ids, max_tokens, ctx, cfg["eos"], cfg["extra"])
        text = cfg["tokenizer"].decode(new_ids, skip_special_tokens=True)
        return text, ptok, len(new_ids), timing

    def _chat(self, payload):
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": {"message": "messages must be a non-empty list", "type": "invalid_request_error"}})
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or self.server.cfg["n"])
        prompt = messages_to_prompt(messages)
        if payload.get("stream"):
            try:
                ids, ctx, _ = self._prepare(prompt, max_tokens)
            except Exception as e:
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            cfg = self.server.cfg
            tokens = stream_engine(cfg["binary"], cfg["model"], ids, max_tokens, ctx, cfg["eos"], cfg["extra"])
            return self._sse(iter_chat_chunks(cfg["model_id"], cfg["tokenizer"], tokens))
        try:
            text, ptok, ctok, timing = self._generate(prompt, max_tokens)
        except EngineError as e:
            return self._json(500, {"error": {"message": str(e), "type": "server_error"}})
        now = int(time.time())
        self._json(200, {
            "id": f"chatcmpl-rabbit2-{now}", "object": "chat.completion", "created": now,
            "model": self.server.cfg["model_id"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok},
            "timing_stderr": timing.strip(),
        })

    def _completion(self, payload):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            return self._json(400, {"error": {"message": "prompt must be a string", "type": "invalid_request_error"}})
        max_tokens = int(payload.get("max_tokens") or self.server.cfg["n"])
        if payload.get("stream"):
            try:
                ids, ctx, _ = self._prepare(prompt, max_tokens)
            except Exception as e:
                return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            cfg = self.server.cfg
            tokens = stream_engine(cfg["binary"], cfg["model"], ids, max_tokens, ctx, cfg["eos"], cfg["extra"])
            return self._sse(iter_completion_chunks(cfg["model_id"], cfg["tokenizer"], tokens))
        try:
            text, ptok, ctok, timing = self._generate(prompt, max_tokens)
        except EngineError as e:
            return self._json(500, {"error": {"message": str(e), "type": "server_error"}})
        now = int(time.time())
        self._json(200, {
            "id": f"cmpl-rabbit2-{now}", "object": "text_completion", "created": now,
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
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0, dest="top_k")
    ap.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    ap.add_argument("--seed", type=int, default=0)
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
    if args.temperature and args.temperature > 0:
        extra += ["--temperature", str(args.temperature)]
    if args.top_k:
        extra += ["--top-k", str(args.top_k)]
    if args.top_p < 1.0:
        extra += ["--top-p", str(args.top_p)]
    if args.seed:
        extra += ["--seed", str(args.seed)]

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.cfg = {
        "binary": binary, "model": args.model, "tokenizer": tokenizer, "eos": eos,
        "n": args.n, "ctx": args.ctx, "max_seq": args.max_seq, "extra": extra,
        "model_id": args.model_id, "max_history_tokens": args.max_history_tokens,
    }
    print(f"serving {args.model_id} on http://{args.host}:{args.port} (SSE streaming supported)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
