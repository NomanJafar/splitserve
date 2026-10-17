"""A fake vLLM instance that speaks the disaggregated handshake, served live in a uvicorn thread.

A request carrying kv_transfer_params.do_remote_decode is treated as a prefill: it is answered with
one token and a kv_transfer_params block, the way a NixlConnector producer answers. Anything else is
decoded: a streamed chat completion of max_tokens chunks. Every request body is kept in app.state.seen
so tests can check what each side was sent."""

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def fake_vllm(name: str, itl_ms: float = 1.0, fail: bool = False) -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "m", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        app.state.seen.append(body)
        if fail:
            return JSONResponse({"error": f"{name} is broken"}, status_code=500)
        ktp = body.get("kv_transfer_params") or {}
        if ktp.get("do_remote_decode"):
            await asyncio.sleep(0.01)
            return {"id": "p", "object": "chat.completion", "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
                    "kv_transfer_params": {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": name,
                                           "remote_block_ids": [1, 2, 3], "remote_host": "127.0.0.1", "remote_port": 5600}}
        n = int(body.get("max_tokens", 4))
        if not body.get("stream"):
            return {"id": "d", "object": "chat.completion", "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "tok " * n}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}}

        async def gen():
            for i in range(n):
                await asyncio.sleep(itl_ms / 1000)
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': f'tok{i} '}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': n}})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def live(app):
    port = _port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if httpx.get(f"{url}/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.02)
    else:
        raise RuntimeError("server did not start")
    try:
        yield url
    finally:
        server.should_exit = True
        th.join(timeout=5)


@pytest.fixture
def live_server():
    return live
