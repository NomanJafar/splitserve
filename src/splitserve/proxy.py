"""The P/D proxy: one OpenAI-compatible endpoint in front of prefill and decode instances (CONCEPTS.md section 4).

Modes:
  nixl      prefill request carries kv_transfer_params.do_remote_decode and max_tokens=1; the prefiller's
            response carries the block ids and side-channel address, which are attached to the decode request
            (vLLM NixlConnector handshake, as in vLLM's own toy proxy).
  shared    prefill request is the same request with max_tokens=1 and no transfer params; the decode instance
            finds the KV by token hash (vLLM ExampleConnector, KV via a shared directory).
  replicas  no prefill hop: round-robin across the decode URLs (two monolithic replicas, the fair 2-GPU baseline).

Every request is timed at each hop (prefill_ms: the P round-trip; handoff_ms: from sending the decode request
to its first byte, which includes the KV transfer) and kept in memory; /records returns them and /metrics
summarises them. Response headers name the instances used and the prefill time.
"""

from __future__ import annotations

import itertools
import time
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

MODES = ("nixl", "shared", "replicas")
_REMOTE_DECODE = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                  "remote_block_ids": None, "remote_host": None, "remote_port": None}


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def prefill_body(body: dict, mode: str) -> dict:
    """The one-token request sent to the prefill instance."""
    b = dict(body)
    b["stream"] = False
    b.pop("stream_options", None)
    b["max_tokens"] = 1
    if "max_completion_tokens" in b:
        b["max_completion_tokens"] = 1
    if mode == "nixl":
        b["kv_transfer_params"] = dict(_REMOTE_DECODE)
    return b


def decode_body(body: dict, prefill_response: dict | None, mode: str) -> dict:
    b = dict(body)
    if mode == "nixl" and prefill_response:
        ktp = prefill_response.get("kv_transfer_params")
        if ktp:
            b["kv_transfer_params"] = ktp
    return b


def create_app(prefill_urls: list[str], decode_urls: list[str], mode: str = "nixl", max_records: int = 200_000) -> FastAPI:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode != "replicas" and not prefill_urls:
        raise ValueError("nixl/shared modes need at least one prefill URL")
    if not decode_urls:
        raise ValueError("at least one decode URL is required")
    prefill_cycle = itertools.cycle(prefill_urls) if prefill_urls else None
    decode_cycle = itertools.cycle(decode_urls)
    app = FastAPI(title="splitserve")
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
    records: deque = deque(maxlen=max_records)
    counters = {"requests": 0, "errors": 0}
    app.state.records = records

    async def handle(request: Request, path: str):
        body = await request.json()
        counters["requests"] += 1
        t0 = time.perf_counter()
        rec = {"t": time.time(), "mode": mode, "ok": False, "stage": None, "prefill_url": None, "decode_url": None,
               "prefill_ms": None, "handoff_ms": None, "total_ms": None}
        records.append(rec)
        headers = {"x-splitserve-mode": mode}
        pre_json = None
        if mode != "replicas":
            purl = next(prefill_cycle)
            rec["prefill_url"] = purl
            headers["x-splitserve-prefill-url"] = purl
            try:
                pr = await client.post(f"{purl}{path}", json=prefill_body(body, mode))
            except httpx.HTTPError as e:
                rec["stage"] = "prefill"
                counters["errors"] += 1
                return JSONResponse({"error": f"prefill unreachable: {type(e).__name__}"}, status_code=502, headers=headers)
            if pr.status_code != 200:
                rec["stage"] = "prefill"
                counters["errors"] += 1
                return JSONResponse({"error": pr.text[:500], "stage": "prefill"}, status_code=502, headers=headers)
            pre_json = pr.json()
            rec["prefill_ms"] = (time.perf_counter() - t0) * 1000
            headers["x-splitserve-prefill-ms"] = f"{rec['prefill_ms']:.1f}"
        durl = next(decode_cycle)
        rec["decode_url"] = durl
        headers["x-splitserve-decode-url"] = durl
        t1 = time.perf_counter()
        upstream = client.build_request("POST", f"{durl}{path}", json=decode_body(body, pre_json, mode))
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            rec["stage"] = "decode"
            counters["errors"] += 1
            return JSONResponse({"error": f"decode unreachable: {type(e).__name__}"}, status_code=502, headers=headers)
        if resp.status_code != 200:
            data = await resp.aread()
            await resp.aclose()
            rec["stage"] = "decode"
            counters["errors"] += 1
            return JSONResponse({"error": data.decode(errors="replace")[:500], "stage": "decode"}, status_code=resp.status_code, headers=headers)
        if body.get("stream"):
            async def gen():
                first = True
                try:
                    async for chunk in resp.aiter_raw():
                        if first:
                            rec["handoff_ms"] = (time.perf_counter() - t1) * 1000
                            rec["ok"] = True
                            first = False
                        yield chunk
                finally:
                    rec["total_ms"] = (time.perf_counter() - t0) * 1000
                    await resp.aclose()
            return StreamingResponse(gen(), media_type=resp.headers.get("content-type", "text/event-stream"), headers=headers)
        data = await resp.aread()
        await resp.aclose()
        rec["handoff_ms"] = (time.perf_counter() - t1) * 1000
        rec["total_ms"] = (time.perf_counter() - t0) * 1000
        rec["ok"] = True
        return JSONResponse(httpx.Response(200, content=data).json(), headers=headers)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, "/v1/completions")

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": mode, "prefill": prefill_urls, "decode": decode_urls}

    @app.get("/v1/models")
    async def models():
        r = await client.get(f"{decode_urls[0]}/v1/models", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.get("/records")
    async def get_records(clear: bool = False):
        out = list(records)
        if clear:
            records.clear()
        return JSONResponse(out)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        ok = [r for r in records if r["ok"]]
        lines = [f"splitserve_requests_total {counters['requests']}", f"splitserve_errors_total {counters['errors']}"]
        for name in ("prefill_ms", "handoff_ms", "total_ms"):
            xs = [r[name] for r in ok if r[name] is not None]
            for q in (0.5, 0.95, 0.99):
                v = _pct(xs, q * 100)
                lines.append(f'splitserve_{name}{{quantile="{q}"}} {v if v is not None else "NaN"}')
        return "\n".join(lines) + "\n"

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="prefill/decode disaggregation proxy for vLLM instances")
    p.add_argument("--prefill", action="append", default=[], help="prefill instance base URL (repeat)")
    p.add_argument("--decode", action="append", required=True, help="decode instance base URL (repeat; replicas in --mode replicas)")
    p.add_argument("--mode", choices=MODES, default="nixl")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    uvicorn.run(create_app(a.prefill, a.decode, a.mode), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
