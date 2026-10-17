import httpx

from splitserve.proxy import create_app
from tests.conftest import fake_vllm, live


def _chat(url, stream=True, max_tokens=3):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens, "stream": stream}
    with httpx.Client(timeout=10) as c:
        if stream:
            with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
                chunks = [line for line in r.iter_lines() if line.startswith("data:")]
                return r, chunks
        r = c.post(f"{url}/v1/chat/completions", json=body)
        return r, r.json()


def test_nixl_mode_threads_kv_transfer_params_from_prefill_to_decode():
    p, d = fake_vllm("P1"), fake_vllm("D1")
    with live(p) as pu, live(d) as du, live(create_app([pu], [du], mode="nixl")) as gw:
        r, chunks = _chat(gw)
        assert r.status_code == 200
        assert len(chunks) == 3 + 2            # 3 tokens, usage, [DONE]
        assert r.headers["x-splitserve-prefill-url"] == pu and r.headers["x-splitserve-decode-url"] == du
        assert float(r.headers["x-splitserve-prefill-ms"]) > 0
        # prefill side: one token, not streamed, asked to hand off
        pre = p.state.seen[0]
        assert pre["max_tokens"] == 1 and pre["stream"] is False and pre["kv_transfer_params"]["do_remote_decode"] is True
        # decode side: the original request plus what the prefiller returned
        dec = d.state.seen[0]
        assert dec["max_tokens"] == 3 and dec["stream"] is True
        assert dec["kv_transfer_params"]["remote_engine_id"] == "P1" and dec["kv_transfer_params"]["remote_block_ids"] == [1, 2, 3]
        rec = httpx.get(f"{gw}/records").json()
        assert len(rec) == 1 and rec[0]["prefill_ms"] > 0 and rec[0]["handoff_ms"] >= 0 and rec[0]["ok"] is True


def test_shared_mode_sends_one_token_prefill_without_kv_params():
    p, d = fake_vllm("P1"), fake_vllm("D1")
    with live(p) as pu, live(d) as du, live(create_app([pu], [du], mode="shared")) as gw:
        r, chunks = _chat(gw)
        assert r.status_code == 200 and len(chunks) == 5
        pre = p.state.seen[0]
        assert pre["max_tokens"] == 1 and "kv_transfer_params" not in pre and pre["stream"] is False
        assert "kv_transfer_params" not in d.state.seen[0]


def test_replicas_mode_round_robins_without_a_prefill_hop():
    a, b = fake_vllm("A"), fake_vllm("B")
    with live(a) as au, live(b) as bu, live(create_app([], [au, bu], mode="replicas")) as gw:
        urls = [_chat(gw)[0].headers["x-splitserve-decode-url"] for _ in range(4)]
        assert urls == [au, bu, au, bu]
        assert len(a.state.seen) == 2 and len(b.state.seen) == 2
        assert all("kv_transfer_params" not in s and s["max_tokens"] == 3 for s in a.state.seen + b.state.seen)
        assert "x-splitserve-prefill-url" not in _chat(gw)[0].headers


def test_non_streaming_request_returns_json():
    p, d = fake_vllm("P1"), fake_vllm("D1")
    with live(p) as pu, live(d) as du, live(create_app([pu], [du], mode="nixl")) as gw:
        r, data = _chat(gw, stream=False, max_tokens=2)
        assert r.status_code == 200 and data["usage"]["completion_tokens"] == 2
        assert d.state.seen[0]["kv_transfer_params"]["remote_engine_id"] == "P1"


def test_prefill_failure_is_a_502_and_decode_is_never_called():
    p, d = fake_vllm("P1", fail=True), fake_vllm("D1")
    with live(p) as pu, live(d) as du, live(create_app([pu], [du], mode="nixl")) as gw:
        r, _ = _chat(gw, stream=False)
        assert r.status_code == 502
        assert d.state.seen == []
        rec = httpx.get(f"{gw}/records").json()
        assert rec[0]["ok"] is False and rec[0]["stage"] == "prefill"


def test_metrics_exposes_counts_and_handoff_percentiles():
    p, d = fake_vllm("P1"), fake_vllm("D1")
    with live(p) as pu, live(d) as du, live(create_app([pu], [du], mode="nixl")) as gw:
        for _ in range(3):
            _chat(gw)
        text = httpx.get(f"{gw}/metrics").text
        assert "splitserve_requests_total 3" in text
        assert "splitserve_prefill_ms{quantile=\"0.5\"}" in text and "splitserve_handoff_ms{quantile=\"0.5\"}" in text
