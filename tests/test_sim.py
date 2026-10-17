import itertools

from splitserve.sim import Req, SimConfig, disaggregated, monolithic, prefill_ms, transfer_ms
from splitserve.trace import mixed_trace


def _decode_only(n=8, out=50):
    return [Req(arrival_s=0.0, prompt_tokens=16, output_tokens=out, kind="chat", rid=i) for i in range(n)]


def test_decode_step_time_follows_the_calibrated_line():
    cfg = SimConfig()
    res = monolithic(_decode_only(8), cfg)
    expect = (cfg.decode_base_ms + cfg.decode_ms_per_seq * 8) / 1000
    itl = res.by_kind["chat"]["itl"]
    assert abs(itl["p50"] - expect) < 0.002
    assert res.requests == 8 and res.by_kind["chat"]["errors"] == 0


def test_prefill_cost_has_linear_and_quadratic_terms():
    cfg = SimConfig()
    assert prefill_ms(256, cfg) < prefill_ms(1500, cfg) < prefill_ms(3500, cfg)
    # the quadratic term makes long prompts cost more per token than short ones
    assert prefill_ms(3500, cfg) / 3500 > prefill_ms(256, cfg) / 256


def test_monolithic_long_prompt_stalls_interactive_users_and_disaggregation_removes_it():
    cfg = SimConfig(chunk_tokens=8192)
    reqs = _decode_only(8, out=200) + [Req(arrival_s=1.0, prompt_tokens=3500, output_tokens=8, kind="long", rid=99)]
    mono = monolithic(reqs, cfg)
    stall = mono.by_kind["chat"]["itl"]["max"]
    assert stall > 0.8 * prefill_ms(3500, cfg) / 1000          # one iteration carries the whole prompt
    split = disaggregated(reqs, cfg, n_prefill=1, n_decode=1)
    assert split.by_kind["chat"]["itl"]["max"] < 2 * (cfg.decode_base_ms + cfg.decode_ms_per_seq * 9) / 1000
    # p50 barely moves either way
    assert abs(mono.by_kind["chat"]["itl"]["p50"] - split.by_kind["chat"]["itl"]["p50"]) < 0.005


def test_chunking_resizes_the_stall_but_does_not_remove_it():
    reqs = _decode_only(8, out=200) + [Req(arrival_s=1.0, prompt_tokens=3500, output_tokens=8, kind="long", rid=99)]
    stalls = [monolithic(reqs, SimConfig(chunk_tokens=b)).by_kind["chat"]["itl"]["max"] for b in (512, 2048, 8192)]
    assert stalls[0] < stalls[1] < stalls[2]
    assert stalls[0] > 3 * (SimConfig().decode_base_ms + 8 * SimConfig().decode_ms_per_seq) / 1000


def test_transfer_adds_to_ttft_and_shrinks_with_link_bandwidth():
    reqs = [Req(arrival_s=0.0, prompt_tokens=1500, output_tokens=16, kind="long", rid=0)]
    slow = disaggregated(reqs, SimConfig(link_gb_s=1.0))
    fast = disaggregated(reqs, SimConfig(link_gb_s=200.0))
    assert transfer_ms(1500, SimConfig(link_gb_s=1.0)) > 40 > transfer_ms(1500, SimConfig(link_gb_s=200.0))
    assert slow.by_kind["long"]["ttft"]["p50"] > fast.by_kind["long"]["ttft"]["p50"]
    assert fast.by_kind["long"]["ttft"]["p50"] > prefill_ms(1500, SimConfig()) / 1000


def test_two_monolithic_replicas_out_produce_one_prefill_plus_one_decode_under_saturation():
    cfg = SimConfig()
    reqs = mixed_trace(duration_s=20, chat_rps=40, long_rps=0.5, seed=1)   # saturating
    two = monolithic(reqs, cfg, replicas=2)
    split = disaggregated(reqs, cfg, n_prefill=1, n_decode=1)
    assert two.output_tokens_per_s >= split.output_tokens_per_s
    assert two.gpus == split.gpus == 2


def test_ratio_sweep_moves_prefill_bottleneck_with_more_prefillers():
    cfg = SimConfig()
    reqs = mixed_trace(duration_s=20, chat_rps=1, long_rps=3, long_prompt=3000, seed=2)   # prefill-heavy
    one = disaggregated(reqs, cfg, n_prefill=1, n_decode=3)
    three = disaggregated(reqs, cfg, n_prefill=3, n_decode=1)
    assert three.by_kind["long"]["ttft"]["p50"] < one.by_kind["long"]["ttft"]["p50"]


def test_trace_is_deterministic_and_shaped():
    a, b = mixed_trace(10, 2, 0.5, seed=3), mixed_trace(10, 2, 0.5, seed=3)
    assert [(r.arrival_s, r.kind) for r in a] == [(r.arrival_s, r.kind) for r in b]
    assert {r.kind for r in a} == {"chat", "long"} and all(r.arrival_s <= 10 for r in a)
    assert all(r.arrival_s <= s.arrival_s for r, s in itertools.pairwise(a))
