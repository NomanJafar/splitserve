from splitserve.analysis import comparison_table, summarise_load, summarise_records


def _level(**kw):
    base = {"requests": 10, "errors": 0, "output_tokens_per_s": 100.0, "wall_s": 10.0, "completion_tokens": 1000,
            "ttft": {"p50": 0.1, "p95": 0.2, "p99": 0.3, "max": 0.4},
            "itl": {"p50": 0.03, "p95": 0.05, "p99": 0.1, "max": 0.5}}
    base.update(kw)
    return base


def test_summarise_load_picks_latency_fields_in_ms():
    row = summarise_load({"config": {}, "levels": [_level()]})
    assert row["requests"] == 10 and row["ttft_p50_ms"] == 100 and row["itl_max_ms"] == 500 and row["out_tok_s"] == 100


def test_summarise_records_gives_prefill_and_handoff_percentiles():
    recs = [{"ok": True, "prefill_ms": 10 + i, "handoff_ms": 5 + i, "total_ms": 50} for i in range(10)] + [{"ok": False, "stage": "prefill"}]
    s = summarise_records(recs)
    assert s["n"] == 10 and s["failed"] == 1 and s["prefill_p50_ms"] == 14.5 and s["handoff_p95_ms"] >= 13


def test_comparison_table_is_markdown_with_one_row_per_condition():
    rows = [{"condition": "a", "x": 1}, {"condition": "b", "x": 2}]
    md = comparison_table(rows, ["condition", "x"])
    assert md.splitlines()[0] == "| condition | x |" and len(md.strip().splitlines()) == 4
