"""Turn tokenbench sweep JSON and the proxy's /records into comparison rows and Markdown."""

from __future__ import annotations


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ms(x) -> float | None:
    return None if x is None else round(x * 1000, 1)


def summarise_load(data: dict, level_index: int = 0) -> dict:
    """One row from a tokenbench sweep result (one level)."""
    lv = data["levels"][level_index]
    return {"requests": lv["requests"], "errors": lv["errors"], "out_tok_s": round(lv["output_tokens_per_s"]),
            "ttft_p50_ms": _ms(lv["ttft"]["p50"]), "ttft_p95_ms": _ms(lv["ttft"]["p95"]), "ttft_p99_ms": _ms(lv["ttft"].get("p99")),
            "itl_p50_ms": _ms(lv["itl"]["p50"]), "itl_p95_ms": _ms(lv["itl"]["p95"]), "itl_p99_ms": _ms(lv["itl"].get("p99")),
            "itl_max_ms": _ms(lv["itl"].get("max"))}


def summarise_records(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("ok")]
    pre = [r["prefill_ms"] for r in ok if r.get("prefill_ms") is not None]
    hand = [r["handoff_ms"] for r in ok if r.get("handoff_ms") is not None]
    return {"n": len(ok), "failed": len(records) - len(ok),
            "prefill_p50_ms": _pct(pre, 50), "prefill_p95_ms": _pct(pre, 95),
            "handoff_p50_ms": _pct(hand, 50), "handoff_p95_ms": _pct(hand, 95)}


def comparison_table(rows: list[dict], columns: list[str]) -> str:
    def cell(v):
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
        return str(v)
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(r.get(c)) for c in columns) + " |" for r in rows) + "\n"
