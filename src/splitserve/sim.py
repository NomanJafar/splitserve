"""Event-driven model of monolithic and disaggregated serving (CONCEPTS.md sections 1-3, 5).

An instance advances in scheduler steps, the way vLLM does. A monolithic step carries one decode token for
every running sequence plus prefill chunks of newly admitted prompts up to a token budget, and lasts as
long as both pieces of work; a prefill-only instance carries only prompts; a decode-only instance only
decode tokens. Between a prefill instance and a decode instance the prompt's KV cache crosses a link with
a fixed cost plus bytes / bandwidth.

Every parameter is a number measured elsewhere in this portfolio (servellm: decode step vs batch size and
short-prompt prefill; chunkedprefill-lab: long-prompt prefill; kvcalc: bytes per KV token) or an explicit
assumption about the link. The model is deliberately simple: no KV-cache capacity limit (servellm never
got near it with this model), no preemption, no prefix caching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # prefill cost of a chunk: fixed + a*n + b*((done+n)^2 - done^2); the quadratic term is attention
    prefill_fixed_ms: float = 5.0
    prefill_ms_per_token: float = 0.10          # servellm: 256 tokens in ~32 ms on a T4, 1.5B fp16
    prefill_ms_per_token_sq: float = 1.5e-4      # fits chunkedprefill-lab's seconds-long prefill of 3,500-token prompts
    # decode step: base + per-sequence slope (servellm: 14.7 ms at 1 stream, 36.8 ms at 64)
    decode_base_ms: float = 14.3
    decode_ms_per_seq: float = 0.35
    # the link between a prefill and a decode instance
    kv_bytes_per_token: int = 28_672             # kvcalc: Qwen2.5-1.5B, fp16, 28 layers x 2 KV heads x 128 dims x 2 bytes x 2
    link_gb_s: float = 3.0                       # PCIe staged through host memory, no GPUDirect: the Kaggle case
    transfer_fixed_ms: float = 2.0
    # scheduler
    chunk_tokens: int = 2048                     # monolithic chunked-prefill budget per step
    prefill_budget_tokens: int = 8192            # a prefill-only instance's budget per step
    max_batch: int = 256                         # vLLM's default max-num-seqs


@dataclass
class Req:
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    kind: str = "chat"
    rid: int = 0


@dataclass
class _Run:
    req: Req
    available_s: float                   # when this instance may start on it
    prefilled: int = 0
    emitted: int = 0
    emit_times: list[float] = field(default_factory=list)
    done_s: float | None = None


def prefill_ms(n: int, cfg: SimConfig, done: int = 0, fixed: bool = True) -> float:
    """Cost of prefilling n more tokens of a prompt that already has `done` tokens in cache."""
    return (cfg.prefill_fixed_ms if fixed else 0.0) + cfg.prefill_ms_per_token * n + \
        cfg.prefill_ms_per_token_sq * ((done + n) ** 2 - done ** 2)


def transfer_ms(n: int, cfg: SimConfig) -> float:
    return cfg.transfer_fixed_ms + n * cfg.kv_bytes_per_token / (cfg.link_gb_s * 1e9) * 1000


def _percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    s = sorted(xs)

    def p(q):
        k = (len(s) - 1) * q / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {"n": len(s), "p50": p(50), "p95": p(95), "p99": p(99), "max": s[-1], "mean": sum(s) / len(s)}


def run_instance(runs: list[_Run], cfg: SimConfig, prefill: bool, decode: bool) -> list[_Run]:
    """Advance one instance until all its work is done. `runs` must be sorted by available_s.

    prefill & decode: monolithic (chunked prefill, decodes first within the budget);
    prefill only: a P instance (runs end when prefilled; emit nothing);
    decode only: a D instance (runs arrive fully prefilled; the first step emits the first token)."""
    budget_total = cfg.chunk_tokens if decode else cfg.prefill_budget_tokens
    pending = deque(runs)
    waiting: deque[_Run] = deque()
    running: list[_Run] = []
    t = 0.0
    while pending or waiting or running:
        while pending and pending[0].available_s <= t:
            waiting.append(pending.popleft())
        if not running and not waiting:
            t = pending[0].available_s
            continue
        while waiting and len(running) < cfg.max_batch:
            running.append(waiting.popleft())
        decoding = [r for r in running if r.prefilled >= r.req.prompt_tokens] if decode else []
        budget = budget_total - len(decoding)
        step_ms = (cfg.decode_base_ms + cfg.decode_ms_per_seq * len(decoding)) if decoding else 0.0
        finished_prefill: list[_Run] = []
        if prefill:
            first = True
            for r in running:
                if budget <= 0:
                    break
                remaining = r.req.prompt_tokens - r.prefilled
                if remaining <= 0:
                    continue
                take = min(budget, remaining)
                step_ms += prefill_ms(take, cfg, done=r.prefilled, fixed=first)
                first = False
                r.prefilled += take
                budget -= take
                if r.prefilled >= r.req.prompt_tokens:
                    finished_prefill.append(r)
        t += step_ms / 1000
        for r in decoding:
            r.emit_times.append(t)
            r.emitted += 1
        if decode:
            for r in finished_prefill:                  # the prefill iteration samples the first token
                r.emit_times.append(t)
                r.emitted += 1
        else:
            for r in finished_prefill:
                r.done_s = t
        still = []
        for r in running:
            if decode and r.emitted >= r.req.output_tokens:
                r.done_s = t
            elif r.done_s is None:
                still.append(r)
        running = still
    return runs


@dataclass
class SimResult:
    label: str
    gpus: int
    requests: int
    wall_s: float
    output_tokens: int
    output_tokens_per_s: float
    by_kind: dict
    per_request: list[dict] = field(default_factory=list)

    def row(self) -> dict:
        out = {"condition": self.label, "gpus": self.gpus, "requests": self.requests, "out_tok_s": round(self.output_tokens_per_s)}
        for kind, s in sorted(self.by_kind.items()):
            out[f"{kind}_ttft_p50_ms"] = round(s["ttft"]["p50"] * 1000) if s["ttft"]["p50"] is not None else None
            out[f"{kind}_itl_p50_ms"] = round(s["itl"]["p50"] * 1000, 1) if s["itl"]["p50"] is not None else None
            out[f"{kind}_itl_p99_ms"] = round(s["itl"]["p99"] * 1000) if s["itl"]["p99"] is not None else None
            out[f"{kind}_itl_max_ms"] = round(s["itl"]["max"] * 1000) if s["itl"]["max"] is not None else None
        return out


def _summarise(label: str, gpus: int, runs: list[_Run], first_arrival: float) -> SimResult:
    kinds: dict[str, dict[str, list[float]]] = {}
    per_request = []
    for r in runs:
        k = kinds.setdefault(r.req.kind, {"ttft": [], "itl": [], "e2e": [], "errors": []})
        if not r.emit_times:
            k["errors"].append(1.0)
            continue
        ttft = r.emit_times[0] - r.req.arrival_s
        k["ttft"].append(ttft)
        k["itl"].extend(b - a for a, b in zip(r.emit_times, r.emit_times[1:]))
        k["e2e"].append(r.emit_times[-1] - r.req.arrival_s)
        per_request.append({"rid": r.req.rid, "kind": r.req.kind, "arrival_s": r.req.arrival_s, "ttft_s": ttft,
                            "e2e_s": r.emit_times[-1] - r.req.arrival_s, "output_tokens": r.emitted})
    by_kind = {k: {"requests": len(v["ttft"]) + len(v["errors"]), "errors": len(v["errors"]), "ttft": _percentiles(v["ttft"]),
                   "itl": _percentiles(v["itl"]), "e2e": _percentiles(v["e2e"])} for k, v in kinds.items()}
    last = max((r.emit_times[-1] for r in runs if r.emit_times), default=first_arrival)
    wall = max(last - first_arrival, 1e-9)
    out_tokens = sum(r.emitted for r in runs)
    return SimResult(label, gpus, len(runs), wall, out_tokens, out_tokens / wall, by_kind, per_request)


def monolithic(reqs: list[Req], cfg: SimConfig, replicas: int = 1, label: str | None = None) -> SimResult:
    """`replicas` monolithic instances behind a round-robin router."""
    reqs = sorted(reqs, key=lambda r: r.arrival_s)
    shards: list[list[_Run]] = [[] for _ in range(replicas)]
    for i, r in enumerate(reqs):
        shards[i % replicas].append(_Run(r, r.arrival_s))
    runs: list[_Run] = []
    for shard in shards:
        runs += run_instance(shard, cfg, prefill=True, decode=True)
    return _summarise(label or (f"monolithic x{replicas}" if replicas > 1 else "monolithic"), replicas, runs,
                      reqs[0].arrival_s if reqs else 0.0)


def disaggregated(reqs: list[Req], cfg: SimConfig, n_prefill: int = 1, n_decode: int = 1, label: str | None = None) -> SimResult:
    """n_prefill P instances (round-robin), a link, n_decode D instances (round-robin)."""
    reqs = sorted(reqs, key=lambda r: r.arrival_s)
    p_shards: list[list[_Run]] = [[] for _ in range(n_prefill)]
    for i, r in enumerate(reqs):
        p_shards[i % n_prefill].append(_Run(r, r.arrival_s))
    handed: list[_Run] = []
    for shard in p_shards:
        for pr in run_instance(shard, cfg, prefill=True, decode=False):
            assert pr.done_s is not None
            handed.append(_Run(pr.req, pr.done_s + transfer_ms(pr.req.prompt_tokens, cfg) / 1000, prefilled=pr.req.prompt_tokens))
    handed.sort(key=lambda r: r.available_s)
    d_shards: list[list[_Run]] = [[] for _ in range(n_decode)]
    for i, r in enumerate(handed):
        d_shards[i % n_decode].append(r)
    runs: list[_Run] = []
    for shard in d_shards:
        runs += run_instance(shard, cfg, prefill=False, decode=True)
    return _summarise(label or f"{n_prefill}P + {n_decode}D", n_prefill + n_decode, runs, reqs[0].arrival_s if reqs else 0.0)
