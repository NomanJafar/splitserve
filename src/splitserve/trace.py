"""Synthetic request traces: a short-prompt interactive stream and a long-prompt stream, Poisson arrivals."""

from __future__ import annotations

import random

from splitserve.sim import Req


def poisson_arrivals(rate_per_s: float, duration_s: float, rng: random.Random) -> list[float]:
    if rate_per_s <= 0:
        return []
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate_per_s)
        if t > duration_s:
            return out
        out.append(t)


def mixed_trace(duration_s: float, chat_rps: float, long_rps: float, chat_prompt: int = 96, chat_out: int = 128,
                long_prompt: int = 1500, long_out: int = 48, seed: int = 0) -> list[Req]:
    rng = random.Random(seed)
    reqs = [Req(t, chat_prompt, chat_out, "chat") for t in poisson_arrivals(chat_rps, duration_s, rng)]
    reqs += [Req(t, long_prompt, long_out, "long") for t in poisson_arrivals(long_rps, duration_s, rng)]
    reqs.sort(key=lambda r: r.arrival_s)
    for i, r in enumerate(reqs):
        r.rid = i
    return reqs
