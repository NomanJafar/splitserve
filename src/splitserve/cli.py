"""`splitserve-sim`: run the simulator scenarios and write tables and plots to a results directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from splitserve.analysis import comparison_table
from splitserve.sim import SimConfig, SimResult, disaggregated, monolithic, prefill_ms, transfer_ms
from splitserve.trace import mixed_trace

COLS = ["condition", "gpus", "requests", "out_tok_s", "chat_ttft_p50_ms", "chat_itl_p50_ms", "chat_itl_p99_ms", "chat_itl_max_ms",
        "long_ttft_p50_ms"]


def scenario_stall(cfg: SimConfig, duration: float, seed: int) -> list[dict]:
    """chunkedprefill-lab's experiment: interactive stream + 3,500-token prompts, chunk budget swept."""
    reqs = mixed_trace(duration, chat_rps=4, long_rps=0.25, chat_prompt=64, chat_out=128, long_prompt=3500, long_out=32, seed=seed)
    rows = []
    for budget in (512, 1024, 2048, 4096, 8192):
        c = SimConfig(**{**asdict(cfg), "chunk_tokens": budget})
        rows.append({"budget": budget, **monolithic(reqs, c, label=f"budget={budget}").row()})
    only = [r for r in reqs if r.kind == "chat"]
    rows.append({"budget": None, **monolithic(only, cfg, label="interactive only").row()})
    rows.append({"budget": None, **disaggregated(reqs, cfg, 1, 1, label="1P + 1D").row()})
    return rows


def scenario_two_gpus(cfg: SimConfig, duration: float, seed: int, chat_rps: float, long_rps: float) -> list[SimResult]:
    reqs = mixed_trace(duration, chat_rps=chat_rps, long_rps=long_rps, seed=seed)
    out = [monolithic(reqs, cfg, 1), monolithic(reqs, cfg, 2)]
    for link, name in ((cfg.link_gb_s, f"1P + 1D, {cfg.link_gb_s:g} GB/s link"), (25.0, "1P + 1D, 25 GB/s link"), (200.0, "1P + 1D, 200 GB/s link")):
        c = SimConfig(**{**asdict(cfg), "link_gb_s": link})
        out.append(disaggregated(reqs, c, 1, 1, label=name))
    return out


def scenario_ratio(cfg: SimConfig, duration: float, seed: int) -> dict[str, list[SimResult]]:
    loads = {"chat-heavy (12 req/s x 96->128, 0.5 req/s x 1500->48)": {"chat_rps": 12, "long_rps": 0.5},
             "prefill-heavy (2 req/s x 96->128, 1 req/s x 3000->32)": {"chat_rps": 2, "long_rps": 1, "long_prompt": 3000, "long_out": 32}}
    out = {}
    for name, kw in loads.items():
        reqs = mixed_trace(duration, seed=seed, **kw)
        res = [monolithic(reqs, cfg, 4)]
        for p in (1, 2, 3):
            res.append(disaggregated(reqs, cfg, p, 4 - p))
        out[name] = res
    return out


def scenario_link(cfg: SimConfig) -> list[dict]:
    rows = []
    for n in (256, 1500, 3500, 8000):
        for link in (1, 3, 12, 25, 50, 200, 400):
            rows.append({"prompt_tokens": n, "link_gb_s": link, "prefill_ms": round(prefill_ms(n, cfg)),
                         "transfer_ms": round(transfer_ms(n, SimConfig(**{**asdict(cfg), "link_gb_s": link})), 1)})
    return rows


def _plots(out: Path, stall: list[dict], two: list[SimResult], ratio: dict, link: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    b = [r for r in stall if r["budget"]]
    ax.plot([r["budget"] for r in b], [r["chat_itl_max_ms"] / 1000 for r in b], marker="o", label="monolithic: interactive ITL max")
    ax.plot([r["budget"] for r in b], [r["chat_itl_p99_ms"] / 1000 for r in b], marker="s", label="monolithic: interactive ITL p99")
    split = next(r for r in stall if r["condition"] == "1P + 1D")
    ax.axhline(split["chat_itl_max_ms"] / 1000, color="C3", ls="--", label="1P + 1D: interactive ITL max")
    ax.set_xscale("log", base=2); ax.set_xlabel("chunked-prefill budget (tokens)"); ax.set_ylabel("seconds")
    ax.set_title("The stall a long prompt causes, by budget (simulated, T4-calibrated)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "stall_vs_budget.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [r.label.replace(", ", "\n") for r in two]
    axes[0].bar(labels, [r.output_tokens_per_s for r in two]); axes[0].set_title("output tokens/s")
    axes[1].bar(labels, [r.by_kind["chat"]["itl"]["max"] * 1000 for r in two]); axes[1].set_title("interactive ITL max (ms)")
    axes[2].bar(labels, [r.by_kind["long"]["ttft"]["p50"] * 1000 for r in two]); axes[2].set_title("long-prompt TTFT p50 (ms)")
    for a in axes:
        a.tick_params(axis="x", labelsize=7); a.grid(alpha=.3, axis="y")
    fig.suptitle("Two GPUs: one monolithic, two monolithic, or one prefill + one decode (simulated)")
    fig.tight_layout(); fig.savefig(out / "two_gpus.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(ratio), figsize=(6 * len(ratio), 4), squeeze=False)
    for a, (name, res) in zip(axes[0], ratio.items()):
        x = [r.label for r in res]
        a.bar(x, [r.output_tokens_per_s for r in res], color="C0", label="output tok/s")
        a2 = a.twinx(); a2.plot(x, [r.by_kind["long"]["ttft"]["p50"] for r in res], color="C3", marker="o", label="long TTFT p50 (s)")
        a.set_title(name, fontsize=9); a.tick_params(axis="x", labelsize=8); a.grid(alpha=.3, axis="y")
        a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Four GPUs: pool ratio vs throughput and long-prompt TTFT (simulated)")
    fig.tight_layout(); fig.savefig(out / "ratio.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted({r["prompt_tokens"] for r in link}):
        rs = [r for r in link if r["prompt_tokens"] == n]
        ax.plot([r["link_gb_s"] for r in rs], [r["transfer_ms"] / r["prefill_ms"] for r in rs], marker="o", label=f"{n}-token prompt")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("link bandwidth (GB/s)"); ax.set_ylabel("transfer time / prefill time")
    ax.axhline(1, color="k", lw=.8, ls=":"); ax.set_title("KV transfer time relative to the prefill it follows (1.5B model, T4 prefill speed)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "link_sweep.png", dpi=130); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="splitserve simulator scenarios")
    p.add_argument("--out", type=Path, default=Path("results/sim"))
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chat-rps", type=float, default=6.0)
    p.add_argument("--long-rps", type=float, default=0.5)
    p.add_argument("--link-gb-s", type=float, default=3.0)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(link_gb_s=a.link_gb_s)
    stall = scenario_stall(cfg, a.duration, a.seed)
    two = scenario_two_gpus(cfg, a.duration, a.seed, a.chat_rps, a.long_rps)
    ratio = scenario_ratio(cfg, a.duration, a.seed)
    link = scenario_link(cfg)
    md = ["## Stall vs chunk budget (4 req/s x 64->128 interactive, 0.25 req/s x 3500->32)\n",
          comparison_table(stall, ["condition", "chat_itl_p50_ms", "chat_itl_p99_ms", "chat_itl_max_ms", "chat_ttft_p50_ms", "long_ttft_p50_ms"]),
          f"\n## Two GPUs ({a.chat_rps:g} req/s x 96->128 interactive, {a.long_rps:g} req/s x 1500->48 long)\n",
          comparison_table([r.row() for r in two], COLS)]
    for name, res in ratio.items():
        md += [f"\n## Four GPUs, {name}\n", comparison_table([r.row() for r in res], COLS)]
    md += ["\n## KV transfer vs prefill\n", comparison_table(link, ["prompt_tokens", "link_gb_s", "prefill_ms", "transfer_ms"])]
    (a.out / "summary.md").write_text("".join(md))
    summary = {"config": asdict(cfg), "args": vars(a) | {"out": str(a.out)}, "stall": stall, "two_gpus": [r.row() for r in two],
               "ratio": {k: [r.row() for r in v] for k, v in ratio.items()}, "link": link}
    with open(a.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    if not a.no_plots:
        _plots(a.out, stall, two, ratio, link)
    print("".join(md))


if __name__ == "__main__":
    main()
