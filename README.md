# splitserve

Disaggregated prefill/decode LLM serving: separate GPU pools with KV transfer, measuring the throughput and latency deltas against a monolithic server.

Project 10 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

- **A P/D proxy** (`splitserve-proxy`): one OpenAI-compatible endpoint in front of vLLM prefill and decode instances. It implements the NixlConnector handshake (prefill with `max_tokens=1` and `kv_transfer_params.do_remote_decode`, forward the returned block ids and side-channel address to the decode instance), the ExampleConnector variant (KV via a shared directory, matched by token hash), and a round-robin mode for monolithic replicas so the three can be compared through the same door. Every request is timed at each hop: the prefill round-trip and the handoff (decode request to first byte, which includes the KV transfer).
- **An event-driven simulator** (`splitserve-sim`): monolithic instances with chunked prefill, prefill-only and decode-only instances, and a link with a per-byte cost. Every parameter is a number measured elsewhere in this portfolio: decode step time against batch size and short-prompt prefill from [servellm](https://github.com/NomanJafar/servellm), long-prompt prefill from [chunkedprefill-lab](https://github.com/NomanJafar/chunkedprefill-lab), KV bytes per token from [kvcalc](https://github.com/NomanJafar/kvcalc). It answers what two T4s cannot: pool ratios, link speeds, four-GPU layouts.
- **A 2 x T4 experiment** (`notebooks/kaggle_splitserve.ipynb`): the same mixed open-loop load through one monolithic server, two monolithic replicas, 1P + 1D over NIXL, and 1P + 1D over a shared directory.

[CONCEPTS.md](CONCEPTS.md) is the primer: why the two phases fight over one GPU, what splitting them buys and costs, how the KV cache crosses, and what a two-GPU toy can and cannot show.

## Why it matters

This is the architecture behind every frontier serving stack in 2026. Knowing what it fixes (tail inter-token latency, independent sizing of the two phases) and what it does not (throughput) is the difference between adopting it and cargo-culting it.

## Status

Proxy, simulator and tests are done; the simulated results are below. The 2 x T4 run is queued on Kaggle and its results will replace the "pending" section when it finishes.

## Simulated results

`splitserve-sim --out results/sim` (120 s traces, Poisson arrivals, seed 0; tables in [results/sim/summary.md](results/sim/summary.md)).

### 1. The stall, and the knob that only resizes it

Interactive stream (4 req/s, 64-token prompts, 128 out) plus 0.25 req/s of 3,500-token prompts on one monolithic GPU, chunk budget swept; then the same load on 1P + 1D.

| condition | interactive ITL p50 | p99 | **max** | interactive TTFT p50 | long TTFT p50 |
|---|---|---|---|---|---|
| budget=512 | 26.6 ms | 489 ms | **0.56 s** | 369 ms | 2.51 s |
| budget=1024 | 25.8 ms | 816 ms | **1.03 s** | 135 ms | 2.37 s |
| budget=2048 | 25.5 ms | 1211 ms | **1.70 s** | 90 ms | 2.31 s |
| budget=4096 | 25.5 ms | 144 ms | **2.46 s** | 90 ms | 2.24 s |
| budget=8192 | 25.5 ms | 109 ms | **4.50 s** | 89 ms | 2.23 s |
| interactive only | 18.1 ms | 31 ms | 39 ms | 40 ms | – |
| **1P + 1D** | 18.5 ms | 27 ms | **27 ms** | 60 ms | 2.24 s |

![stall](results/sim/stall_vs_budget.png)

The model reproduces the shape chunkedprefill-lab measured on a real T4: the maximum grows with the budget while p99 collapses once a long prompt fits in one chunk (the lab saw p99 152 ms against a 3.48 s maximum at budget 4096; the model gives 144 ms against 2.46 s for the smaller model). The pair removes the stall by construction: interactive ITL max 27 ms, within 10 ms of the "interactive only" line.

### 2. Two GPUs: replicate or split

6 req/s interactive (96 → 128) plus 0.5 req/s long (1,500 → 48), the load the notebook uses.

| condition | GPUs | out tok/s | interactive TTFT p50 | interactive ITL p50 / p99 / max | long TTFT p50 |
|---|---|---|---|---|---|
| monolithic | 1 | 773 | 67 ms | 27.6 / 527 / 661 ms | 542 ms |
| monolithic x2 | 2 | 782 | 43 ms | 17.5 / 36 / 595 ms | 521 ms |
| 1P + 1D, 3 GB/s link | 2 | 781 | 55 ms | 20.6 / 24 / 25 ms | 542 ms |
| 1P + 1D, 25 GB/s link | 2 | 781 | 54 ms | 20.6 / 24 / 25 ms | 529 ms |
| 1P + 1D, 200 GB/s link | 2 | 781 | 54 ms | 20.6 / 24 / 25 ms | 527 ms |

![two gpus](results/sim/two_gpus.png)

At this load none of the layouts is throughput-bound, so tokens per second are identical and the differences are all latency structure. Two replicas halve the median ITL but each replica still stalls (max 595 ms); the pair has a slightly higher median (one GPU decodes the whole batch) and no stall at all. The link barely matters for a 1.5B model: a 1,500-token prompt's KV is 43 MB, 16 ms at 3 GB/s against a 490 ms prefill.

### 3. Four GPUs: the ratio has to match the load

| load | layout | out tok/s | interactive ITL p99 / max | long TTFT p50 |
|---|---|---|---|---|
| chat-heavy (12 req/s x 96→128, 0.5 req/s x 1500→48) | monolithic x4 | 1,516 | 33 / 533 ms | 517 ms |
| | 1P + 3D | 1,515 | 20 / 21 ms | 539 ms |
| | 2P + 2D | 1,517 | 22 / 23 ms | 538 ms |
| | 3P + 1D | 1,511 | 39 / 41 ms | 558 ms |
| prefill-heavy (2 req/s x 96→128, 1 req/s x 3000→32) | monolithic x4 | 268 | 796 / 1,427 ms | 1.70 s |
| | 1P + 3D | 163 | 16 / 16 ms | **49 s** (prefill pool saturated) |
| | 2P + 2D | 271 | 17 / 17 ms | 4.47 s |
| | 3P + 1D | 271 | 19 / 20 ms | 1.72 s |

![ratio](results/sim/ratio.png)

A 3,000-token prompt costs about 1.65 s of prefill on this GPU, so one request per second needs more than one prefill GPU; 1P + 3D falls over while 3P + 1D matches four monolithic replicas on TTFT and beats them on tail ITL by 70x. Disaggregation turns "how many GPUs" into "how many of each", and the second question has a wrong answer the first does not.

### 4. When the transfer matters

![link](results/sim/link_sweep.png)

For this model on this GPU the KV transfer is never more than a quarter of the prefill it follows, even at 1 GB/s, because the T4 prefills slowly (0.1 ms per token and up) while 28 KB per token crosses even a slow link in 0.03 ms. The ratio flips on hardware that prefills 50x faster with a model whose KV per token is larger: there the link is the design constraint, which is why production stacks run it over NVLink or RDMA.

## Measured results (2 x T4)

Pending: the Kaggle run is queued. The notebook records, per condition, tokenbench's TTFT and ITL percentiles for both streams, total output tokens per second, and the proxy's prefill and handoff times; this section will carry that table and `results/kaggle_run1/`.

## Layout

| path | what |
|---|---|
| `CONCEPTS.md` | the primer |
| `src/splitserve/proxy.py` | the P/D proxy (`splitserve-proxy --mode nixl\|shared\|replicas --prefill URL --decode URL`), `/records`, `/metrics` |
| `src/splitserve/sim.py` | `SimConfig`, `monolithic()`, `disaggregated()`, `prefill_ms()`, `transfer_ms()` |
| `src/splitserve/trace.py` | Poisson mixed traces |
| `src/splitserve/analysis.py` | tokenbench JSON and proxy records to comparison rows and Markdown |
| `src/splitserve/cli.py` | `splitserve-sim`: the four scenarios, tables and plots |
| `tests/` | 18 tests: the handshake against a fake vLLM pair that speaks it, the simulator's calibration and claims, the analysis helpers |
| `notebooks/kaggle_splitserve.ipynb` | the 2 x T4 experiment |
| `results/sim/` | simulated tables and plots |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
splitserve-sim --out results/sim                 # a few seconds; tables and plots
# against real vLLM instances (same host, two GPUs):
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
splitserve-proxy --mode nixl --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 --port 8080
tokenbench sweep --base-url http://127.0.0.1:8080 --model Qwen/Qwen2.5-1.5B-Instruct --mode open --levels 6 --duration 120 --prompts unique --out results/chat.json
```

## What the simulator leaves out

No KV-cache capacity limit or preemption (servellm never got near the limit with this model), no prefix caching, no network variance, a fixed per-step cost model with no CUDA-graph or kernel-launch effects, and round-robin placement only. It is a model of the scheduler's arithmetic, checked against measured behaviour in one configuration, not a performance predictor for another GPU.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels, PCIe without GPUDirect between the two cards), the README says so rather than papering over it.

## License

MIT
