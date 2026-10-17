# Concepts: splitting prefill from decode, and what a two-GPU toy can prove

`splitserve` runs vLLM as two instances, one that only prefills and one that only decodes, with the prompt's KV cache handed from the first to the second, and puts the same mixed load through that pair, through one monolithic server, and through two monolithic replicas. The question is not "is disaggregation faster" (on two T4s with a 1.5B model it is mostly not) but *what it changes and why*: which latency it protects, what it costs, and which knobs only exist once the two phases are separate. A small event-driven simulator, calibrated on the numbers measured in `servellm` and `chunkedprefill-lab`, carries the questions a two-GPU box cannot answer: pool ratios, bigger models, faster links.

## 1. Two phases with opposite resource profiles

A request to an LLM server has two phases that use the GPU in opposite ways.

**Prefill** processes the whole prompt in one pass: every prompt token's query attends to every earlier key, the MLPs run over all tokens at once, and the key/value vectors of every layer are written to the KV cache. It is a matrix-matrix workload, arithmetic-heavy, and it scales with prompt length: linearly in the MLPs, quadratically in attention. On a T4, `chunkedprefill-lab` measured a 3,500-token prompt at about 3.5 s for a 3B model (roughly 1,000 prompt tokens per second); `servellm` measured 256 tokens of a 1.5B model in 32 ms. The phase is **compute-bound**; it benefits from tensor cores and, above all, from not being interrupted.

**Decode** produces one token per forward pass. Each step reads the *entire* weight matrix once and multiplies it by one vector per active sequence; the arithmetic is trivial, the memory traffic is the whole model. The step time is set by bytes moved, not FLOPs: 2.98 GiB of weights over a 320 GB/s bus is about 10 ms, and `servellm` measured 14.7 ms per token for one stream and 36.8 ms with 64 streams in the batch. The phase is **memory-bound**; it benefits from batching many sequences into the same weight read, and it is sensitive to anything that makes a step longer.

A monolithic server runs both on the same GPU, in the same scheduler loop. vLLM's continuous batching builds each step from whatever is ready: the decode tokens of every running request *plus* a prefill chunk of any newly admitted one. That is the source of the interaction this project is about.

## 2. The interference, and the knob that only half fixes it

When a long prompt arrives at a monolithic server, its prefill is folded into the next step. Every decoding user's next token is delayed by the prefill's duration, because the step does not finish until all its work has run. `chunkedprefill-lab` saw exactly this: an interactive user's inter-token gap, normally 35 ms, became 3.5 s whenever a 3,500-token prompt was prefilled in a single iteration. Chunked prefill splits the prompt into budget-sized pieces (512, 1,024, 2,048 tokens...), so the stall is shorter but repeated; the p95 and the maximum move in opposite directions as the budget changes, and no budget removes the stall, it only resizes it. The long prompt, meanwhile, waits in prefill chunks interleaved with decode steps, and the decode steps that carry a chunk are slower for everyone.

The two phases also want different batch compositions. Decode wants the largest batch the KV cache allows; prefill wants a handful of prompts, each big enough to fill the tensor cores. A scheduler that mixes them is always compromising one against the other, and the chunk budget is the single dial it has.

## 3. Disaggregation: give each phase its own GPU

Disaggregated prefill/decode ("P/D disaggregation", DistServe in the research literature, the default architecture of the large hosted inference stacks in 2026) runs prefill instances and decode instances as separate processes on separate GPUs. A request is prefilled on a P instance; the KV cache it produced is transferred to a D instance, which generates the output tokens. Nothing on the D GPU ever runs a prefill, so decode steps are never lengthened by a prompt; nothing on the P GPU ever waits for a decode batch, so the prompt is processed as fast as the GPU can prefill.

What it buys:

- **Tail inter-token latency.** The stall in section 2 is gone by construction, not by tuning. vLLM's own documentation gives this as the second reason to use the feature.
- **Independent tuning.** The P pool can use a different tensor-parallel degree, a different batch budget, even different hardware from the D pool, because TTFT and ITL now come from different machines. Scaling the P pool changes TTFT; scaling the D pool changes ITL and concurrency.
- **Independent sizing.** A workload with long prompts and short answers (retrieval, summarisation) needs many prefill GPUs and few decode GPUs; a chat workload needs the opposite. A monolithic fleet has to be sized for both at once.

What it costs:

- **The transfer.** The KV cache of a prompt must move from P to D before decode can start. For this model that is 28,672 bytes per token (`kvcalc`): 43 MB for a 1,500-token prompt, 100 MB for 3,500 tokens. Over NVLink or RDMA that is a few milliseconds; over PCIe through host memory it is tens of milliseconds; through a shared filesystem it is hundreds. The transfer adds to TTFT and, if it is slow, it can be larger than the prefill it follows.
- **A second scheduler and a proxy.** Requests now make two hops, and something has to route them, carry the handshake metadata between P and D, and handle a failure on either side.
- **Capacity.** A pair of GPUs used as 1P + 1D has less total throughput than the same pair used as two monolithic replicas whenever one of the two phases is the bottleneck and the other GPU sits partly idle. vLLM's documentation is explicit: "disaggregated prefill does not improve throughput." It buys latency structure, not tokens per second.

## 4. How the KV cache gets across

vLLM 0.31 implements disaggregation through **KV connectors**, pluggable classes that hook into the scheduler (which decides *what* to send or receive) and the worker (which moves the bytes, layer by layer, during the forward pass). Three matter here:

- **NixlConnector** uses NVIDIA's NIXL library over UCX. The P instance finishes its prefill and holds the blocks under a lease; the D instance, given the P's engine id, block ids and side-channel address, *reads* them with an RDMA-style transfer directly into its own paged cache, asynchronously, and starts decoding as soon as the read completes. On one host it uses shared memory or CUDA IPC. This is the production path and the one the notebook tries first.
- **ExampleConnector** writes the KV tensors to a directory on the filesystem after prefill and reads them back on the decode side, matched by a hash of the token ids. It needs no extra library and no network, which makes it the fallback when NIXL cannot load; its cost is the disk round-trip, which is the slow end of the transfer spectrum and therefore a useful second data point rather than merely a fallback.
- **MultiConnector** chains connectors; **LMCache** and **Mooncake** add external KV stores. Out of scope here.

The handshake is visible in the HTTP API. The proxy sends the request to P with `max_tokens=1` and `kv_transfer_params={"do_remote_decode": true}`; P prefills, returns its first token and a `kv_transfer_params` block describing where the blocks live; the proxy forwards the original request to D with that block attached (and the prompt's token ids, so D skips re-tokenising); D pulls the blocks, generates, and streams back to the client. The client sees one ordinary streaming response. `splitserve` implements this proxy and timestamps each hop so the transfer can be measured from the outside.

## 5. What a 2 x T4 can and cannot show

Kaggle gives two T4s on one host, connected over PCIe with no NVLink, and a 1.5B model whose prefill is fast (a 1,500-token prompt in well under a second) relative to its decode. That constrains the experiment in ways worth stating before the numbers exist:

- **It can show the interference disappearing.** Interactive users' ITL maximum and p99 under a mixed load are the direct measurement, and the P/D pair should remove the stall that the monolithic server shows.
- **It can measure the transfer cost on a slow link.** The TTFT of the P/D pair minus the prefill time is the handoff; on PCIe without GPUDirect it will be a visible fraction of TTFT for this small model, and that is a real result about where disaggregation pays and where it does not.
- **It can show the throughput argument honestly.** Two monolithic replicas behind a round-robin proxy are the fair comparison for two GPUs, and the P/D pair should lose to them on total tokens per second.
- **It cannot show pool ratios** (1P:3D, 2P:1D), **bigger models** where prefill is minutes rather than milliseconds, or **fast links** where the transfer vanishes. The simulator in `splitserve.sim` covers those: it models a P pool, a D pool and a link with explicit per-token costs, every one of which is set from a measured number in this portfolio, and it is checked against the measured monolithic behaviour before it is asked to extrapolate.

## 6. What to look at

For each of the three configurations under the same mixed open-loop load (a short-prompt interactive stream and a long-prompt stream, unique prompts so prefix caching cannot hide the prefill):

| metric | why |
|---|---|
| interactive ITL p50 / p99 / max | the stall is in the max and p99; p50 should barely move |
| interactive TTFT | the P pool's queue, plus the transfer in the P/D case |
| long-prompt TTFT | prefill time plus, for P/D, the handoff |
| total output tokens per second | the throughput cost of splitting the pair |
| proxy-measured prefill time and handoff time | the transfer cost in isolation |

The simulator reports the same metrics for the same scenarios plus a P:D ratio sweep and a link-bandwidth sweep.

## Further reading

- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024).
- Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (ISCA 2024).
- vLLM documentation, *Disaggregated Prefilling* and *NixlConnector Usage Guide* (v0.31).
- Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024).
