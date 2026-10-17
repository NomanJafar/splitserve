| condition | requests | errors | out_tok_s | chat_ttft_p50_ms | chat_itl_p50_ms | chat_itl_p99_ms | chat_itl_max_ms | long_ttft_p50_ms | long_ttft_p95_ms | prefill_p50_ms | handoff_p50_ms | startup_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| monolithic 1xT4 | 1158 | 0 | 776 | 136 | 30.7 | 572 | 1130 | 645 | 1049 | – | 136 | 169 |
| monolithic x2 (round-robin) | 1158 | 0 | 779 | 74.5 | 18.1 | 47.5 | 591 | 608 | 645 | – | 72.7 | 110 |
| 1P + 1D, NixlConnector | 1158 | 0 | 775 | 305 | 23.4 | 50.3 | 129 | 1067 | 1662 | 107 | 208 | 177 |
| 1P + 1D, ExampleConnector (shared dir) | 1158 | 878 | 187 | 235 | 24.1 | 82.5 | 201 | 963 | 1493 | 137 | 138 | 129 |
