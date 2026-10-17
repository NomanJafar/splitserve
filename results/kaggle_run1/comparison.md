| condition | requests | errors | out_tok_s | chat_ttft_p50_ms | chat_itl_p50_ms | chat_itl_p99_ms | chat_itl_max_ms | long_ttft_p50_ms | long_ttft_p95_ms | prefill_p50_ms | handoff_p50_ms | startup_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| monolithic 1xT4 | 1158 | 0 | 775 | 148 | 32.9 | 584 | 749 | 655 | 1153 | – | 150 | 163 |
| monolithic x2 (round-robin) | 1158 | 0 | 777 | 84.1 | 19.8 | 55.0 | 613 | 623 | 804 | – | 82.9 | 103 |
| 1P + 1D, NixlConnector | 1158 | 0 | 774 | 303 | 26.7 | 54.8 | 134 | 1072 | 1549 | 100 | 212 | 157 |
| 1P + 1D, ExampleConnector (shared dir) | 1158 | 984 | 114 | 222 | 27.5 | 91.6 | 171 | 897 | 1141 | 125 | 138 | 124 |
