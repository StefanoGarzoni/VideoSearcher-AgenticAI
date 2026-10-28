# Final model comparison — all 14 metrics

VQA held constant (`Phi-4-reasoning-vision`). Each model from its latest full 66-scenario run. `phi4-mini*` = broken-parsing baseline (fix in registry, but too weak to complete chains). `–` = metric not recorded for that run.

| Metric | gemma-12b | gemma-E2B | gpt-oss-120b | qwen3.6-27B | qwen3.6-35B-A3B |
|---|---|---|---|---|---|
| _Country_ | USA | USA | USA | China | China |
| _Size · Arch_ | 12 d | 5 d | 120 MoE | 27 d | 35 MoE |
| 1 · Overall % | 93.9 | 90.9 | 87.9 | 87.9 | 95.5 |
| 2 · Handoff % | 97.0 | 93.9 | 87.9 | 89.4 | 95.5 |
| 3 · Vision % | 100.0 | 88.9 | 88.9 | 100.0 | 100.0 |
| 4 · Result % | 92.6 | 92.6 | 96.3 | 96.3 | 100.0 |
| 5 · Tool-call eff. (extra/task) | 0.094 | 0.145 | 0.155 | 0.186 | 0.175 |
| 6 · Error-handling % | 100.0 | 100.0 | 100.0 | 73.3 | 93.3 |
| 7a · Robust vague % | 90.9 | 86.4 | 81.8 | 86.4 | 90.9 |
| 7b · Robust clear % | 90.9 | 90.9 | 90.9 | 86.4 | 95.5 |
| 7c · Robust algo % | 100.0 | 95.5 | 90.9 | 90.9 | 100.0 |
| 8 · Avg latency (s) | 92.1 | 37.1 | 30.2 | 77.1 | 32.0 |
| 9 · TTFT (ms) | 432.9 | 90.9 | 204.3 | 426.3 | 134.8 |
| 10 · Throughput (tok/s) | 16.8 | 83.2 | 53.5 | 11.6 | 74.1 |
| 11 · Tokens/task | 14991 | 14991 | 12103 | 15962 | 16508 |
| 12 · Energy/run (kWh) | 0.092779 | 0.035274 | 0.028289 | 0.086534 | 0.029132 |
| 13 · Passes/kWh | 668.3 | 1701.0 | 2050.3 | 670.3 | 2162.6 |
| 14 · Passes/min | 0.61 | 1.47 | 1.74 | 0.68 | 1.79 |
