# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | bundled | 0.7222 | 0.5139 | 0.5833 | 0.5833 | 0.7116 | — | — | — | 496.915 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | perturbed | 0.7153 | 0.5139 | 0.5729 | 0.5729 | 0.7067 | — | — | 0.0093 | 545.62 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | typesafe | 0.6767 | 0.8577 | 0.25 | 0.25 | 0.6502 | — | — | — | 1479.04 | 45 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | bundled | 0.7245 | 0.5139 | 0.4167 | 0.4167 | 0.6892 | 0.2315 | 0.0417 | — | 244.19 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | perturbed | 0.7303 | 0.5139 | 0.4271 | 0.4271 | 0.7 | 0.2198 | 0.0479 | 0.0463 | 237.485 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | typesafe | 0.8213 | 0.8601 | 0.2 | 0.2045 | 0.7124 | 0.1102 | 0.0298 | — | 586.425 | 44 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | bundled | 0.6111 | 0.5139 | 0 | 0 | 0.5694 | 0.2387 | 0.3028 | — | 199.85 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | perturbed | 0.5984 | 0.5139 | 0 | 0 | 0.5585 | 0.2443 | 0.3111 | 0.1806 | 242.92 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | typesafe | 0.6322 | 0.8601 | 0.1111 | 0.1136 | 0.6153 | 0.12 | 0.1557 | — | 633.445 | 44 | 45 |
