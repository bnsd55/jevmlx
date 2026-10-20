# Bench summary

| machine | model | track | scorer | dataset | field acc | majority baseline | case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | bundled | 0.8194 | 0.5139 | 0.625 | 0.625 | 0.7834 | — | — | — | 433.86 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | perturbed | 0.816 | 0.5139 | 0.6354 | 0.6354 | 0.7864 | — | — | 0.0139 | 551.08 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | typesafe | 0.6932 | 0.8577 | 0.3864 | 0.3864 | 0.6718 | — | — | — | 1584.43 | 45 | 45 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | bundled | 0.7917 | 0.5139 | 0.5 | 0.5 | 0.7827 | 0.1841 | 0.0222 | — | 233.935 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | perturbed | 0.7847 | 0.5139 | 0.5104 | 0.5104 | 0.7646 | 0.187 | 0.034 | 0.037 | 256.015 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | typesafe | 0.75 | 0.8577 | 0.2444 | 0.2444 | 0.7402 | 0.2025 | 0.0426 | — | 646.875 | 44 | 45 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | bundled | 0.5833 | 0.5139 | 0.2917 | 0.2917 | 0.4949 | 0.3335 | 0.2806 | — | 256.95 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | perturbed | 0.5729 | 0.5139 | 0.2292 | 0.2292 | 0.4951 | 0.3321 | 0.2778 | 0.1343 | 252.06 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | typesafe | 0.4511 | 0.8577 | 0.2444 | 0.2444 | 0.5063 | 0.417 | 0.1022 | — | 691 | 44 | 45 |
