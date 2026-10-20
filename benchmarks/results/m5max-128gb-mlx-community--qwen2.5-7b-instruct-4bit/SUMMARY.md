# Bench summary

| machine | model | track | scorer | dataset | field acc | majority baseline | case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | bundled | 0.7222 | 0.5139 | 0.5833 | 0.5833 | 0.7116 | — | — | — | 496.915 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | perturbed | 0.7153 | 0.5139 | 0.5729 | 0.5729 | 0.7067 | — | — | 0.0093 | 545.62 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | typesafe | 0.6767 | 0.8577 | 0.25 | 0.25 | 0.6502 | — | — | — | 1479.04 | 45 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | bundled | 0.7083 | 0.5139 | 0.5 | 0.5 | 0.6702 | 0.258 | 0.0417 | — | 244.19 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | perturbed | 0.7292 | 0.5139 | 0.5521 | 0.5521 | 0.7 | 0.2255 | 0.0479 | 0.0463 | 237.485 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | typesafe | 0.7283 | 0.8577 | 0.2222 | 0.2222 | 0.7205 | 0.1991 | 0.0298 | — | 586.425 | 44 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | bundled | 0.6528 | 0.5139 | 0.3333 | 0.3333 | 0.5873 | 0.202 | 0.3028 | — | 199.85 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | perturbed | 0.6146 | 0.5139 | 0.2812 | 0.2812 | 0.552 | 0.2223 | 0.3111 | 0.1806 | 242.92 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | typesafe | 0.6196 | 0.8577 | 0.1556 | 0.1556 | 0.6151 | 0.1354 | 0.1557 | — | 633.445 | 44 | 45 |
