# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | naive_local | slots | bundled | 0.0556 | 0 | 0.0694 | — | — | — | 99.545 | 24 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | naive_local | slots | perturbed | 0.0263 | 0 | 0.0266 | — | — | — | 106.025 | 76 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | naive_local | slots | typesafe | 0.011 | 0.0227 | 0.0315 | — | — | — | 2741.16 | 45 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | labels | bundled | 0.3171 | 0 | 0.3559 | 0.3299 | 0.0472 | — | 19.345 | 24 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | labels | perturbed | 0.3151 | 0 | 0.3605 | 0.3395 | 0.0467 | — | 21.755 | 76 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | labels | typesafe | 0.318 | 0 | 0.3359 | 0.5536 | 0.068 | — | 805.03 | 45 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | slots | bundled | 0.3079 | 0 | 0.3446 | 0.4073 | 0.3944 | — | 19.38 | 24 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | slots | perturbed | 0.3143 | 0 | 0.3585 | 0.4032 | 0.3908 | — | 19.46 | 76 |
| m5max-128gb | mlx-community--qwen2.5-0.5b-instruct-4bit | parallel | slots | typesafe | 0.3162 | 0 | 0.3524 | 0.4515 | 0.1299 | — | 658.45 | 45 |
