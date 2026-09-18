# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | naive_local | slots | bundled | 0.5278 | 0.25 | 0.5253 | — | — | — | 113.84 | 24 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | naive_local | slots | perturbed | 0.4956 | 0.2632 | 0.501 | — | — | — | 114.245 | 76 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | naive_local | slots | typesafe | 0.5151 | 0.1364 | 0.5017 | — | — | — | 1375.43 | 45 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | labels | bundled | 0.5185 | 0.0833 | 0.5502 | 0.1121 | 0.1972 | — | 49.96 | 24 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | labels | perturbed | 0.4664 | 0.0526 | 0.512 | 0.1683 | 0.2035 | — | 39.125 | 76 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | labels | typesafe | 0.3653 | 0.0909 | 0.4417 | 0.4438 | 0.0542 | — | 2304.45 | 45 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | slots | bundled | 0.4815 | 0 | 0.4469 | 0.1464 | 0.35 | — | 53.245 | 24 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | slots | perturbed | 0.4781 | 0 | 0.4511 | 0.1387 | 0.3678 | — | 52.52 | 76 |
| m5max-128gb | mlx-community--qwen2.5-1.5b-instruct-4bit | parallel | slots | typesafe | 0.2547 | 0.0455 | 0.3799 | 0.5003 | 0.1462 | — | 2298 | 45 |
