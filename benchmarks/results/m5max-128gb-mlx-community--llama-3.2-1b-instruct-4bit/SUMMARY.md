# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | naive_local | slots | bundled | 0 | 0 | 0 | — | — | — | 1971.125 | 24 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | naive_local | slots | perturbed | 0.0044 | 0 | 0.0038 | — | — | — | 1985.065 | 76 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | naive_local | slots | typesafe | 0.0493 | 0 | 0.0313 | — | — | — | 3560.09 | 45 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | labels | bundled | 0.3889 | 0 | 0.4036 | 0.3093 | 0.0861 | — | 40.415 | 24 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | labels | perturbed | 0.3808 | 0 | 0.4126 | 0.3011 | 0.0902 | — | 40.285 | 76 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | labels | typesafe | 0.2889 | 0 | 0.3179 | 0.5936 | 0.0643 | — | 1476.17 | 45 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | slots | bundled | 0.3796 | 0 | 0.3673 | 0.3078 | 0.4083 | — | 40.805 | 24 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | slots | perturbed | 0.3699 | 0 | 0.3401 | 0.3282 | 0.3936 | — | 41.09 | 76 |
| m5max-128gb | mlx-community--llama-3.2-1b-instruct-4bit | parallel | slots | typesafe | 0.328 | 0.0227 | 0.3537 | 0.5279 | 0.1295 | — | 1535.53 | 45 |
