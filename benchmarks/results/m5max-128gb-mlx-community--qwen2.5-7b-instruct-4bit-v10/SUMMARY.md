# Bench summary

| machine | model | track | scorer | dataset | field acc | majority baseline | case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | bundled | 0.7222 | 0.5139 | 0.5833 | 0.5833 | 0.7116 | — | — | — | 455.67 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | perturbed | 0.7153 | 0.5139 | 0.5729 | 0.5729 | 0.7067 | — | — | 0.0093 | 468.635 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | naive_local | slots | typesafe | 0.7041 | 0.8577 | 0.2955 | 0.2955 | 0.69 | — | — | — | 1357.19 | 45 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | bundled | 0.6944 | 0.5139 | 0.375 | 0.375 | 0.7045 | 0.2116 | 0.0417 | — | 254.85 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | perturbed | 0.6997 | 0.5139 | 0.3438 | 0.3438 | 0.7099 | 0.2058 | 0.0486 | 0.0556 | 294.595 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | labels | typesafe | 0.716 | 0.8601 | 0.1778 | 0.1818 | 0.6667 | 0.1724 | 0.0196 | — | 801.465 | 44 | 45 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | bundled | 0.706 | 0.5139 | 0.1667 | 0.1667 | 0.7213 | 0.2104 | 0.0889 | — | 213.795 | 24 | 24 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | perturbed | 0.7095 | 0.5139 | 0.2083 | 0.2083 | 0.7189 | 0.2137 | 0.0882 | 0.0972 | 213.985 | 96 | 96 |
| m5max-128gb | mlx-community--qwen2.5-7b-instruct-4bit | parallel | slots | typesafe | 0.7317 | 0.8601 | 0.1333 | 0.1364 | 0.6944 | 0.1682 | 0.0408 | — | 658.585 | 44 | 45 |
