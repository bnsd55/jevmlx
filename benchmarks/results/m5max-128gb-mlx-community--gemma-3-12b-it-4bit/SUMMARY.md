# Bench summary

| machine | model | track | scorer | dataset | field acc | majority baseline | case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | bundled | 0.7917 | 0.5139 | 0.625 | 0.625 | 0.7676 | — | — | — | 930.165 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | perturbed | 0.7604 | 0.5139 | 0.6146 | 0.6146 | 0.7433 | — | — | 0.0972 | 1019.755 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | typesafe | 0.4192 | 0.8577 | 0.3636 | 0.3636 | 0.4062 | — | — | — | 2923.7 | 45 | 45 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | bundled | 0.7222 | 0.5139 | 0.5 | 0.5 | 0.7099 | 0.2413 | 0.0389 | — | 408.68 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | perturbed | 0.7257 | 0.5139 | 0.4896 | 0.4896 | 0.7157 | 0.2363 | 0.0382 | 0.0509 | 442.48 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | typesafe | 0.6332 | 0.8577 | 0.3111 | 0.3111 | 0.6342 | 0.2301 | 0.0692 | — | 1181.59 | 44 | 45 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | bundled | 0.5417 | 0.5139 | 0.1667 | 0.1667 | 0.5817 | 0.3294 | 0.25 | — | 458.105 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | perturbed | 0.5347 | 0.5139 | 0.1667 | 0.1667 | 0.5808 | 0.3385 | 0.2549 | 0.1898 | 447.48 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | typesafe | 0.4511 | 0.8577 | 0.1556 | 0.1556 | 0.484 | 0.3823 | 0.1025 | — | 1224.805 | 44 | 45 |
