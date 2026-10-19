# Bench summary

| machine | model | track | scorer | dataset | field acc | majority baseline | case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | bundled | 0.6667 | 0.5139 | 0.4167 | 0.4167 | 0.6373 | — | — | — | 458.55 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | perturbed | 0.6146 | 0.5139 | 0.3854 | 0.3854 | 0.582 | — | — | 0.1574 | 575.755 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | typesafe | 0.2548 | 0.8577 | 0.1591 | 0.1591 | 0.2735 | — | — | — | 1491.08 | 45 | 45 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | bundled | 0.75 | 0.5139 | 0.3333 | 0.3333 | 0.7267 | 0.076 | 0.0528 | — | 248.365 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | perturbed | 0.7153 | 0.5139 | 0.3333 | 0.3333 | 0.6905 | 0.0592 | 0.0674 | 0.088 | 263.895 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | typesafe | 0.6495 | 0.8577 | 0.1333 | 0.1333 | 0.6291 | 0.1008 | 0.0556 | — | 707.1 | 44 | 45 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | bundled | 0.4306 | 0.5139 | 0.0833 | 0.0833 | 0.377 | 0.2717 | 0.3361 | — | 243.155 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | perturbed | 0.4653 | 0.5139 | 0.1042 | 0.1042 | 0.3918 | 0.2128 | 0.3097 | 0.2407 | 262.435 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | typesafe | 0.5652 | 0.8577 | 0.0667 | 0.0667 | 0.5527 | 0.0634 | 0.1093 | — | 705.405 | 44 | 45 |
