# jevmlx eval report

## Environment

| key | value |
| --- | --- |
| chip | Apple M5 Max |
| git_sha | e0675624c30be62b5bffcbffc960e48d79b90683 |
| jevmlx_version | 0.1.0 |
| machine_model | Mac17,7 |
| macos_version | 26.6.2 |
| mlx_lm_version | 0.31.3 |
| mlx_version | 0.32.2 |
| python_version | 3.12.14 |
| ram_gb | 128.0000 |
| timestamp_utc | 2026-09-23T14:58:23+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7153 [0.6389, 0.7812] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.5729 |
| case_exact_match | 0.5729 |
| balanced_accuracy[action] | 0.4944 |
| balanced_accuracy[category] | 1.0000 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.8571 |
| balanced_accuracy[priority] | 0.8095 |
| balanced_accuracy[risk] | 0.4125 |
| macro_f1[action] | 0.4489 |
| macro_f1[category] | 1.0000 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.8333 |
| macro_f1[priority] | 0.6889 |
| macro_f1[risk] | 0.3783 |
| perturbation_flip_rate | 0.0093 |
| valid_accuracy | 0.7153 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 48 | 0.4792 [0.3447, 0.6167] (wilson) | 0.4167 |
| risk | 48 | 0.4792 [0.3447, 0.6167] (wilson) | 0.4167 |
| priority | 48 | 0.6667 [0.5254, 0.7832] (wilson) | 0.5833 |
| fraud | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.7500 |
| needs_human | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.5833 |
| category | 48 | 1.0000 [0.9259, 1.0000] (wilson) | 0.3333 |
