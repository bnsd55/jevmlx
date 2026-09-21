# jevmlx eval report

## Environment

| key | value |
| --- | --- |
| chip | Apple M5 Max |
| git_sha | dbb1ff1f04a0bce4b20817e4c09eae2b5a3d2ad3 |
| jevmlx_version | 0.1.0 |
| machine_model | Mac17,7 |
| macos_version | 26.6.2 |
| mlx_lm_version | 0.31.3 |
| mlx_version | 0.32.2 |
| python_version | 3.12.14 |
| ram_gb | 128.0000 |
| timestamp_utc | 2026-09-21T06:32:04+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7222 [0.5694, 0.8611] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.5833 |
| case_exact_match | 0.5833 |
| balanced_accuracy[action] | 0.5111 |
| balanced_accuracy[category] | 1.0000 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.8571 |
| balanced_accuracy[priority] | 0.8095 |
| balanced_accuracy[risk] | 0.4250 |
| macro_f1[action] | 0.4722 |
| macro_f1[category] | 1.0000 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.8333 |
| macro_f1[priority] | 0.6889 |
| macro_f1[risk] | 0.3958 |
| valid_accuracy | 0.7222 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| risk | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| priority | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| fraud | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.7500 |
| needs_human | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.5833 |
| category | 12 | 1.0000 [0.7575, 1.0000] (wilson) | 0.3333 |
