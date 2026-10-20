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
| timestamp_utc | 2026-09-21T15:02:29+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7500 [0.6806, 0.8333] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.3333 |
| case_exact_match | 0.3333 |
| brier | 0.4646 [0.3667, 0.5668] (case_cluster_bootstrap) |
| correctness_auroc | 0.6836 |
| ece_5bin_equal_mass | 0.0760 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.1667 |
| any_flip_rate[category] | 0.0333 |
| any_flip_rate[fraud] | 0.0667 |
| any_flip_rate[needs_human] | 0.0333 |
| any_flip_rate[priority] | 0.0167 |
| any_flip_rate[risk] | 0.0000 |
| balanced_accuracy[action] | 0.6500 |
| balanced_accuracy[category] | 0.9167 |
| balanced_accuracy[fraud] | 1.0000 |
| balanced_accuracy[needs_human] | 0.5714 |
| balanced_accuracy[priority] | 0.7222 |
| balanced_accuracy[risk] | 0.5000 |
| macro_f1[action] | 0.6104 |
| macro_f1[category] | 0.9143 |
| macro_f1[fraud] | 1.0000 |
| macro_f1[needs_human] | 0.4375 |
| macro_f1[priority] | 0.7806 |
| macro_f1[risk] | 0.5119 |
| mean_tvd[action] | 0.1300 |
| mean_tvd[category] | 0.0491 |
| mean_tvd[fraud] | 0.0140 |
| mean_tvd[needs_human] | 0.0415 |
| mean_tvd[priority] | 0.0707 |
| mean_tvd[risk] | 0.0582 |
| order_flip_rate[action] | 0.1667 |
| order_flip_rate[category] | 0.0333 |
| order_flip_rate[fraud] | 0.0667 |
| order_flip_rate[needs_human] | 0.0333 |
| order_flip_rate[priority] | 0.0167 |
| order_flip_rate[risk] | 0.0000 |
| valid_accuracy | 0.7500 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 12 | 0.5000 [0.2538, 0.7462] (wilson) † | 0.5833 |
| action | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.4167 |
| risk | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.4167 |
| priority | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.5833 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
| fraud | 12 | 1.0000 [0.7575, 1.0000] (wilson) | 0.7500 |
