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
| timestamp_utc | 2026-09-21T17:55:19+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5417 [0.4306, 0.6528] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.1667 |
| case_exact_match | 0.1667 |
| brier | 0.8580 [0.7307, 0.9813] (case_cluster_bootstrap) |
| correctness_auroc | 0.6647 |
| ece_5bin_equal_mass | 0.3294 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.4167 |
| any_flip_rate[category] | 0.1167 |
| any_flip_rate[fraud] | 0.0167 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.4000 |
| any_flip_rate[risk] | 0.5500 |
| balanced_accuracy[action] | 0.5000 |
| balanced_accuracy[category] | 0.7500 |
| balanced_accuracy[fraud] | 0.8889 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.3333 |
| balanced_accuracy[risk] | 0.3750 |
| macro_f1[action] | 0.3481 |
| macro_f1[category] | 0.6643 |
| macro_f1[fraud] | 0.8125 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.1429 |
| macro_f1[risk] | 0.3068 |
| mean_tvd[action] | 0.3736 |
| mean_tvd[category] | 0.1257 |
| mean_tvd[fraud] | 0.0496 |
| mean_tvd[needs_human] | 0.0084 |
| mean_tvd[priority] | 0.4079 |
| mean_tvd[risk] | 0.4320 |
| order_flip_rate[action] | 0.4167 |
| order_flip_rate[category] | 0.1167 |
| order_flip_rate[fraud] | 0.0167 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.4000 |
| order_flip_rate[risk] | 0.5500 |
| valid_accuracy | 0.5417 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| priority | 12 | 0.2500 [0.0889, 0.5323] (wilson) † | 0.5833 |
| action | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.4167 |
| risk | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.4167 |
| needs_human | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| category | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.3333 |
| fraud | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.7500 |
