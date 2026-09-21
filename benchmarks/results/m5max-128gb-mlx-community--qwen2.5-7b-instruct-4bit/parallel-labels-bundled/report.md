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
| timestamp_utc | 2026-09-21T04:27:47+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7245 [0.5926, 0.8495] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.4167 |
| case_exact_match | 0.4167 |
| brier | 0.6869 [0.4892, 0.8949] (case_cluster_bootstrap) |
| correctness_auroc | 0.7115 |
| ece_5bin_equal_mass | 0.2315 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0667 |
| any_flip_rate[category] | 0.0000 |
| any_flip_rate[fraud] | 0.0333 |
| any_flip_rate[needs_human] | 0.0500 |
| any_flip_rate[priority] | 0.0833 |
| any_flip_rate[risk] | 0.0167 |
| balanced_accuracy[action] | 0.5389 |
| balanced_accuracy[category] | 1.0000 |
| balanced_accuracy[fraud] | 0.5556 |
| balanced_accuracy[needs_human] | 0.8071 |
| balanced_accuracy[priority] | 0.8294 |
| balanced_accuracy[risk] | 0.4042 |
| macro_f1[action] | 0.5476 |
| macro_f1[category] | 1.0000 |
| macro_f1[fraud] | 0.5355 |
| macro_f1[needs_human] | 0.7913 |
| macro_f1[priority] | 0.7346 |
| macro_f1[risk] | 0.4074 |
| mean_tvd[action] | 0.0617 |
| mean_tvd[category] | 0.0057 |
| mean_tvd[fraud] | 0.0266 |
| mean_tvd[needs_human] | 0.0327 |
| mean_tvd[priority] | 0.0790 |
| mean_tvd[risk] | 0.0554 |
| order_flip_rate[action] | 0.0667 |
| order_flip_rate[category] | 0.0000 |
| order_flip_rate[fraud] | 0.0333 |
| order_flip_rate[needs_human] | 0.0500 |
| order_flip_rate[priority] | 0.0833 |
| order_flip_rate[risk] | 0.0167 |
| valid_accuracy | 0.7245 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| risk | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| fraud | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.7500 |
| needs_human | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.5833 |
| priority | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.5833 |
| category | 12 | 1.0000 [0.7575, 1.0000] (wilson) | 0.3333 |
