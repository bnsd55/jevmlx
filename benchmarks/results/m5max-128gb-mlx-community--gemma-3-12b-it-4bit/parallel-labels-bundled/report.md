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
| timestamp_utc | 2026-09-21T23:18:26+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7222 [0.5972, 0.8472] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.5000 |
| case_exact_match | 0.5000 |
| brier | 0.6350 [0.4632, 0.8116] (case_cluster_bootstrap) |
| correctness_auroc | 0.6620 |
| ece_5bin_equal_mass | 0.2413 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0333 |
| any_flip_rate[category] | 0.0000 |
| any_flip_rate[fraud] | 0.0500 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.0333 |
| any_flip_rate[risk] | 0.1167 |
| balanced_accuracy[action] | 0.6278 |
| balanced_accuracy[category] | 0.9167 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.6429 |
| balanced_accuracy[risk] | 0.7625 |
| macro_f1[action] | 0.6389 |
| macro_f1[category] | 0.9143 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.6040 |
| macro_f1[risk] | 0.7208 |
| mean_tvd[action] | 0.0220 |
| mean_tvd[category] | 0.0001 |
| mean_tvd[fraud] | 0.0210 |
| mean_tvd[needs_human] | 0.0202 |
| mean_tvd[priority] | 0.0372 |
| mean_tvd[risk] | 0.1276 |
| order_flip_rate[action] | 0.0333 |
| order_flip_rate[category] | 0.0000 |
| order_flip_rate[fraud] | 0.0500 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.0333 |
| order_flip_rate[risk] | 0.1167 |
| valid_accuracy | 0.7222 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| priority | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| action | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.4167 |
| risk | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.4167 |
| fraud | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.7500 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
