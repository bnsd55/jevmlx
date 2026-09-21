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
| timestamp_utc | 2026-09-21T02:22:28+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6111 [0.5255, 0.6852] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.0000 |
| case_exact_match | 0.0000 |
| brier | 0.7613 [0.6812, 0.8429] (case_cluster_bootstrap) |
| correctness_auroc | 0.5930 |
| ece_5bin_equal_mass | 0.2387 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.4000 |
| any_flip_rate[category] | 0.2333 |
| any_flip_rate[fraud] | 0.0500 |
| any_flip_rate[needs_human] | 0.0333 |
| any_flip_rate[priority] | 0.6000 |
| any_flip_rate[risk] | 0.5000 |
| balanced_accuracy[action] | 0.4352 |
| balanced_accuracy[category] | 0.8021 |
| balanced_accuracy[fraud] | 0.5093 |
| balanced_accuracy[needs_human] | 0.7238 |
| balanced_accuracy[priority] | 0.5397 |
| balanced_accuracy[risk] | 0.4062 |
| macro_f1[action] | 0.4329 |
| macro_f1[category] | 0.8081 |
| macro_f1[fraud] | 0.4704 |
| macro_f1[needs_human] | 0.7188 |
| macro_f1[priority] | 0.4973 |
| macro_f1[risk] | 0.3634 |
| mean_tvd[action] | 0.3603 |
| mean_tvd[category] | 0.2597 |
| mean_tvd[fraud] | 0.0719 |
| mean_tvd[needs_human] | 0.0676 |
| mean_tvd[priority] | 0.5579 |
| mean_tvd[risk] | 0.4394 |
| order_flip_rate[action] | 0.4000 |
| order_flip_rate[category] | 0.2333 |
| order_flip_rate[fraud] | 0.0500 |
| order_flip_rate[needs_human] | 0.0333 |
| order_flip_rate[priority] | 0.6000 |
| order_flip_rate[risk] | 0.5000 |
| valid_accuracy | 0.6111 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| priority | 12 | 0.4167 [0.1933, 0.6805] (wilson) † | 0.5833 |
| risk | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| action | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.4167 |
| fraud | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.7500 |
| needs_human | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.5833 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
