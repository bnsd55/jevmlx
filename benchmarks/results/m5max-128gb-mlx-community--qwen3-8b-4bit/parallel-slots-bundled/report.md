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
| timestamp_utc | 2026-09-21T06:43:34+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5833 [0.4444, 0.7083] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.2917 |
| case_exact_match | 0.2917 |
| brier | 0.8772 [0.6666, 1.0913] (case_cluster_bootstrap) |
| correctness_auroc | 0.7242 |
| ece_5bin_equal_mass | 0.3335 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.3833 |
| any_flip_rate[category] | 0.3833 |
| any_flip_rate[fraud] | 0.0167 |
| any_flip_rate[needs_human] | 0.0167 |
| any_flip_rate[priority] | 0.3667 |
| any_flip_rate[risk] | 0.5167 |
| balanced_accuracy[action] | 0.4444 |
| balanced_accuracy[category] | 0.3542 |
| balanced_accuracy[fraud] | 0.8333 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.4444 |
| balanced_accuracy[risk] | 0.2500 |
| macro_f1[action] | 0.3750 |
| macro_f1[category] | 0.2611 |
| macro_f1[fraud] | 0.8737 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.4259 |
| macro_f1[risk] | 0.1471 |
| mean_tvd[action] | 0.3658 |
| mean_tvd[category] | 0.3706 |
| mean_tvd[fraud] | 0.0131 |
| mean_tvd[needs_human] | 0.0190 |
| mean_tvd[priority] | 0.3550 |
| mean_tvd[risk] | 0.4382 |
| order_flip_rate[action] | 0.3833 |
| order_flip_rate[category] | 0.3833 |
| order_flip_rate[fraud] | 0.0167 |
| order_flip_rate[needs_human] | 0.0167 |
| order_flip_rate[priority] | 0.3667 |
| order_flip_rate[risk] | 0.5167 |
| valid_accuracy | 0.5833 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| category | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.3333 |
| risk | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.4167 |
| action | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| needs_human | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| priority | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| fraud | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.7500 |
