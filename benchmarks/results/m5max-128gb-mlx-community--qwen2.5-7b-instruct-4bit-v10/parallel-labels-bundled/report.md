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
| timestamp_utc | 2026-09-23T12:08:15+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6944 [0.5787, 0.8056] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.3750 |
| case_exact_match | 0.3750 |
| brier | 0.6975 [0.5106, 0.8845] (case_cluster_bootstrap) |
| correctness_auroc | 0.6407 |
| ece_5bin_equal_mass | 0.2116 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0333 |
| any_flip_rate[category] | 0.0500 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.0667 |
| any_flip_rate[risk] | 0.1000 |
| balanced_accuracy[action] | 0.3935 |
| balanced_accuracy[category] | 0.9583 |
| balanced_accuracy[fraud] | 1.0000 |
| balanced_accuracy[needs_human] | 0.7143 |
| balanced_accuracy[priority] | 0.7275 |
| balanced_accuracy[risk] | 0.4333 |
| macro_f1[action] | 0.3602 |
| macro_f1[category] | 0.9580 |
| macro_f1[fraud] | 1.0000 |
| macro_f1[needs_human] | 0.6571 |
| macro_f1[priority] | 0.7083 |
| macro_f1[risk] | 0.3282 |
| mean_tvd[action] | 0.0531 |
| mean_tvd[category] | 0.0289 |
| mean_tvd[fraud] | 0.0000 |
| mean_tvd[needs_human] | 0.0000 |
| mean_tvd[priority] | 0.0786 |
| mean_tvd[risk] | 0.1049 |
| order_flip_rate[action] | 0.0333 |
| order_flip_rate[category] | 0.0500 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.0667 |
| order_flip_rate[risk] | 0.1000 |
| valid_accuracy | 0.6944 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 12 | 0.3333 [0.1381, 0.6094] (wilson) † | 0.4167 |
| risk | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.4167 |
| needs_human | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| priority | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.5833 |
| category | 12 | 1.0000 [0.7575, 1.0000] (wilson) | 0.3333 |
| fraud | 12 | 1.0000 [0.7575, 1.0000] (wilson) | 0.7500 |
