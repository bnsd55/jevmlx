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
| timestamp_utc | 2026-09-22T04:25:43+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7257 [0.6632, 0.7847] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.4896 |
| case_exact_match | 0.4896 |
| brier | 0.6312 [0.5455, 0.7245] (case_cluster_bootstrap) |
| correctness_auroc | 0.6592 |
| ece_5bin_equal_mass | 0.2363 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0500 |
| any_flip_rate[category] | 0.0125 |
| any_flip_rate[fraud] | 0.0250 |
| any_flip_rate[needs_human] | 0.0125 |
| any_flip_rate[priority] | 0.0292 |
| any_flip_rate[risk] | 0.1000 |
| balanced_accuracy[action] | 0.6556 |
| balanced_accuracy[category] | 0.9375 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.6667 |
| balanced_accuracy[risk] | 0.7250 |
| macro_f1[action] | 0.6730 |
| macro_f1[category] | 0.9365 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.6356 |
| macro_f1[risk] | 0.7048 |
| mean_tvd[action] | 0.0372 |
| mean_tvd[category] | 0.0116 |
| mean_tvd[fraud] | 0.0230 |
| mean_tvd[needs_human] | 0.0242 |
| mean_tvd[priority] | 0.0321 |
| mean_tvd[risk] | 0.1068 |
| order_flip_rate[action] | 0.0500 |
| order_flip_rate[category] | 0.0125 |
| order_flip_rate[fraud] | 0.0250 |
| order_flip_rate[needs_human] | 0.0125 |
| order_flip_rate[priority] | 0.0292 |
| order_flip_rate[risk] | 0.1000 |
| perturbation_flip_rate | 0.0509 |
| valid_accuracy | 0.7257 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 48 | 0.5833 [0.4428, 0.7115] (wilson) | 0.5833 |
| priority | 48 | 0.6250 [0.4836, 0.7478] (wilson) | 0.5833 |
| action | 48 | 0.6875 [0.5467, 0.8005] (wilson) | 0.4167 |
| risk | 48 | 0.6875 [0.5467, 0.8005] (wilson) | 0.4167 |
| fraud | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.7500 |
| category | 48 | 0.9375 [0.8316, 0.9785] (wilson) | 0.3333 |
