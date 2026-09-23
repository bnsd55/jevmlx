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
| timestamp_utc | 2026-09-23T14:52:48+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6997 [0.6400, 0.7546] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.3438 |
| case_exact_match | 0.3438 |
| brier | 0.6904 [0.5980, 0.7888] (case_cluster_bootstrap) |
| correctness_auroc | 0.6242 |
| ece_5bin_equal_mass | 0.2058 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0542 |
| any_flip_rate[category] | 0.0500 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.0958 |
| any_flip_rate[risk] | 0.0917 |
| balanced_accuracy[action] | 0.3979 |
| balanced_accuracy[category] | 0.9601 |
| balanced_accuracy[fraud] | 1.0000 |
| balanced_accuracy[needs_human] | 0.6964 |
| balanced_accuracy[priority] | 0.7834 |
| balanced_accuracy[risk] | 0.4214 |
| macro_f1[action] | 0.3648 |
| macro_f1[category] | 0.9595 |
| macro_f1[fraud] | 1.0000 |
| macro_f1[needs_human] | 0.6329 |
| macro_f1[priority] | 0.7623 |
| macro_f1[risk] | 0.3314 |
| mean_tvd[action] | 0.0642 |
| mean_tvd[category] | 0.0402 |
| mean_tvd[fraud] | 0.0000 |
| mean_tvd[needs_human] | 0.0000 |
| mean_tvd[priority] | 0.0914 |
| mean_tvd[risk] | 0.0916 |
| order_flip_rate[action] | 0.0542 |
| order_flip_rate[category] | 0.0500 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.0958 |
| order_flip_rate[risk] | 0.0917 |
| perturbation_flip_rate | 0.0556 |
| valid_accuracy | 0.6997 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 48 | 0.3333 [0.2168, 0.4746] (wilson) † | 0.4167 |
| risk | 48 | 0.4167 [0.2885, 0.5572] (wilson) | 0.4167 |
| needs_human | 48 | 0.6458 [0.5044, 0.7657] (wilson) | 0.5833 |
| priority | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.5833 |
| category | 48 | 1.0000 [0.9259, 1.0000] (wilson) | 0.3333 |
| fraud | 48 | 1.0000 [0.9259, 1.0000] (wilson) | 0.7500 |
