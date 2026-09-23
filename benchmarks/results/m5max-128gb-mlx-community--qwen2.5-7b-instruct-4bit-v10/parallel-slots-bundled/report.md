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
| timestamp_utc | 2026-09-23T05:47:48+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7060 [0.6019, 0.8171] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.1667 |
| case_exact_match | 0.1667 |
| brier | 0.7491 [0.5808, 0.9249] (case_cluster_bootstrap) |
| correctness_auroc | 0.7890 |
| ece_5bin_equal_mass | 0.2104 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0833 |
| any_flip_rate[category] | 0.0167 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.1667 |
| any_flip_rate[risk] | 0.2667 |
| balanced_accuracy[action] | 0.6046 |
| balanced_accuracy[category] | 0.9028 |
| balanced_accuracy[fraud] | 0.8333 |
| balanced_accuracy[needs_human] | 0.8571 |
| balanced_accuracy[priority] | 0.7156 |
| balanced_accuracy[risk] | 0.4146 |
| macro_f1[action] | 0.5312 |
| macro_f1[category] | 0.8990 |
| macro_f1[fraud] | 0.8737 |
| macro_f1[needs_human] | 0.8333 |
| macro_f1[priority] | 0.6240 |
| macro_f1[risk] | 0.3644 |
| mean_tvd[action] | 0.0723 |
| mean_tvd[category] | 0.0174 |
| mean_tvd[fraud] | 0.0000 |
| mean_tvd[needs_human] | 0.0000 |
| mean_tvd[priority] | 0.1801 |
| mean_tvd[risk] | 0.2572 |
| order_flip_rate[action] | 0.0833 |
| order_flip_rate[category] | 0.0167 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.1667 |
| order_flip_rate[risk] | 0.2667 |
| valid_accuracy | 0.7060 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| risk | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| action | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.4167 |
| priority | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| needs_human | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.5833 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
| fraud | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.7500 |
