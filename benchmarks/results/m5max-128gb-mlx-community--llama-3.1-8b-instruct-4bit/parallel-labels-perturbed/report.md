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
| timestamp_utc | 2026-09-21T17:39:44+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7153 [0.6667, 0.7639] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.3333 |
| case_exact_match | 0.3333 |
| brier | 0.4768 [0.4252, 0.5300] (case_cluster_bootstrap) |
| correctness_auroc | 0.7131 |
| ece_5bin_equal_mass | 0.0592 |
| tie_rate | 0.0035 |
| any_flip_rate[action] | 0.2083 |
| any_flip_rate[category] | 0.0625 |
| any_flip_rate[fraud] | 0.0375 |
| any_flip_rate[needs_human] | 0.0167 |
| any_flip_rate[priority] | 0.0458 |
| any_flip_rate[risk] | 0.0333 |
| balanced_accuracy[action] | 0.5347 |
| balanced_accuracy[category] | 0.8385 |
| balanced_accuracy[fraud] | 0.9722 |
| balanced_accuracy[needs_human] | 0.5714 |
| balanced_accuracy[priority] | 0.7262 |
| balanced_accuracy[risk] | 0.5000 |
| macro_f1[action] | 0.5132 |
| macro_f1[category] | 0.8485 |
| macro_f1[fraud] | 0.9473 |
| macro_f1[needs_human] | 0.4375 |
| macro_f1[priority] | 0.7620 |
| macro_f1[risk] | 0.5119 |
| mean_tvd[action] | 0.1426 |
| mean_tvd[category] | 0.0535 |
| mean_tvd[fraud] | 0.0173 |
| mean_tvd[needs_human] | 0.0403 |
| mean_tvd[priority] | 0.0708 |
| mean_tvd[risk] | 0.0583 |
| order_flip_rate[action] | 0.2083 |
| order_flip_rate[category] | 0.0625 |
| order_flip_rate[fraud] | 0.0375 |
| order_flip_rate[needs_human] | 0.0167 |
| order_flip_rate[priority] | 0.0458 |
| order_flip_rate[risk] | 0.0333 |
| perturbation_flip_rate | 0.0880 |
| valid_accuracy | 0.7153 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 48 | 0.5000 [0.3639, 0.6361] (wilson) | 0.4167 |
| needs_human | 48 | 0.5000 [0.3639, 0.6361] (wilson) † | 0.5833 |
| risk | 48 | 0.6667 [0.5254, 0.7832] (wilson) | 0.4167 |
| priority | 48 | 0.8125 [0.6806, 0.8981] (wilson) | 0.5833 |
| category | 48 | 0.8542 [0.7283, 0.9275] (wilson) | 0.3333 |
| fraud | 48 | 0.9583 [0.8602, 0.9885] (wilson) | 0.7500 |
