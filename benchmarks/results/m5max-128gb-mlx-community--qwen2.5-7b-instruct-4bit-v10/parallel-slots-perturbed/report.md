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
| timestamp_utc | 2026-09-23T12:06:50+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7095 [0.6522, 0.7627] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.2083 |
| case_exact_match | 0.2083 |
| brier | 0.7406 [0.6562, 0.8311] (case_cluster_bootstrap) |
| correctness_auroc | 0.7808 |
| ece_5bin_equal_mass | 0.2137 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.0792 |
| any_flip_rate[category] | 0.0208 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.1708 |
| any_flip_rate[risk] | 0.2583 |
| balanced_accuracy[action] | 0.5481 |
| balanced_accuracy[category] | 0.9201 |
| balanced_accuracy[fraud] | 0.8333 |
| balanced_accuracy[needs_human] | 0.8571 |
| balanced_accuracy[priority] | 0.7351 |
| balanced_accuracy[risk] | 0.4193 |
| macro_f1[action] | 0.4849 |
| macro_f1[category] | 0.9180 |
| macro_f1[fraud] | 0.8737 |
| macro_f1[needs_human] | 0.8333 |
| macro_f1[priority] | 0.6685 |
| macro_f1[risk] | 0.3681 |
| mean_tvd[action] | 0.0816 |
| mean_tvd[category] | 0.0203 |
| mean_tvd[fraud] | 0.0000 |
| mean_tvd[needs_human] | 0.0000 |
| mean_tvd[priority] | 0.1652 |
| mean_tvd[risk] | 0.2478 |
| order_flip_rate[action] | 0.0792 |
| order_flip_rate[category] | 0.0208 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.1708 |
| order_flip_rate[risk] | 0.2583 |
| perturbation_flip_rate | 0.0972 |
| valid_accuracy | 0.7095 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| risk | 48 | 0.4583 [0.3258, 0.5971] (wilson) | 0.4167 |
| action | 48 | 0.5000 [0.3639, 0.6361] (wilson) | 0.4167 |
| priority | 48 | 0.7500 [0.6122, 0.8508] (wilson) | 0.5833 |
| needs_human | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.5833 |
| fraud | 48 | 0.9167 [0.8045, 0.9671] (wilson) | 0.7500 |
| category | 48 | 0.9375 [0.8316, 0.9785] (wilson) | 0.3333 |
