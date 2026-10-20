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
| timestamp_utc | 2026-09-21T09:39:11+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5729 [0.5139, 0.6319] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.2292 |
| case_exact_match | 0.2292 |
| brier | 0.8664 [0.7709, 0.9631] (case_cluster_bootstrap) |
| correctness_auroc | 0.7424 |
| ece_5bin_equal_mass | 0.3321 |
| tie_rate | 0.0174 |
| any_flip_rate[action] | 0.3500 |
| any_flip_rate[category] | 0.3625 |
| any_flip_rate[fraud] | 0.0250 |
| any_flip_rate[needs_human] | 0.0125 |
| any_flip_rate[priority] | 0.3833 |
| any_flip_rate[risk] | 0.5333 |
| balanced_accuracy[action] | 0.5000 |
| balanced_accuracy[category] | 0.4219 |
| balanced_accuracy[fraud] | 0.7917 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.4048 |
| balanced_accuracy[risk] | 0.2094 |
| macro_f1[action] | 0.4373 |
| macro_f1[category] | 0.4046 |
| macro_f1[fraud] | 0.8360 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.3750 |
| macro_f1[risk] | 0.2009 |
| mean_tvd[action] | 0.3378 |
| mean_tvd[category] | 0.3648 |
| mean_tvd[fraud] | 0.0130 |
| mean_tvd[needs_human] | 0.0250 |
| mean_tvd[priority] | 0.3645 |
| mean_tvd[risk] | 0.4500 |
| order_flip_rate[action] | 0.3500 |
| order_flip_rate[category] | 0.3625 |
| order_flip_rate[fraud] | 0.0250 |
| order_flip_rate[needs_human] | 0.0125 |
| order_flip_rate[priority] | 0.3833 |
| order_flip_rate[risk] | 0.5333 |
| perturbation_flip_rate | 0.1343 |
| valid_accuracy | 0.5729 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| risk | 48 | 0.3125 [0.1995, 0.4533] (wilson) † | 0.4167 |
| category | 48 | 0.4792 [0.3447, 0.6167] (wilson) | 0.3333 |
| action | 48 | 0.5417 [0.4029, 0.6742] (wilson) | 0.4167 |
| needs_human | 48 | 0.5833 [0.4428, 0.7115] (wilson) | 0.5833 |
| priority | 48 | 0.6250 [0.4836, 0.7478] (wilson) | 0.5833 |
| fraud | 48 | 0.8958 [0.7783, 0.9547] (wilson) | 0.7500 |
