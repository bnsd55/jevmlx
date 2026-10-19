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
| timestamp_utc | 2026-09-21T12:26:23+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7847 [0.7326, 0.8333] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.5104 |
| case_exact_match | 0.5104 |
| brier | 0.5410 [0.4725, 0.6119] (case_cluster_bootstrap) |
| correctness_auroc | 0.8146 |
| ece_5bin_equal_mass | 0.1870 |
| tie_rate | 0.0035 |
| any_flip_rate[action] | 0.0167 |
| any_flip_rate[category] | 0.0333 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0125 |
| any_flip_rate[priority] | 0.0708 |
| any_flip_rate[risk] | 0.0708 |
| balanced_accuracy[action] | 0.8333 |
| balanced_accuracy[category] | 0.8854 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.6786 |
| balanced_accuracy[priority] | 0.6488 |
| balanced_accuracy[risk] | 0.8750 |
| macro_f1[action] | 0.8110 |
| macro_f1[category] | 0.8794 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.6080 |
| macro_f1[priority] | 0.6286 |
| macro_f1[risk] | 0.8359 |
| mean_tvd[action] | 0.0233 |
| mean_tvd[category] | 0.0301 |
| mean_tvd[fraud] | 0.0043 |
| mean_tvd[needs_human] | 0.0166 |
| mean_tvd[priority] | 0.0811 |
| mean_tvd[risk] | 0.0792 |
| order_flip_rate[action] | 0.0167 |
| order_flip_rate[category] | 0.0333 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0125 |
| order_flip_rate[priority] | 0.0708 |
| order_flip_rate[risk] | 0.0708 |
| perturbation_flip_rate | 0.0370 |
| valid_accuracy | 0.7847 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 48 | 0.6250 [0.4836, 0.7478] (wilson) | 0.5833 |
| priority | 48 | 0.6458 [0.5044, 0.7657] (wilson) | 0.5833 |
| action | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.4167 |
| fraud | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.7500 |
| category | 48 | 0.8750 [0.7530, 0.9414] (wilson) | 0.3333 |
| risk | 48 | 0.8958 [0.7783, 0.9547] (wilson) | 0.4167 |
