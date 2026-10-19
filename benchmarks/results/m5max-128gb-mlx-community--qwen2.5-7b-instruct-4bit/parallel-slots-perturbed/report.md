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
| timestamp_utc | 2026-09-21T04:26:29+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6146 [0.5521, 0.6736] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.2812 |
| case_exact_match | 0.2812 |
| brier | 0.7424 [0.6680, 0.8240] (case_cluster_bootstrap) |
| correctness_auroc | 0.6284 |
| ece_5bin_equal_mass | 0.2223 |
| tie_rate | 0.0035 |
| any_flip_rate[action] | 0.3917 |
| any_flip_rate[category] | 0.2625 |
| any_flip_rate[fraud] | 0.0625 |
| any_flip_rate[needs_human] | 0.0458 |
| any_flip_rate[priority] | 0.6042 |
| any_flip_rate[risk] | 0.5000 |
| balanced_accuracy[action] | 0.4722 |
| balanced_accuracy[category] | 0.8073 |
| balanced_accuracy[fraud] | 0.4861 |
| balanced_accuracy[needs_human] | 0.7821 |
| balanced_accuracy[priority] | 0.3611 |
| balanced_accuracy[risk] | 0.4031 |
| macro_f1[action] | 0.4436 |
| macro_f1[category] | 0.8246 |
| macro_f1[fraud] | 0.4217 |
| macro_f1[needs_human] | 0.7699 |
| macro_f1[priority] | 0.3568 |
| macro_f1[risk] | 0.2621 |
| mean_tvd[action] | 0.3450 |
| mean_tvd[category] | 0.2789 |
| mean_tvd[fraud] | 0.0667 |
| mean_tvd[needs_human] | 0.0672 |
| mean_tvd[priority] | 0.5259 |
| mean_tvd[risk] | 0.4313 |
| order_flip_rate[action] | 0.3917 |
| order_flip_rate[category] | 0.2625 |
| order_flip_rate[fraud] | 0.0625 |
| order_flip_rate[needs_human] | 0.0458 |
| order_flip_rate[priority] | 0.6042 |
| order_flip_rate[risk] | 0.5000 |
| perturbation_flip_rate | 0.1806 |
| valid_accuracy | 0.6146 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| priority | 48 | 0.4167 [0.2885, 0.5572] (wilson) † | 0.5833 |
| risk | 48 | 0.4167 [0.2885, 0.5572] (wilson) | 0.4167 |
| action | 48 | 0.5417 [0.4029, 0.6742] (wilson) | 0.4167 |
| fraud | 48 | 0.7292 [0.5900, 0.8343] (wilson) † | 0.7500 |
| needs_human | 48 | 0.7708 [0.6346, 0.8669] (wilson) | 0.5833 |
| category | 48 | 0.8125 [0.6806, 0.8981] (wilson) | 0.3333 |
