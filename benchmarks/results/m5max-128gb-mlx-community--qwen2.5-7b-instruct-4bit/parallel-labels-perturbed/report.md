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
| timestamp_utc | 2026-09-21T06:31:33+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7303 [0.6615, 0.7922] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.4271 |
| case_exact_match | 0.4271 |
| brier | 0.6842 [0.5885, 0.7883] (case_cluster_bootstrap) |
| correctness_auroc | 0.6967 |
| ece_5bin_equal_mass | 0.2198 |
| tie_rate | 0.0006 |
| any_flip_rate[action] | 0.0542 |
| any_flip_rate[category] | 0.0042 |
| any_flip_rate[fraud] | 0.0208 |
| any_flip_rate[needs_human] | 0.0125 |
| any_flip_rate[priority] | 0.1292 |
| any_flip_rate[risk] | 0.0667 |
| balanced_accuracy[action] | 0.5370 |
| balanced_accuracy[category] | 0.9974 |
| balanced_accuracy[fraud] | 0.5903 |
| balanced_accuracy[needs_human] | 0.8446 |
| balanced_accuracy[priority] | 0.8095 |
| balanced_accuracy[risk] | 0.4214 |
| macro_f1[action] | 0.5463 |
| macro_f1[category] | 0.9970 |
| macro_f1[fraud] | 0.5929 |
| macro_f1[needs_human] | 0.8229 |
| macro_f1[priority] | 0.7155 |
| macro_f1[risk] | 0.4168 |
| mean_tvd[action] | 0.0535 |
| mean_tvd[category] | 0.0054 |
| mean_tvd[fraud] | 0.0166 |
| mean_tvd[needs_human] | 0.0339 |
| mean_tvd[priority] | 0.0948 |
| mean_tvd[risk] | 0.0747 |
| order_flip_rate[action] | 0.0542 |
| order_flip_rate[category] | 0.0042 |
| order_flip_rate[fraud] | 0.0208 |
| order_flip_rate[needs_human] | 0.0125 |
| order_flip_rate[priority] | 0.1292 |
| order_flip_rate[risk] | 0.0667 |
| perturbation_flip_rate | 0.0463 |
| valid_accuracy | 0.7303 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 48 | 0.5208 [0.3833, 0.6553] (wilson) | 0.4167 |
| risk | 48 | 0.5208 [0.3833, 0.6553] (wilson) | 0.4167 |
| priority | 48 | 0.7292 [0.5900, 0.8343] (wilson) | 0.5833 |
| fraud | 48 | 0.7917 [0.6574, 0.8827] (wilson) | 0.7500 |
| needs_human | 48 | 0.8125 [0.6806, 0.8981] (wilson) | 0.5833 |
| category | 48 | 1.0000 [0.9259, 1.0000] (wilson) | 0.3333 |
