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
| timestamp_utc | 2026-09-21T12:37:32+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.4306 [0.3472, 0.5278] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.0833 |
| case_exact_match | 0.0833 |
| brier | 0.8206 [0.7178, 0.9226] (case_cluster_bootstrap) |
| correctness_auroc | 0.7368 |
| ece_5bin_equal_mass | 0.2717 |
| tie_rate | 0.0000 |
| any_flip_rate[action] | 0.5833 |
| any_flip_rate[category] | 0.5000 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.4000 |
| any_flip_rate[risk] | 0.5333 |
| balanced_accuracy[action] | 0.1333 |
| balanced_accuracy[category] | 0.1667 |
| balanced_accuracy[fraud] | 0.5000 |
| balanced_accuracy[needs_human] | 0.6286 |
| balanced_accuracy[priority] | 0.3333 |
| balanced_accuracy[risk] | 0.5000 |
| macro_f1[action] | 0.1333 |
| macro_f1[category] | 0.0833 |
| macro_f1[fraud] | 0.4286 |
| macro_f1[needs_human] | 0.6250 |
| macro_f1[priority] | 0.2456 |
| macro_f1[risk] | 0.2222 |
| mean_tvd[action] | 0.2887 |
| mean_tvd[category] | 0.3874 |
| mean_tvd[fraud] | 0.0276 |
| mean_tvd[needs_human] | 0.0189 |
| mean_tvd[priority] | 0.2632 |
| mean_tvd[risk] | 0.3937 |
| order_flip_rate[action] | 0.5833 |
| order_flip_rate[category] | 0.5000 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.4000 |
| order_flip_rate[risk] | 0.5333 |
| valid_accuracy | 0.4306 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 12 | 0.1667 [0.0470, 0.4480] (wilson) † | 0.4167 |
| category | 12 | 0.1667 [0.0470, 0.4480] (wilson) † | 0.3333 |
| risk | 12 | 0.2500 [0.0889, 0.5323] (wilson) † | 0.4167 |
| priority | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| needs_human | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| fraud | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.7500 |
