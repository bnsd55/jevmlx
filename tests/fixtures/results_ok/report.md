# jevmlx eval report

## Environment

| key | value |
| --- | --- |
| chip | Apple M2 Pro |
| git_sha | ead13cbb067abee22f0af1f8e53cec7f175d0857 |
| jevmlx_version | 0.1.0 |
| machine_model | Mac14,9 |
| macos_version | 26.5.1 |
| mlx_lm_version | 0.31.3 |
| mlx_version | 0.32.2 |
| python_version | 3.12.2 |
| ram_gb | 32.0000 |
| timestamp_utc | 2026-09-17T19:37:07+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5000 [0.0000, 1.0000] (case_cluster_bootstrap) |
| case_exact_match | 0.5000 |
| brier | 0.7736 [0.0339, 1.5134] (case_cluster_bootstrap) |
| log_loss | 1.0894 [0.1394, 2.0394] (case_cluster_bootstrap) |
| correctness_auroc | 1.0000 |
| ece_5bin_equal_mass | 0.3000 |
| tie_rate | 0.0000 |
| balanced_accuracy[risk] | 0.5000 |
| exact_record_accuracy | 0.5000 |
| macro_f1[risk] | 0.3333 |
| majority_class_baseline[risk] | 0.5000 |
| valid_accuracy | 0.5000 |

## Per-field accuracy

| field | n | accuracy |
| --- | --- | --- |
| risk | 2 | 0.5000 [0.0945, 0.9055] (wilson) |
