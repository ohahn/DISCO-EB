# Benchmarks

All benchmarks are run using `pytest`. You can find the code run in [tests/test_background.py](tests/test_background.py) and [tests/test_perturbations.py](tests/test_perturbations.py).

## NVIDIA GeForce RTX 3090

| Version       | [`0b03016bf1de1dd0f06c17468f1c552510f93982`](https://github.com/ohahn/DISCO-EB/commit/0b03016bf1de1dd0f06c17468f1c552510f93982) |
|---------------|---------------------------------------------------------------------------------------------------------------------------------|
| jax           | 0.4.33                                                                                                                          |
| jaxlib        | 0.4.33                                                                                                                          |
| numpy         | 2.1.1                                                                                                                           |
| python        | 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0]                                                                               |
| platform      | Linux astroadmin-MS-7C37 6.8.0-45-generic #45-Ubuntu                                                                            |
| CUDA Version  | 12.5                                                                                                                            |
| NVIDIA Driver | 555.42.06                                                                                                                       |
| runtime tests | 494.91s (0:08:14)                                                                                                               |

```text
-------------------------------------------------------------------------------------------------- benchmark: 3 tests --------------------------------------------------------------------------------------------------
Name (time in s)                                                 Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_evolve_background_recfast_vs_DISCOEB_v0_1_0_baseline     1.2663 (1.0)      1.2672 (1.0)      1.2668 (1.0)      0.0003 (2.50)     1.2668 (1.0)      0.0004 (2.68)          2;0  0.7894 (1.0)           5           1
test_benchmark_evolve_perturbations_batched                   4.2246 (3.34)     4.2249 (3.33)     4.2248 (3.34)     0.0001 (1.0)      4.2248 (3.34)     0.0002 (1.0)           2;0  0.2367 (0.30)          5           1
test_benchmark_evolve_perturbations                           5.1479 (4.07)     5.1492 (4.06)     5.1487 (4.06)     0.0006 (4.64)     5.1490 (4.06)     0.0011 (6.48)          1;0  0.1942 (0.25)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## NVIDIA A100-PCIE-40GB

- single GPU

| Version       | [`0b03016bf1de1dd0f06c17468f1c552510f93982`](https://github.com/ohahn/DISCO-EB/commit/0b03016bf1de1dd0f06c17468f1c552510f93982) |
|---------------|---------------------------------------------------------------------------------------------------------------------------------|
| jax           | 0.4.33                                                                                                                          |
| jaxlib        | 0.4.33                                                                                                                          |
| numpy         | 2.1.1                                                                                                                           |
| python        | 3.11.3 (main, May 24 2023, 13:28:22) [GCC 12.2.0]                                                                               |
| platform      | Linux n3071-001 4.18.0-348.12.2.el8_5.x86_64                                                                                    |
| CUDA Version  | 12.3                                                                                                                            |
| NVIDIA Driver | 545.23.08                                                                                                                       |
| runtime tests | 386.07s (0:06:26)                                                                                                               |

```text
-------------------------------------------------------------------------------------------------- benchmark: 3 tests --------------------------------------------------------------------------------------------------
Name (time in s)                                                 Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_evolve_background_recfast_vs_DISCOEB_v0_1_0_baseline     1.7406 (1.0)      1.8040 (1.0)      1.7757 (1.0)      0.0256 (42.99)    1.7704 (1.0)      0.0393 (53.22)         2;0  0.5632 (1.0)           5           1
test_benchmark_evolve_perturbations_batched                   3.4632 (1.99)     3.4647 (1.92)     3.4637 (1.95)     0.0006 (1.0)      3.4636 (1.96)     0.0007 (1.0)           1;0  0.2887 (0.51)          5           1
test_benchmark_evolve_perturbations                           4.1515 (2.39)     4.1561 (2.30)     4.1535 (2.34)     0.0017 (2.87)     4.1532 (2.35)     0.0023 (3.08)          2;0  0.2408 (0.43)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## NVIDIA A40

- single GPU

| Version       | [`0b03016bf1de1dd0f06c17468f1c552510f93982`](https://github.com/ohahn/DISCO-EB/commit/0b03016bf1de1dd0f06c17468f1c552510f93982) |
|---------------|---------------------------------------------------------------------------------------------------------------------------------|
| jax           | 0.4.33                                                                                                                          |
| jaxlib        | 0.4.33                                                                                                                          |
| numpy         | 2.1.1                                                                                                                           |
| python        | 3.11.3 (main, May 24 2023, 13:28:22) [GCC 12.2.0]                                                                               |
| platform      | Linux n3067-016 4.18.0-477.10.1.el8_8.x86_64                                                                                    |
| CUDA Version  | 12.3                                                                                                                            |
| NVIDIA Driver | 545.23.08                                                                                                                       |
| runtime tests | 1168.34s (0:19:28)                                                                                                              |

```text
---------------------------------------------------------------------------------------------------- benchmark: 3 tests ----------------------------------------------------------------------------------------------------
Name (time in s)                                                  Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_evolve_background_recfast_vs_DISCOEB_v0_1_0_baseline      3.2201 (1.0)       3.2277 (1.0)       3.2243 (1.0)      0.0037 (19.69)     3.2257 (1.0)      0.0069 (32.55)         1;0  0.3101 (1.0)           5           1
test_benchmark_evolve_perturbations_batched                   10.9231 (3.39)     10.9248 (3.38)     10.9240 (3.39)     0.0007 (3.81)     10.9240 (3.39)     0.0012 (5.50)          2;0  0.0915 (0.30)          5           1
test_benchmark_evolve_perturbations                           13.8567 (4.30)     13.8572 (4.29)     13.8570 (4.30)     0.0002 (1.0)      13.8571 (4.30)     0.0002 (1.0)           1;0  0.0722 (0.23)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## NVIDIA RTX 4090

| Version       | [`0b03016bf1de1dd0f06c17468f1c552510f93982`](https://github.com/ohahn/DISCO-EB/commit/0b03016bf1de1dd0f06c17468f1c552510f93982) |
|---------------|---------------------------------------------------------------------------------------------------------------------------------|
| python        | 3.12.6                                                                            |
| runtime tests | 262.48s (0:04:22)                                                                                                             |

(units below are in ms unlike in the other tests)

```text

---------------------------------------------------------------------------------------------------------- benchmark: 3 tests ----------------------------------------------------------------------------------------------------------
Name (time in ms)                                                    Min                   Max                  Mean            StdDev                Median               IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_evolve_background_recfast_vs_DISCOEB_v0_1_0_baseline       983.3992 (1.0)        984.4454 (1.0)        983.8281 (1.0)      0.3887 (1.0)        983.7525 (1.0)      0.4361 (1.0)           2;0  1.0164 (1.0)           5           1
test_benchmark_evolve_perturbations_batched                   2,488.2322 (2.53)     2,489.4975 (2.53)     2,488.8748 (2.53)     0.4557 (1.17)     2,488.8383 (2.53)     0.4903 (1.12)          2;0  0.4018 (0.40)          5           1
test_benchmark_evolve_perturbations                           2,961.1251 (3.01)     2,962.3737 (3.01)     2,961.9982 (3.01)     0.5093 (1.31)     2,962.0995 (3.01)     0.5390 (1.24)          1;0  0.3376 (0.33)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```