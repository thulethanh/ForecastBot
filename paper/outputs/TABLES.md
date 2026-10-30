# CSA2026 tables

Tables 2–6 are computed from saved scores and artifacts without a manuscript. Table 1 (literature overview) is not generated.

## Table 2

| Dataset – model | Router | SQL | Valid forecast | Expl. | E2E |
| --- | --- | --- | --- | --- | --- |
| M5 – Gemini | 95.24 | 100.00 / 95.24 | 100.00 / 94.44 | 4.78 | 98.41 |
| M5 – Gemma 4 | 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 4.43 | 100.00 |
| Bus – Gemini | 95.24 | 100.00 / 95.24 | 100.00 / 93.18 | 4.80 | 95.24 |
| Bus – Gemma 4 | 96.83 | 100.00 / 96.83 | 100.00 / 97.73 | 4.48 | 98.41 |

## Table 3

| Dataset – model | Cases | Fallback | MAE | RMSE | MAE w/o fallback |
| --- | --- | --- | --- | --- | --- |
| M5 – Gemini | 51 | 0 (0%) | 63.71 | 70.70 | 63.71 |
| M5 – Gemma 4 | 54 | 9 (16.7%) | 148.06 | 155.28 | 177.25 |
| Bus – Gemini | 41 | 0 (0%) | 12,044 | 12,879 | 12,044 |
| Bus – Gemma 4 | 43 | 13 (30.2%) | 23,691 | 25,097 | 33,790 |

## Table 4

| Dataset – model | Relevance | Faithfulness | Clarity | Usefulness | Mean |
| --- | --- | --- | --- | --- | --- |
| M5 – Gemini | 5.00 | 4.34 | 4.98 | 4.81 | 4.78 |
| M5 – Gemma 4 | 4.79 | 4.05 | 4.77 | 4.11 | 4.43 |
| Bus – Gemini | 5.00 | 4.36 | 5.00 | 4.83 | 4.80 |
| Bus – Gemma 4 | 4.78 | 4.10 | 4.86 | 4.19 | 4.48 |

## Table 5

| Dataset – model | Router (M / S) | SQL (M / S) | Valid forecast (M / S) | Completion (M / S) |
| --- | --- | --- | --- | --- |
| Bus – Gemini | 95.2 / 93.7 | 100.0 / 100.0 | 100.0 / 100.0 | 95.2 / 100.0 |
| Bus – Gemma 4 | 96.8 / 93.7 | 100.0 / 100.0 | 100.0 / 100.0 | 98.4 / 98.4 |
| M5 – Gemini | 95.2 / 92.1 | 100.0 / 100.0 | 100.0 / 100.0 | 98.4 / 100.0 |
| M5 – Gemma 4 | 100.0 / 96.8 | 100.0 / 91.8 | 100.0 / 100.0 | 100.0 / 92.1 |

## Table 6

| Dataset – model | MAE (M / S) | Expl. (M / S) | E2E s (M / S) | Tokens (M / S) |
| --- | --- | --- | --- | --- |
| Bus – Gemini | 10,227 / 11,360 | 4.80 / 4.79 | 111.9 / 75.0* | 19,801 / 14,472* |
| Bus – Gemma 4 | 24,038 / 25,163 | 4.48 / 4.63 | 365.1 / 376.6 | 10,368 / 9,348* |
| M5 – Gemini | 42.3 / 45.4 | 4.78 / 4.77 | 152.0 / 87.5* | 24,062 / 15,939* |
| M5 – Gemma 4 | 167.6 / 148.8 | 4.46 / 4.54 | 377.3 / 406.5 | 9,447 / 8,359* |

## Verification

Input coverage and provenance checks passed; no manuscript comparison is performed.
See cohort_audit.csv for denominators and paired_tests_48.csv for unrounded values, paired N, raw p and Holm-adjusted p.
Table 2 SQL: correct Router / all cases. Forecast: correct Router and eligible / all eligible cases. Table 3 retains valid forecasts in the conditional cohort, including fallback.
Table 4 uses shared valid explanations across Gemini/Gemma within a dataset (same judge). Table 5 uses architecture-specific upstream cohorts. Table 6 uses common forecast holdouts, common valid explanations, and successful paired pipelines for performance; finite values are filtered per metric.
IMPORTANT: Table 3 Fallback means deterministic_recent_mean only. MAE w/o fallback retains short_only/long_only LLM branch recovery. The broader fallback_used flag counts both kinds; cohort_audit.csv records both. The paper should explicitly define this distinction; Gemini does use branch recovery despite zero deterministic fallback.
Stars use two-sided Wilcoxon and Holm over all 48 tests, including strict-cohort and criterion tests. No hard-coded numerical paper results are used.
Narrative-only Chronos benchmark and latency percentages are outside these experimental tables and are not reproduced by this script.
