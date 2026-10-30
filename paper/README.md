# Reproduce the CSA2026 v6 tables

Run from the `ForecastBot` project root (the directory containing `paper/`, `evaluation/`, and `requirements.txt`):

```bash
source .venv/bin/activate
python paper/scripts/build_tables.py
```

No database, API, or manuscript access is required. Install the dependencies from the project root if needed:

```bash
python -m pip install -r requirements.txt
```

The script uses pandas, numpy, scipy, and openpyxl. It computes Tables 2–6 from saved scores and case artifacts, with column headings defined in Python. The `paper/reference/` directory and the DOCX manuscript can be removed without affecting this script, experiments, or the demo. The `--reference` option is no longer used.

Table 1 is a literature overview, not an experimental calculation, and is no longer generated. No manuscript cell comparison, comparison worksheet, reference hash, or manuscript-match claim is produced.

Results are written to `paper/outputs/`: `table_2.csv` through `table_6.csv`, `CSA2026_tables.xlsx`, `TABLES.md`, cohort/case/pair/provenance audits, `paired_tests_48.csv`, and `manifest.json`. Rerunning overwrites these generated reports without modifying the source experiment artifacts. Existing `table_1.csv` and `paper_cell_comparison.csv` files are historical outputs; the script leaves them untouched and reports their presence. They are not outputs of the current workflow.

To preserve existing reports and obtain a clean output directory:

```bash
python paper/scripts/build_tables.py --output paper/outputs_new
```

Default paths are resolved relative to the script, so the root directory name `ForecastBot` requires no special configuration. A relative `--output` path is resolved from the current working directory.

## Table definitions

- Table 1: excluded; literature overview, not an experimental result.
- Table 2: Router accuracy, conditional/overall SQL accuracy, conditional/all-eligible forecast validity, explanation scores on shared valid cases, and E2E completion.
- Table 3: MAE/RMSE for valid forecasts with correct Router results and available holdout observations, including fallback outputs.
- Table 4: four explanation criteria on cases valid for both Gemini and Gemma (62 M5 and 59 Bus cases in this snapshot).
- Table 5: architecture-specific cohorts; SQL requires a correct Router result, while forecast validity requires correct upstream results and holdout eligibility.
- Table 6: pairs matched by dataset, model, and case ID. Forecasts must share a holdout; explanations must be valid and use the same judge; performance comparisons require both pipelines to succeed. Holm correction covers all 48 two-sided Wilcoxon tests, not only the four displayed metrics.

## Fallback definition requiring clarification in the paper

To reproduce Table 3, **Fallback means `deterministic_recent_mean`**, not every case with `fallback_used=True`.

| Dataset/model | Deterministic fallback (paper) | All fallback_used cases | Recovery using one LLM branch |
|---|---:|---:|---:|
| M5 / Gemini | 0 | 3 | 3 |
| M5 / Gemma | 9 | 14 | 5 |
| Bus / Gemini | 0 | 1 | 1 |
| Bus / Gemma | 13 | 15 | 2 |

`short_only` and `long_only` still use a forecast from one LLM branch. Therefore, “MAE w/o fallback” in the paper excludes deterministic fallback but retains branch-recovery cases. Suggested labels are “Deterministic fallback” and “MAE w/o deterministic fallback”. Replace “Gemini never used the fallback” with “Gemini never used deterministic fallback; branch recovery was used in three M5 cases and one Bus case.” The script does not read or edit a manuscript.

`cohort_audit.csv` records both counting conventions; `multi_agent_provenance.csv` records each case's selected mode. These definitions are applied directly to the saved experiment artifacts.

## Traceability and limitations

`paired_tests_48.csv` contains unrounded values, paired sample sizes, raw p-values, and Holm-adjusted p-values. `manifest.json` records SHA-256 hashes of the inputs actually read. The script checks coverage, duplicate IDs, required source availability, and fallback consistency, and stops on detected violations. It uses saved scores rather than rerunning LLM judging or fully revalidating ground truth.

The narrative-only 28-day Chronos benchmark and latency percentages are outside Tables 2–6 and are not reproduced by this script. The script does not generate or verify literature claims.
