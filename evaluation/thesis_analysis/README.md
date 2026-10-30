# ForecastBot thesis_analysis — self-contained v3

These scripts regenerate their supporting files and do not depend on previously generated supporting reports.

## Inputs

- `evaluation/results/gemini/*`
- `evaluation/results/gemma/*`
- `evaluation/results/llm_judge/*`
- Original case artifacts under `outputs/m5_exp/.../cases` and `outputs/hcmcbus_exp/.../cases`

## Run

From the package root:

```bash
python evaluation/thesis_analysis/scripts/run_all.py
```

Generated reports are written to `evaluation/thesis_analysis/outputs/`. Existing reports at the same paths are overwritten. To retain an earlier version, back up that output directory before running.

## Outputs

1. `01_overall_summary.csv`: overall ForecastBot results.
2. `02_router_sql_summary.csv`: Router and SQL results.
3. `03_forecast_summary.csv`: overall forecast results and day/month/quarter/year breakdowns.
4. `04_explanation_summary.csv`: explanation evaluation.
5. `05_performance_summary.csv`: E2E and per-agent performance.

`supporting/06_paired_wilcoxon.csv` supports the forecast comparison discussion. Additional files under `supporting/` are regenerated for traceability.

## E2E timing

Script 05 reads recorded E2E time from `metadata.json` or `result.json`. It does not substitute the sum of LLM call durations. If E2E timing cannot be found, the script stops and requires inspection of the case artifact schema.

These original thesis summaries use their own cohorts. For the exact CSA2026 manuscript tables, use `python paper/scripts/build_tables.py`; see `paper/README.md`.
