# ForecastBot — four appendix summary tables

This script set replaces the earlier eight per-case presentation tables with four summary tables.

## Outputs

| File | Contents |
|---|---|
| `PL_A1_forecast_by_granularity.csv` | Day/month/quarter/year breakdowns, valid forecast counts, mean and median MAE/RMSE |
| `PL_A2_agent_performance.csv` | Router/SQL/Forecast/Explanation case counts, LLM calls, errors, timing, and tokens |
| `PL_A3_e2e_performance.csv` | E2E performance by dataset/model, mean, median, minimum, maximum, and tokens |
| `PL_A4_paired_wilcoxon.csv` | Paired Gemini 2.5 Pro vs. Gemma 4 (`gemma4:e4b`) comparisons: MAE/RMSE, pair counts, wins/losses/ties, Wilcoxon statistic, and p-value |

## Run

From the package root:

```bash
python evaluation/thesis_analysis/scripts/run_appendix_summary.py
```

Reports are written under `evaluation/thesis_analysis/outputs/appendix/`. Back up previous reports if needed before regenerating them. Retain the original per-case files as reproduction evidence, even when only the summary tables are included in the thesis.
