# ForecastBot 63-query evaluation

Before running live experiments, restore the three PostgreSQL databases and configure `.env` using [the database import guide](../database/README.md).

Place real UTF-8 CSV exports at:
- `evaluation/evaluation_set/m5_63_queries.csv`
- `evaluation/evaluation_set/hcmcbus_63_queries.csv`

Supported LLM configurations only:
- `gemini`: Gemini 2.5 Pro for all four LLM agents.
- `gemma`: Gemma 4 (`gemma4:e4b`) through Ollama for all four LLM agents.

Smoke tests:
```bash
python -m evaluation.runner --dataset M5 --model gemini --mode case
python -m evaluation.runner --dataset HCMC_BUS --model gemini --mode case
python -m evaluation.runner --dataset M5 --model gemma --mode case
python -m evaluation.runner --dataset HCMC_BUS --model gemma --mode case
```
Use `--case-id M01` / `H01` if those IDs exist in the CSV.

Full runs after all four smoke tests succeed:
```bash
python -m evaluation.runner --dataset M5 --model gemini --mode full --resume
python -m evaluation.runner --dataset HCMC_BUS --model gemini --mode full --resume
python -m evaluation.runner --dataset M5 --model gemma --mode full --resume
python -m evaluation.runner --dataset HCMC_BUS --model gemma --mode full --resume
```

Outputs are isolated by dataset and LLM under `outputs/m5_exp/...` and `outputs/hcmcbus_exp/...`.

Forecast policy:
- day: h180/h365, weights 0.6/0.4.
- month: h24/h36 for H<=6, h36/h60 otherwise.
- quarter: h8/h12 for H<=4, h12/h20 otherwise.
- year: h5/h10 for H<=3, h10/h20 otherwise.
- actual windows are capped by available history; identical windows invoke the LLM once.

Evaluation policy:
- Production forecast always attempts the full requested horizon.
- Backtest holds out the final H observations only when enough training history remains.
- Metrics: MAE and RMSE only.
- Insufficient long-horizon history => `skipped_insufficient_history`, not failure.

Each LLM invocation trace records prompt, raw response, provider/model, tokens when supplied by backend, latency, attempt, status/error. Summary records component times and end-to-end time.

### Score Gemini single-agent ablation

Run from the project root after generating the single-agent artifacts:

```bash
python -m evaluation.scoring.evaluate_gemini --architecture single_agent
```

This scores both datasets from `outputs/single_agent_ablation/{m5_exp,hcmcbus_exp}/gemini_2_5_pro/cases/`
and writes eight CSV reports to `evaluation/results/single_agent_ablation/gemini/`.
It reuses the existing ground truth and scoring rules, makes no LLM calls, and does not score explanations.
The default `--architecture multi_agent` retains the original input and result paths.
In single-agent reports, `orchestration` refers to request extraction quality, not a separate agent.

Gemma uses the same architecture selector:

```bash
python -m evaluation.scoring.evaluate_gemma --architecture single_agent
```

Scores go to `evaluation/results/single_agent_ablation/gemma/`. Run after the
Gemma cases finish; scoring an incomplete run produces provisional reports.

Explanation scoring calls the OpenAI judge and keeps a separate single-agent cache:

```bash
python -m evaluation.scoring.evaluate_explanation --architecture single_agent --target-model gemini --dataset all --judge-model gpt-5.6-terra --resume
python -m evaluation.scoring.evaluate_explanation --architecture single_agent --target-model gemma --dataset all --judge-model gpt-5.6-terra --resume
```

Results and `judge_raw` are under
`evaluation/results/single_agent_ablation/llm_judge/{gemini,gemma}/`.
Use the same judge configuration as the comparison experiment. `--resume` reuses
successful cached judgments; do not reuse that cache after changing source artifacts
or judge configuration. Without `--architecture`, original multi-agent paths remain in use.
