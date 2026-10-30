# ForecastBot

ForecastBot: A Multiagent LLM Framework for Natural Language Time Series Forecasting over Relational Databases

This standalone package contains the original evaluation workflow and a demo that uses the same runner. The original project outside this folder is unchanged.

## Package contents

- **126 scenarios: 63 M5 + 63 HCMC_BUS**, rather than 128. Two models produce 252 scenario/model configurations per architecture.
- `evaluation/runner.py`: the original multi-agent pipeline, preserving prompts, retries, fallback, and holdout policies.
- `app_experiment_chat.py` and `forecastbot/demo_chat/`: a demo that calls `evaluation.runner.run_case()` directly.
- `evaluation/ground_truth/`: reference requests, SQL queries, and CSV data used for scoring.
- `outputs/m5_exp/` and `outputs/hcmcbus_exp/`: existing result snapshots, not newly executed runs or a claim that every case succeeded.
- `evaluation/results/`: saved evaluation scores and LLM judge results.
- `evaluation/thesis_analysis/scripts/`: scripts for regenerating summary tables.
- `evaluation/single_agent_runner.py`, single-agent artifacts, and scores: inputs for the CSA2026 architecture comparison.
- `paper/`: the independent table-generation script, generated tables, and audit reports.

The package excludes the original app_threads demo, the newer LangGraph runner, the legacy 42-case set, credentials, `.env`, and personal chat history.

## Installation

Run all commands from the project root (`ForecastBot`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Fill in the settings in `.env`. The copy command preserves an existing `.env`. Use `requirements.txt` for the complete dependency list. `requirements-lock.txt` is a historical environment snapshot and omits scipy, openpyxl, and openai; it is not sufficient on its own for the current table-generation and translation features.

## Restore PostgreSQL before live runs

Database backups and restore instructions are in [database/README.md](database/README.md). Restore `forecastbot`, `m5_db`, and `hcmcbus_db`, verify their data, and configure `.env` before running either experiments or the connected demo.

## Run the connected demo

```bash
python -m streamlit run app_experiment_chat.py
```

Submitting a question makes live model/database calls. Session results are stored separately under `outputs/demo_chat/`. The demo does not currently provide a viewer for the saved paper artifacts. When sufficient history is available, the runner forecasts the final H held-out observations; these are not forecasts beyond the last date in the database. See [the demo guide](forecastbot/demo_chat/README.md).

## Regenerate the CSA2026 paper tables

```bash
python paper/scripts/build_tables.py
```

Outputs are written to `paper/outputs/`: `CSA2026_tables.xlsx`, `TABLES.md`, `table_2.csv` through `table_6.csv`, case/pair audits, and 48 tests with Holm correction.

Tables 2–6 are computed from saved case-level scores and artifacts without a reference manuscript. Table 1 (literature overview) and manuscript comparisons are not generated. The `paper/reference/` directory is not required. Existing Table 1 and manuscript comparison files are historical reports and are not refreshed. This command makes no API/database calls and does not rerun forecasts. Use `--output` to select another output directory. See [the paper reproduction guide](paper/README.md) for fallback definitions and scope.

## Regenerate the original evaluation summaries

```bash
python -m evaluation.scoring.evaluate_gemini
python -m evaluation.scoring.evaluate_gemma
python evaluation/thesis_analysis/scripts/run_all.py
```

These commands do not call models. They rewrite score and summary files inside this package. Explanation summaries use saved judge scores. Running `evaluation.scoring.evaluate_explanation` separately requires OpenAI API access and may incur charges; consult `--help` and explicitly select the judge model.

## Run new experiments

The following commands call live models/databases and may overwrite the snapshot for the same case. Make a separate copy of the package for a new experiment. `--resume` skips existing successful cases; it does not create an independent run.

```bash
python -m evaluation.runner --dataset M5 --model gemini --mode case --case-id M01
python -m evaluation.runner --dataset M5 --model gemini --mode full
python -m evaluation.runner --dataset HCMC_BUS --model gemini --mode full
python -m evaluation.runner --dataset M5 --model gemma --mode full
python -m evaluation.runner --dataset HCMC_BUS --model gemma --mode full
```

## Reproduction requirements

The complete local distribution includes three PostgreSQL dumps directly under `database/`. For a GitHub clone, download the dump files from the repository Release and place them there; see [restore instructions](database/README.md). Live pipeline/demo execution requires three database connections: ForecastBot metadata (the `schema_knowledge` table), M5, and HCMC_BUS. The business data and schema knowledge must match the original experiment. Ground-truth CSV files do not fully replace these databases. Metadata queries are defined in `forecastbot/db/schema_repository.py`; reference SQL is under `evaluation/ground_truth/`.

On a fresh machine, obtain and restore all three database backups before live execution. The configured Gemini/Ollama models must remain available, and new LLM responses may differ from the saved snapshot.

## Offline checks

```bash
python -m unittest forecastbot.demo_chat.test_demo forecastbot.demo_chat.test_translation
python -m unittest evaluation.architecture_comparison.test_analysis
```

`PACKAGE_MANIFEST.json` records the SHA-256 hashes of the distributed files, including the database assets supplied separately through Releases. Regenerating or rescoring results may change these hashes. This is a package prepared for a paper submission, not certification of compliance with a particular conference's artifact requirements.
