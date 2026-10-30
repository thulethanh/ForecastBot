# Chat demo using the experimental runner

First restore the three PostgreSQL databases using [the database import guide](../../database/README.md) and configure `.env`.

Run from the package root:

```bash
source .venv/bin/activate
python -m streamlit run app_experiment_chat.py
```

Each submission calls `evaluation.runner.run_case(dataset, model, uuid, question)`. The runner controls the model, prompts, history windows, 0.6/0.4 weights, retries, fallback, and holdout policy. The original set contains 126 questions: 63 M5 and 63 HCMC_BUS. Two models yield 252 case/model configurations.

The question selector loads the experimental CSV files directly. Filter choices are retrieved from PostgreSQL; users can also enter filters manually. These controls help construct a question and do not replace actual database retrieval.

Experiment questions appear before the custom question builder. The builder displays only the fields used in the reference experiments: M5 uses `state_id`, `store_id`, and `item_id`; HCMC_BUS uses `route_id`, `operator`, and `vehicle_group`. Leave a value blank to omit that filter. An English preview of the custom question updates with the configuration.

## Workflow

1. Create or open a session in the sidebar; rename it if needed.
2. Select a dataset, model, time unit, horizon, and filters.
3. Use the action that inserts the generated question into the input, or select the original experimental question. Edit and submit it.
4. If the Router requests clarification, answer in the chat input. The application combines the request and clarification, then restarts the runner from the Router. The new turn links to the previous turn through `parent_id`.
5. On failure, inspect the details and return the question to the input. Check the currently selected dataset/model before resubmitting. The application does not infer unrestricted conversational context.

The demo interface and generated forecast requests use English. Original experiment questions, saved conversations, and model responses retain their original language.

## Persistence and execution

SQLite storage: `outputs/demo_chat/sessions.sqlite3`.

Case artifacts: `outputs/demo_chat/sessions/<session_id>/cases/<request_id>/`.

Results are saved to SQLite before file export, so a chart/export failure does not discard the chat turn. Rerunning the interface or reopening a session does not call the LLM. A running turn blocks another submission in the same session. If its process has stopped, mark it as interrupted before submitting a new turn. This action does not cancel an active HTTP request.

The runner's holdout policy is preserved: when sufficient history is available, forecasts target the final H held-out observations. These are not forecasts beyond the last date in the full database. If history is insufficient for holdout evaluation, the runner forecasts from the full history. Fallback use is reported in the response.

The configuration controls only construct the question. They do not force Router outputs or pass held-out target values to the model. The older demo's repeated-run settings and 0.5 ensemble weighting are not used.

Live execution requires the same environment settings, PostgreSQL data/schema, and Ollama/model configuration as the runner. This is a local demo without multi-user authentication. SQLite data, SQL, and traces may contain business information and should not be exposed publicly.

## Offline checks

These tests do not call live models or databases:

```bash
python -m unittest forecastbot.demo_chat.test_demo
```

## Session names and filters

A new session receives a title from the first question, whether drafted from filters/a sample case or submitted directly. Users can rename it; subsequent questions do not overwrite a manually assigned title. The sidebar shows the 30 most recent sessions and supports searching for older sessions by name. Ordering reflects creation, submission, and renaming, not interface reruns.

Filter catalogs use distinct database values cached for five minutes: M5 reads `m5_sales_train`; Bus reads `daily_passenger_stats`. Earlier selections restrict subsequent choices. Invalid selections are cleared, and fields with no valid choices are disabled. The refresh action clears the cache. Connection failures are reported and manual entry remains available; failures are not presented as empty catalogs. Catalog queries are read-only with an eight-second statement timeout.

The M5 catalog uses `MIN(d)`, the first reference date, to avoid scanning the full historical table for distinct values. The existing snapshot contains 30,490 series, 3,049 items, and 10 stores. For series introduced later, enter filters in the question or update the catalog source. This catalog restriction does not constrain the runner's forecasting SQL.

## Clarification in chat

When clarification is needed, the interface displays the follow-up question without opening error, SQL, or trace panels. The user replies directly in chat. The application combines the original request, the assistant's clarification question, and the new answer, then invokes the runner again. No separate input field or confirmation step is required.

The visible message preserves the user's submitted text; the complete combined request is stored for auditing. Clarification uses the dataset identified by the Router when valid and retains the previous model selection.

After a successful turn, the result remains visible. The next submitted message starts a new session without carrying over the previous clarification context. Completed sessions remain accessible in the sidebar. Actions that generate a question from configuration or a sample case explicitly start a new request.

## Bilingual explanations

Successful new demo runs preserve the original explanation under **Tiếng Việt** and automatically translate its summary and insights into **English** using `gpt-5.6-terra`. Set `OPENAI_API_KEY` in `.env` and install `openai` (included in `requirements.txt`). Existing sessions and failed translations offer **Translate to English**; reopening a session never automatically calls the translation API.

Translations are cached by source text, model, and translation version under `outputs/demo_chat/translations/`. The bilingual download contains `Tiếng Việt` and `English`, plus separate translation usage and duration. Translation failures leave the original forecast available. Runner prompts, original result JSON, evaluation outputs, and forecast timing/token metrics are unchanged. The source label assumes the runner's explanation is Vietnamese; the demo does not rewrite the original language.
