from __future__ import annotations

"""
Single-Agent ablation runner for ForecastBot.

Design:
- ONE LLM agent role is reused across the whole case.
- Phase 1: understand request + generate SQL.
- Python executes SQL, performs the SAME holdout split/history-window rules
  as the existing multi-agent runner.
- Phase 2: the SAME agent role forecasts short/long history windows.
- Python combines forecasts with the SAME 0.6/0.4 rule.
- Phase 3: the SAME agent role explains the final forecast.
- No LangGraph is used here: this is intentionally a controlled Single-Agent
  ablation against the existing Python-orchestrated Multi-Agent runner.

Important:
- Ground-truth test rows are NEVER supplied to the LLM.
- SQL execution, holdout split, window selection, ensemble and artifact writing
  remain deterministic Python operations for fairness.
"""

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from forecastbot.llm_provider import LLMProvider
from forecastbot.agents.utils import extract_json
from forecastbot.db.schema_repository import SchemaKnowledgeRepository
from forecastbot.db.postgres import fetch_dataframe


ROOT = Path(__file__).resolve().parents[1]
SET_DIR = ROOT / "evaluation" / "evaluation_set"

# For the ablation, run Gemini first so model is held constant.
MODEL_MAP = {
    "gemini": ("gemini", "gemini-2.5-pro"),
    # Kept optional for later extension.
    "gemma": ("gemma", "gemma4:e4b"),
}

ALLOWED = {
    "day": [7, 28, 30, 60, 90],
    "month": [1, 3, 6, 12, 24],
    "quarter": [1, 2, 3, 4, 8, 12],
    "year": [1, 2, 3, 5, 10],
}

VALID_FILTERS = {
    "M5": {"item_id", "store_id", "state_id", "cat_id", "dept_id", "id"},
    "HCMC_BUS": {
        "route_id",
        "route_name",
        "operator",
        "vehicle_group",
        "start_stop",
        "end_stop",
    },
}


# ---------------------------------------------------------------------
# Single-Agent prompts
# ---------------------------------------------------------------------

SINGLE_AGENT_REQUEST_SQL_PROMPT = r"""
You are ForecastBot's SINGLE forecasting agent.

You are responsible for understanding the user's forecasting request and
generating the SQL needed to retrieve the corresponding time series.

USER REQUEST:
{user_query}

EXPECTED DATASET FOR THIS EVALUATION CASE:
{expected_dataset}

DATABASE SCHEMA KNOWLEDGE:
{llm_schema_context}

SQL GENERATION RULES:
{sql_generation_rules}

Allowed granularities and horizons:
- day: 7, 28, 30, 60, 90
- month: 1, 3, 6, 12, 24
- quarter: 1, 2, 3, 4, 8, 12
- year: 1, 2, 3, 5, 10

Requirements:
1. Infer dataset, granularity, horizon and filters from the user's request.
2. Do not invent filters that are not stated or semantically implied.
3. Generate SQL using only the supplied schema/rules.
4. SQL must return exactly two logical columns:
   - ds: time index
   - y: numeric value
5. Return STRICT JSON only. No Markdown fences and no extra text.

Output schema:
{{
  "intent": "forecast_request",
  "dataset": "M5 or HCMC_BUS or null",
  "granularity": "day or month or quarter or year or null",
  "horizon": 7,
  "filters": {{}},
  "missing_fields": [],
  "needs_clarification": false,
  "user_message_for_clarification": "",
  "sql": "SELECT ...",
  "sql_explanation": "brief explanation"
}}
""".strip()


SINGLE_AGENT_SQL_REPAIR_PROMPT = r"""
You are the SAME ForecastBot SINGLE forecasting agent.

The SQL you previously generated failed during execution.
Repair ONLY the SQL while preserving the forecasting request.

FORECAST REQUEST:
{forecast_request}

DATABASE SCHEMA KNOWLEDGE:
{llm_schema_context}

SQL GENERATION RULES:
{sql_generation_rules}

PREVIOUS SQL:
{previous_sql}

DATABASE ERROR:
{sql_error}

Return STRICT JSON only:
{{
  "sql": "corrected SELECT ...",
  "sql_explanation": "brief explanation of the correction"
}}
""".strip()


SINGLE_AGENT_FORECAST_PROMPT = r"""
You are the SAME ForecastBot SINGLE forecasting agent.

Earlier in this case you interpreted the user's request and generated SQL.
Now forecast the supplied historical time series.

FORECAST REQUEST:
{forecast_request}

HISTORY BRANCH:
{branch_label}

TIME UNIT:
{time_unit_label}

FORECAST HORIZON:
{horizon}

NUMBER OF HISTORY POINTS:
{history_length}

HISTORY VALUES IN CHRONOLOGICAL ORDER:
{history_json}

Rules:
1. Forecast exactly {horizon} future values.
2. Use ONLY the history supplied above.
3. Do not assume access to held-out actual/test values.
4. All values must be finite numbers.
5. Demand/passenger forecasts must not be negative.
6. Return STRICT JSON only, with no Markdown or extra text.

Output:
{{
  "forecast": [0.0, 0.0]
}}

The forecast array MUST contain exactly {horizon} values.
""".strip()


SINGLE_AGENT_EXPLANATION_PROMPT = r"""
You are the SAME ForecastBot SINGLE forecasting agent.

Earlier in this case you interpreted the request, generated SQL and produced
the forecast. Explain the final result using ONLY the information below.

FORECAST REQUEST:
{forecast_request}

HISTORICAL DATA SUMMARY:
{historical_data_summary}

FINAL FORECAST:
{forecast_result}

Requirements:
1. Do not claim knowledge of actual future/test values.
2. Do not invent causes or external events not present in the inputs.
3. Keep the explanation concise, clear and useful.
4. Return STRICT JSON only.

Output:
{{
  "title": "Kết quả dự báo",
  "summary_text": "...",
  "insights": ["...", "..."],
  "chart_note": "..."
}}
""".strip()


# ---------------------------------------------------------------------
# Shared deterministic helpers (kept equivalent to multi-agent runner)
# ---------------------------------------------------------------------

def js(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def clean_df(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty or not {"ds", "y"}.issubset(df.columns):
        return []
    d = df[["ds", "y"]].copy()
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna().sort_values("ds")
    d["ds"] = d["ds"].dt.strftime("%Y-%m-%d")
    return d.to_dict("records")


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n_points": 0}
    ys = [float(x["y"]) for x in rows]
    return {
        "n_points": len(rows),
        "start_date": rows[0]["ds"],
        "end_date": rows[-1]["ds"],
        "min_y": min(ys),
        "max_y": max(ys),
        "mean_y": sum(ys) / len(ys),
    }


def windows(g: str, h: int, n: int) -> tuple[int, int]:
    if g == "day":
        a, b = 180, 365
    elif g == "month":
        (a, b) = (24, 36) if h <= 6 else (36, 60)
    elif g == "quarter":
        (a, b) = (8, 12) if h <= 4 else (12, 20)
    elif g == "year":
        (a, b) = (5, 10) if h <= 3 else (10, 20)
    else:
        raise ValueError(g)
    return min(a, n), min(b, n)


def future_dates(last: str, h: int, g: str) -> list[str]:
    x = pd.to_datetime(last)
    if g == "day":
        idx = pd.date_range(x + pd.Timedelta(days=1), periods=h, freq="D")
    elif g == "month":
        idx = pd.date_range(x + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    elif g == "quarter":
        idx = pd.date_range(
            x + pd.offsets.QuarterBegin(startingMonth=1), periods=h, freq="QS"
        )
    else:
        idx = pd.date_range(x + pd.offsets.YearBegin(1), periods=h, freq="YS")
    return [z.strftime("%Y-%m-%d") for z in idx]


def is_raw_json_compliant(content: Any) -> bool:
    if not isinstance(content, str) or not content.strip():
        return False
    try:
        json.loads(content.strip())
        return True
    except Exception:
        return False


def token_trace(
    stage,
    provider,
    prompt,
    r,
    attempt=1,
    status="success",
    error="",
    parse_ok=None,
    raw_json_compliant=None,
):
    return {
        "stage": stage,
        "type": "single_agent",
        "provider": provider.provider,
        "model": provider.model,
        "attempt": attempt,
        "prompt": prompt,
        "raw_response": getattr(r, "content", ""),
        "input_tokens": getattr(r, "input_tokens", None),
        "output_tokens": getattr(r, "output_tokens", None),
        "total_tokens": getattr(r, "total_tokens", None),
        "llm_time_s": getattr(r, "duration_seconds", None),
        "status": status,
        "parse_ok": parse_ok,
        "raw_json_compliant": raw_json_compliant,
        "error": error,
    }


def call_json(provider, prompt, stage, traces, max_attempts=2):
    last = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        r = None
        try:
            r = provider.invoke(prompt)
            last = r.content
            parsed = extract_json(r.content)
            traces.append(
                token_trace(
                    stage,
                    provider,
                    prompt,
                    r,
                    attempt,
                    status="success",
                    parse_ok=True,
                    raw_json_compliant=is_raw_json_compliant(r.content),
                )
            )
            return parsed
        except Exception as e:
            last_error = str(e)
            if r is not None:
                traces.append(
                    token_trace(
                        stage,
                        provider,
                        prompt,
                        r,
                        attempt,
                        status="error",
                        error=last_error,
                        parse_ok=False,
                        raw_json_compliant=is_raw_json_compliant(
                            getattr(r, "content", "")
                        ),
                    )
                )
            else:
                traces.append(
                    {
                        "stage": stage,
                        "type": "single_agent",
                        "provider": provider.provider,
                        "model": provider.model,
                        "attempt": attempt,
                        "prompt": prompt,
                        "raw_response": last,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "llm_time_s": None,
                        "status": "error",
                        "parse_ok": False,
                        "raw_json_compliant": False,
                        "error": last_error,
                    }
                )
    raise RuntimeError(
        f"{stage} failed after {max_attempts} total attempts: {last_error}"
    )


def validate_request(x: dict) -> dict:
    d = x.get("dataset")
    g = x.get("granularity")
    h = x.get("horizon")
    raw_filters = x.get("filters", {})

    if isinstance(raw_filters, dict):
        f = raw_filters.copy()
    elif isinstance(raw_filters, list):
        f = {}
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            if "field" in item and "value" in item:
                field = item.get("field")
                value = item.get("value")
                if field:
                    f[str(field)] = value
            else:
                for key, value in item.items():
                    f[str(key)] = value
    else:
        f = {}

    if d not in VALID_FILTERS:
        d = None
    if g not in ALLOWED:
        g = None

    try:
        h = int(h) if h is not None else None
    except (TypeError, ValueError):
        h = None

    # Same behavior as current multi-agent runner.
    if g and h is not None and h not in ALLOWED[g]:
        h = min(ALLOWED[g], key=lambda z: abs(z - h))

    f = {
        k: v
        for k, v in f.items()
        if d and k in VALID_FILTERS[d] and v not in (None, "")
    }

    missing = [
        k
        for k, v in [("dataset", d), ("granularity", g), ("horizon", h)]
        if v is None
    ]

    return {
        "intent": "forecast_request",
        "dataset": d,
        "granularity": g,
        "horizon": h,
        "filters": f,
        "missing_fields": missing,
        "needs_clarification": bool(missing),
        "user_message_for_clarification": (
            x.get("user_message_for_clarification", "") if missing else ""
        ),
    }


def schema_context(rows, dataset):
    norm = []
    for r in rows:
        try:
            norm.append(dict(r))
        except Exception:
            pass

    ds = [r for r in norm if r.get("dataset_code") == dataset]
    facts = [r for r in ds if r.get("table_type") == "fact"]
    src = facts or ds
    ctx = next(
        (r.get("llm_schema_context", "") for r in src if r.get("llm_schema_context")),
        "",
    )
    rules = next(
        (
            r.get("sql_generation_rules", "")
            for r in src
            if r.get("sql_generation_rules")
        ),
        "",
    )
    return ctx, rules


def parse_fc(x, h):
    a = x.get("forecast") if isinstance(x, dict) else None
    if not isinstance(a, list) or len(a) != h:
        return None
    try:
        vals = [max(0.0, float(v)) for v in a]
        return vals if all(math.isfinite(v) for v in vals) else None
    except Exception:
        return None


def fallback_vals(rows, h):
    ys = [float(x["y"]) for x in rows[-7:]]
    v = max(0.0, sum(ys) / len(ys)) if ys else 0.0
    return [v] * h


def single_agent_forecast(
    provider,
    req,
    rows,
    traces,
    prefix="single_agent",
    target_dates=None,
):
    """
    Same history-window and ensemble policy as the multi-agent runner.

    The same LLM provider/model and the same Single-Agent role are reused for
    short and long branches. Python performs the deterministic 0.6/0.4 ensemble.
    """
    h = int(req["horizon"])
    g = req["granularity"]
    sn, ln = windows(g, h, len(rows))
    short = rows[-sn:] if sn else []
    long = rows[-ln:] if ln else []

    if not rows:
        return {
            "forecast": [],
            "status": "failed",
            "fallback_used": True,
            "window_info": {},
        }

    unit = {
        "day": "DAILY",
        "month": "MONTHLY",
        "quarter": "QUARTERLY",
        "year": "YEARLY",
    }[g]

    def branch(label, data):
        prompt = SINGLE_AGENT_FORECAST_PROMPT.format(
            forecast_request=js(req),
            branch_label=label,
            time_unit_label=unit,
            horizon=h,
            history_length=len(data),
            history_json=js([float(x["y"]) for x in data]),
        )
        try:
            val = parse_fc(
                call_json(
                    provider,
                    prompt,
                    f"{prefix}.forecast_{label}",
                    traces,
                    2,
                ),
                h,
            )
        except Exception:
            val = None
        return val

    sv = branch("short", short)

    # Same optimization as the existing runner:
    # if both effective windows collapse to the same history, call LLM once.
    if sn == ln:
        lv = sv
    else:
        lv = branch("long", long)

    fallback = False
    mode = "ensemble"

    if sv is not None and lv is not None:
        if sn != ln:
            vals = [round(0.6 * sv[i] + 0.4 * lv[i], 4) for i in range(h)]
        else:
            vals = [round(v, 4) for v in sv]
            mode = "single"
    elif sv is not None:
        vals = sv
        mode = "short_only"
        fallback = True
    elif lv is not None:
        vals = lv
        mode = "long_only"
        fallback = True
    else:
        vals = fallback_vals(rows, h)
        mode = "deterministic_recent_mean"
        fallback = True

    dates = (
        list(target_dates)
        if target_dates is not None
        else future_dates(rows[-1]["ds"], h, g)
    )

    if len(dates) != h:
        raise ValueError(
            f"target_dates length={len(dates)} does not match horizon={h}"
        )

    return {
        "forecast": [
            {"step": i + 1, "ds": dates[i], "yhat": round(vals[i], 4)}
            for i in range(h)
        ],
        "status": "success",
        "fallback_used": fallback,
        "selected_mode": mode,
        "weights": {"short": 0.6, "long": 0.4},
        "window_info": {
            "short_n": sn,
            "long_n": ln,
            "short_window_name": f"h{sn}",
            "long_window_name": f"h{ln}",
        },
    }


def load_cases(dataset):
    path = SET_DIR / (
        "m5_63_queries.csv" if dataset == "M5" else "hcmcbus_63_queries.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(
            f"{path} is not a readable CSV. Replace it with the real CSV export. "
            f"Original error: {e}"
        )

    idcol = next(
        (c for c in ["item", "STT", "ID", "Case_ID", "case_id"] if c in df.columns),
        None,
    )
    qcol = next(
        (
            c
            for c in [
                "user_request",
                "User_request",
                "Câu hỏi đầu vào",
                "Câu hỏi",
                "Question",
                "query",
                "Query",
            ]
            if c in df.columns
        ),
        None,
    )

    if not qcol:
        raise ValueError(f"Cannot find query column. Columns={list(df.columns)}")

    if not idcol:
        df["Case_ID"] = [f"{dataset}_{i+1:02d}" for i in range(len(df))]
        idcol = "Case_ID"

    return [
        (str(r[idcol]).strip(), str(r[qcol]).strip())
        for _, r in df.iterrows()
        if str(r[qcol]).strip()
    ]


def save_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def create_forecast_chart(
    history_path: Path,
    forecast_path: Path,
    test_path: Path,
    output_path: Path,
    granularity: str,
):
    import matplotlib.pyplot as plt

    history_obj = json.loads(history_path.read_text(encoding="utf-8"))
    forecast_obj = json.loads(forecast_path.read_text(encoding="utf-8"))
    test_obj = (
        json.loads(test_path.read_text(encoding="utf-8"))
        if test_path.exists()
        else {}
    )

    history = history_obj.get("history", [])
    fc = forecast_obj.get("forecast", [])
    actual = test_obj.get("test", []) if test_obj.get("evaluation_eligible") else []

    if not history or not fc:
        raise ValueError("history.json or forecast.json has no plottable data")

    display_n = {
        "day": 180,
        "month": 24,
        "quarter": 12,
        "year": 10,
    }.get(granularity, len(history))

    hist = history[-min(display_n, len(history)) :]
    hx = pd.to_datetime([x["ds"] for x in hist])
    hy = [float(x["y"]) for x in hist]
    fx = pd.to_datetime([x["ds"] for x in fc])
    fy = [float(x["yhat"]) for x in fc]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(hx, hy, label="History (train)")
    ax.plot([hx[-1], *fx], [hy[-1], *fy], marker="o", label="Forecast")

    if actual:
        axx = pd.to_datetime([x["ds"] for x in actual])
        axy = [float(x["y"]) for x in actual]
        ax.plot(axx, axy, marker="o", linestyle="--", label="Actual (test)")

    ax.axvline(hx[-1], linestyle="--", linewidth=1, label="Forecast cutoff")
    ax.set_title(
        "Historical data, forecast and actual"
        if actual
        else "Historical data and forecast"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Single-Agent case execution
# ---------------------------------------------------------------------

def run_case(dataset, model_key, cid, question):
    provider_name, model = MODEL_MAP[model_key]
    provider = LLMProvider(provider_name, model)

    traces = []
    timings = {}
    t_all = time.perf_counter()

    result = {
        "architecture": "single_agent",
        "case_id": cid,
        "question": question,
        "requested_dataset": dataset,
        "provider": provider_name,
        "model": model,
    }

    try:
        # -------------------------------------------------------------
        # 1) Load schema FIRST.
        # Single Agent needs schema to jointly understand request + SQL.
        # -------------------------------------------------------------
        t = time.perf_counter()
        schema = SchemaKnowledgeRepository().get_by_dataset(dataset)
        timings["load_schema"] = time.perf_counter() - t
        ctx, rules = schema_context(schema, dataset)

        # -------------------------------------------------------------
        # 2) SAME Single Agent: request understanding + SQL generation.
        # This replaces separate Router Agent + SQL Agent.
        # -------------------------------------------------------------
        t = time.perf_counter()
        p1 = SINGLE_AGENT_REQUEST_SQL_PROMPT.format(
            user_query=question,
            expected_dataset=dataset,
            llm_schema_context=ctx,
            sql_generation_rules=rules,
        )
        plan = call_json(
            provider,
            p1,
            "single_agent.request_sql",
            traces,
            2,
        )
        timings["request_sql"] = time.perf_counter() - t

        req = validate_request(plan)
        result["forecast_request"] = req
        result["single_agent_plan"] = plan

        if req["needs_clarification"]:
            raise ValueError(
                "needs_clarification: " + ",".join(req["missing_fields"])
            )

        if req["dataset"] != dataset:
            raise ValueError(
                f"Single Agent dataset={req['dataset']}, expected={dataset}"
            )

        sql = str(plan.get("sql", "") or "").strip()
        if not sql:
            raise ValueError("Single Agent did not generate SQL")

        result["generated_sql"] = sql
        result["sql_explanation"] = plan.get("sql_explanation", "")

        # -------------------------------------------------------------
        # 3) Python executes SQL + bounded repair.
        # Repair is performed by the SAME Single-Agent role.
        # -------------------------------------------------------------
        t = time.perf_counter()
        sql_error = ""
        records = []

        for repair in range(0, 3):
            try:
                df = fetch_dataframe(sql, dataset_code=dataset)
                records = clean_df(df)
                break
            except Exception as e:
                sql_error = str(e)
                if repair >= 2:
                    raise

                rp = SINGLE_AGENT_SQL_REPAIR_PROMPT.format(
                    forecast_request=js(req),
                    llm_schema_context=ctx,
                    sql_generation_rules=rules,
                    previous_sql=sql,
                    sql_error=sql_error,
                )
                rr = call_json(
                    provider,
                    rp,
                    "single_agent.sql_repair",
                    traces,
                    2,
                )
                sql = str(rr.get("sql", "") or "").strip()
                result["generated_sql"] = sql
                result["sql_explanation"] = rr.get(
                    "sql_explanation",
                    result.get("sql_explanation", ""),
                )

        timings["sql_execution_and_repair"] = time.perf_counter() - t

        if not records:
            raise ValueError("SQL succeeded but returned no historical data")

        result["sql_result"] = records
        result["historical_data_summary"] = summary(records)
        result["historical_n"] = len(records)

        # -------------------------------------------------------------
        # 4) SAME holdout eligibility/split as multi-agent runner.
        # -------------------------------------------------------------
        h = int(req["horizon"])
        g = req["granularity"]
        n_total = len(records)

        if g == "year":
            required_points = math.ceil(1.5 * h)
            split_rule = "n_total >= ceil(1.5 * horizon)"
        else:
            required_points = 2 * h
            split_rule = "n_total >= 2 * horizon"

        min_train_required = required_points - h

        if n_total >= required_points:
            forecast_history = records[:-h]
            ground_truth_rows = records[-h:]

            test_artifact = {
                "status": "available",
                "evaluation_eligible": True,
                "reason": None,
                "horizon": h,
                "n_test": len(ground_truth_rows),
                "test": ground_truth_rows,
            }

            split_info = {
                "split_mode": "holdout",
                "evaluation_eligible": True,
                "evaluation_skip_reason": None,
                "n_total": n_total,
                "n_train": len(forecast_history),
                "n_test": len(ground_truth_rows),
                "horizon": h,
                "minimum_train_required": min_train_required,
                "required_points_for_evaluation": required_points,
                "split_rule": split_rule,
            }

            # Forecast timestamps exactly match held-out GT timestamps.
            target_dates = [x["ds"] for x in ground_truth_rows]

        else:
            forecast_history = records
            ground_truth_rows = []

            test_artifact = {
                "status": "not_available",
                "evaluation_eligible": False,
                "reason": "insufficient_history_for_holdout",
                "horizon": h,
                "available_points": n_total,
                "minimum_train_required": min_train_required,
                "required_points_for_evaluation": required_points,
                "n_test": 0,
                "test": [],
            }

            split_info = {
                "split_mode": "full_history_forecast",
                "evaluation_eligible": False,
                "evaluation_skip_reason": "insufficient_history_for_holdout",
                "n_total": n_total,
                "n_train": n_total,
                "n_test": 0,
                "horizon": h,
                "minimum_train_required": min_train_required,
                "required_points_for_evaluation": required_points,
                "split_rule": split_rule,
            }

            target_dates = None

        result["forecast_history"] = forecast_history
        result["test_artifact"] = test_artifact
        result["split_info"] = split_info

        # -------------------------------------------------------------
        # 5) SAME Single Agent forecasts using SAME adaptive windows.
        # Python still performs deterministic ensemble.
        # -------------------------------------------------------------
        t = time.perf_counter()
        fc = single_agent_forecast(
            provider,
            req,
            forecast_history,
            traces,
            prefix="single_agent",
            target_dates=target_dates,
        )
        timings["forecast"] = time.perf_counter() - t
        result["forecast_result"] = fc

        # -------------------------------------------------------------
        # 6) SAME Single Agent explains final forecast.
        # It sees n_train summary + forecast only; NEVER held-out test.
        # -------------------------------------------------------------
        t = time.perf_counter()
        ep = SINGLE_AGENT_EXPLANATION_PROMPT.format(
            forecast_request=js(req),
            historical_data_summary=js(summary(forecast_history)),
            forecast_result=js(fc),
        )

        try:
            expl = call_json(
                provider,
                ep,
                "single_agent.explanation",
                traces,
                2,
            )
        except Exception:
            expl = {
                "title": "Kết quả dự báo",
                "summary_text": (
                    "Hệ thống đã tạo dự báo. "
                    "Phần diễn giải LLM không khả dụng."
                ),
                "insights": [],
                "chart_note": (
                    "Hiển thị dữ liệu lịch sử dùng cho dự báo "
                    "và kết quả dự báo."
                ),
            }
            result["explanation_fallback_used"] = True

        timings["explanation"] = time.perf_counter() - t
        result["explanation_result"] = expl
        result["status"] = "success"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    timings["end_to_end"] = time.perf_counter() - t_all
    result["timings"] = timings
    result["traces"] = traces

    toks = [
        x.get("total_tokens")
        for x in traces
        if isinstance(x.get("total_tokens"), int)
    ]
    result["total_tokens"] = sum(toks) if toks else None

    return result


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

def write_case(out, result):
    c = out / "cases" / result["case_id"]
    c.mkdir(parents=True, exist_ok=True)

    save_json(c / "request.json", result.get("forecast_request", {}))
    save_json(c / "single_agent_plan.json", result.get("single_agent_plan", {}))

    (c / "generated.sql").write_text(
        result.get("generated_sql", ""),
        encoding="utf-8",
    )

    sql_rows = result.get("sql_result", [])
    if sql_rows:
        pd.DataFrame(sql_rows).to_csv(
            c / "sql_result.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_json(
        c / "history.json",
        {
            "summary": summary(result.get("forecast_history", [])),
            "history": result.get("forecast_history", []),
        },
    )
    save_json(c / "test.json", result.get("test_artifact", {}))
    save_json(c / "forecast.json", result.get("forecast_result", {}))

    try:
        granularity = result.get("forecast_request", {}).get("granularity")
        create_forecast_chart(
            c / "history.json",
            c / "forecast.json",
            c / "test.json",
            c / "forecast_chart.png",
            granularity,
        )
        result["chart_status"] = "success"
        result["chart_error"] = ""
    except Exception as e:
        result["chart_status"] = "failed"
        result["chart_error"] = str(e)

    save_json(c / "explanation.json", result.get("explanation_result", {}))
    save_json(c / "trace.json", result.get("traces", []))

    save_json(
        c / "metadata.json",
        {
            "architecture": "single_agent",
            "case_id": result.get("case_id"),
            "question": result.get("question"),
            "requested_dataset": result.get("requested_dataset"),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "status": result.get("status"),
            "split_info": result.get("split_info", {}),
            "timings": result.get("timings", {}),
            "total_tokens": result.get("total_tokens"),
            "chart_status": result.get("chart_status"),
            "chart_error": result.get("chart_error", ""),
            "error": result.get("error", ""),
        },
    )

    save_json(c / "result.json", result)

    if result.get("status") != "success":
        save_json(
            out / "errors" / f"{result['case_id']}.json",
            {
                "error": result.get("error"),
                "traceback": result.get("traceback"),
            },
        )


def summary_row(r):
    q = r.get("forecast_request", {})
    f = r.get("forecast_result", {})
    t = r.get("timings", {})
    s = r.get("split_info", {})
    traces = r.get("traces", [])

    retries = sum(
        1
        for x in traces
        if isinstance(x, dict) and int(x.get("attempt", 1) or 1) > 1
    )

    successful_traces = [
        x
        for x in traces
        if isinstance(x, dict)
        and x.get("type") == "single_agent"
        and x.get("status") == "success"
    ]

    parse_ok_all = (
        all(x.get("parse_ok") is True for x in successful_traces)
        if successful_traces
        else None
    )

    raw_json_compliant_all = (
        all(x.get("raw_json_compliant") is True for x in successful_traces)
        if successful_traces
        else None
    )

    return {
        "architecture": "single_agent",
        "case_id": r["case_id"],
        "dataset": q.get("dataset"),
        "model": r.get("model"),
        "provider": r.get("provider"),
        "granularity": q.get("granularity"),
        "horizon": q.get("horizon"),
        "filters": js(q.get("filters", {})),
        "n_total": s.get("n_total"),
        "n_train": s.get("n_train"),
        "n_test": s.get("n_test"),
        "minimum_train_required": s.get("minimum_train_required"),
        "required_points_for_evaluation": s.get(
            "required_points_for_evaluation"
        ),
        "split_rule": s.get("split_rule"),
        "split_mode": s.get("split_mode"),
        "evaluation_eligible": s.get("evaluation_eligible"),
        "evaluation_skip_reason": s.get("evaluation_skip_reason"),
        "status": r.get("status"),
        "forecast_valid": len(f.get("forecast", [])) == q.get("horizon"),
        "forecast_mode": f.get("selected_mode"),
        "short_window_n": f.get("window_info", {}).get("short_n"),
        "long_window_n": f.get("window_info", {}).get("long_n"),
        "fallback_used": f.get("fallback_used"),
        "chart_status": r.get("chart_status"),
        "parse_ok_all": parse_ok_all,
        "raw_json_compliant_all": raw_json_compliant_all,
        "retry_count": retries,
        "llm_call_count": len(successful_traces),
        "total_tokens": r.get("total_tokens"),
        "request_sql_time_s": t.get("request_sql"),
        "sql_execution_repair_time_s": t.get("sql_execution_and_repair"),
        "forecast_time_s": t.get("forecast"),
        "explanation_time_s": t.get("explanation"),
        "end_to_end_time_s": t.get("end_to_end"),
        "error": r.get("error", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["M5", "HCMC_BUS"],
        required=True,
    )
    ap.add_argument(
        "--model",
        choices=["gemini", "gemma"],
        default="gemini",
    )
    ap.add_argument(
        "--mode",
        choices=["case", "full"],
        default="case",
    )
    ap.add_argument("--case-id")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.dataset)

    if args.mode == "case":
        if args.case_id:
            cases = [x for x in cases if x[0] == args.case_id]
        else:
            cases = cases[:1]

        if not cases:
            raise SystemExit("Case not found")

    slug = "gemini_2_5_pro" if args.model == "gemini" else "gemma4"

    # Separate output tree: never overwrite current Multi-Agent results.
    out = (
        ROOT
        / "outputs"
        / "single_agent_ablation"
        / ("m5_exp" if args.dataset == "M5" else "hcmcbus_exp")
        / slug
    )
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_path = out / "summary.csv"

    if args.resume and summary_path.exists():
        old = pd.read_csv(summary_path)
        rows = old.to_dict("records")
        done = {
            str(x["case_id"])
            for x in rows
            if x.get("status") == "success"
        }
        cases = [x for x in cases if x[0] not in done]

    for i, (cid, q) in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] {cid} | {args.dataset} | "
            f"{args.model} | SINGLE-AGENT"
        )

        r = run_case(
            args.dataset,
            args.model,
            cid,
            q,
        )

        write_case(out, r)

        rows = [
            x for x in rows if str(x.get("case_id")) != cid
        ] + [summary_row(r)]

        pd.DataFrame(rows).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            " ->",
            r["status"],
            "E2E=",
            round(r["timings"]["end_to_end"], 2),
            "s",
        )

    print("Output:", out)


if __name__ == "__main__":
    main()
