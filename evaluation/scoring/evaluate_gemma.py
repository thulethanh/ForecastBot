from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_ROOT = PROJECT_ROOT / "evaluation" / "ground_truth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

RESULT_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "gemma"
)

CONFIGS = {
    "M5": {
        "prefix": "m5",
        "case_prefix": "M",
        "gt_base": GT_ROOT / "m5",
        "cases_dir": (
            OUTPUT_ROOT
            / "m5_exp"
            / "gemma4"
            / "cases"
        ),
        # Hỗ trợ cả hai khả năng bố trí errors
        "error_dirs": [
            OUTPUT_ROOT / "m5_exp" / "errors",
            OUTPUT_ROOT
            / "m5_exp"
            / "gemma4"
            / "errors",
        ],
    },

    "HCMC_BUS": {
        "prefix": "hcmcbus",
        "case_prefix": "H",
        "gt_base": GT_ROOT / "hcmcbus",
        "cases_dir": (
            OUTPUT_ROOT
            / "hcmcbus_exp"
            / "gemma4"
            / "cases"
        ),
        "error_dirs": [
            OUTPUT_ROOT / "hcmcbus_exp" / "errors",
            OUTPUT_ROOT
            / "hcmcbus_exp"
            / "gemma4"
            / "errors",
        ],
    },
}

MODEL_KEY = "gemma"
MODEL_NAME = "Gemma 4 E4B"


# ============================================================
# JSON / BASIC HELPERS
# ============================================================

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json(path: Path) -> Any | None:
    try:
        if path.exists():
            return read_json(path)
    except Exception:
        pass
    return None


def clean_value(value):
    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

        if value.lower() == "all":
            return None

    return value


def normalize_filters(filters: Any) -> dict:
    """
    None / "" / All = không áp dụng filter.

    Ví dụ:
    {
        "item_id": "FOODS_3_228",
        "store_id": None,
        "state_id": None
    }

    được chuẩn hóa thành:

    {
        "item_id": "FOODS_3_228"
    }
    """
    if not isinstance(filters, dict):
        return {}

    result = {}

    for key, value in filters.items():
        value = clean_value(value)

        if value is None:
            continue

        result[str(key)] = value

    return result


def normalize_request(obj: dict | None) -> dict:
    obj = obj or {}

    return {
        "dataset": clean_value(
            obj.get("dataset")
        ),
        "granularity": clean_value(
            obj.get("granularity")
        ),
        "horizon": obj.get("horizon"),
        "filters": normalize_filters(
            obj.get("filters")
        ),
    }


# ============================================================
# ERROR DISCOVERY
# ============================================================

CASE_RE = re.compile(r"\b([MH]\d{2})\b", re.I)


def extract_case_id_from_path(path: Path) -> str | None:
    """
    Thử lấy M01 / H01 từ filename hoặc path.
    """
    text = str(path)

    match = CASE_RE.search(text)

    if match:
        return match.group(1).upper()

    return None


def discover_errors(
    dataset: str,
    cfg: dict,
) -> dict[str, dict]:

    errors: dict[str, dict] = {}

    for error_dir in cfg["error_dirs"]:

        if not error_dir.exists():
            continue

        for path in error_dir.rglob("*"):

            if not path.is_file():
                continue

            case_id = extract_case_id_from_path(
                path
            )

            # Nếu tên file không chứa case ID,
            # thử đọc JSON.
            obj = safe_json(path)

            if (
                case_id is None
                and isinstance(obj, dict)
            ):
                candidate = (
                    obj.get("case_id")
                    or obj.get("item")
                    or obj.get("id")
                )

                if candidate:
                    candidate = str(
                        candidate
                    ).upper()

                    if CASE_RE.fullmatch(
                        candidate
                    ):
                        case_id = candidate

            if case_id is None:
                continue

            # Tránh nhầm H và M.
            required_prefix = cfg[
                "case_prefix"
            ]

            if not case_id.startswith(
                required_prefix
            ):
                continue

            error_message = None

            if isinstance(obj, dict):
                error_message = (
                    obj.get("error")
                    or obj.get(
                        "message"
                    )
                    or obj.get(
                        "exception"
                    )
                )

            errors[case_id] = {
                "error_file": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "error_message": (
                    str(error_message)
                    if error_message
                    else ""
                ),
            }

    return errors


# ============================================================
# TIME-SERIES HELPERS
# ============================================================

def normalize_ts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if not {"ds", "y"}.issubset(
        df.columns
    ):
        raise ValueError(
            "Expected columns ds,y; got "
            f"{df.columns.tolist()}"
        )

    out = df[["ds", "y"]].copy()

    out["ds"] = pd.to_datetime(
        out["ds"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    out["y"] = pd.to_numeric(
        out["y"],
        errors="raise",
    ).astype(float)

    return (
        out.sort_values("ds")
        .reset_index(drop=True)
    )


def compare_ts(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
) -> dict:

    gt = normalize_ts(gt)
    pred = normalize_ts(pred)

    n_gt = len(gt)
    n_pred = len(pred)

    row_count_match = (
        n_gt == n_pred
    )

    if not row_count_match:
        return {
            "gt_rows": n_gt,
            "pred_rows": n_pred,
            "row_count_match": False,
            "date_match": False,
            "value_match": False,
            "exact_match": False,
        }

    date_match = (
        gt["ds"].tolist()
        == pred["ds"].tolist()
    )

    value_match = bool(
        np.allclose(
            gt["y"].to_numpy(float),
            pred["y"].to_numpy(float),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )
    )

    return {
        "gt_rows": n_gt,
        "pred_rows": n_pred,
        "row_count_match": True,
        "date_match": bool(
            date_match
        ),
        "value_match": value_match,
        "exact_match": bool(
            date_match
            and value_match
        ),
    }


# ============================================================
# ORCHESTRATION
# ============================================================

def evaluate_orchestration(
    case_id: str,
    dataset: str,
    gt_path: Path,
    output_path: Path,
    system_error: bool,
) -> dict:

    base = {
        "case_id": case_id,
        "dataset": dataset,
        "model": MODEL_NAME,
        "system_error": system_error,
    }

    gt_obj = safe_json(gt_path)

    if gt_obj is None:
        return {
            **base,
            "status": (
                "missing_ground_truth"
            ),
            "overall_exact_match": False,
        }

    pred_obj = safe_json(output_path)

    if pred_obj is None:
        return {
            **base,
            "status": (
                "missing_request_output"
            ),
            "dataset_match": False,
            "granularity_match": False,
            "horizon_match": False,
            "filters_match": False,
            "overall_exact_match": False,
        }

    gt = normalize_request(gt_obj)
    pred = normalize_request(pred_obj)

    dataset_match = (
        gt["dataset"]
        == pred["dataset"]
    )

    granularity_match = (
        gt["granularity"]
        == pred["granularity"]
    )

    horizon_match = (
        gt["horizon"]
        == pred["horizon"]
    )

    filters_match = (
        gt["filters"]
        == pred["filters"]
    )

    overall = (
        dataset_match
        and granularity_match
        and horizon_match
        and filters_match
    )

    return {
        **base,
        "status": "success",
        "gt_dataset": gt["dataset"],
        "pred_dataset": pred["dataset"],
        "gt_granularity": (
            gt["granularity"]
        ),
        "pred_granularity": (
            pred["granularity"]
        ),
        "gt_horizon": gt["horizon"],
        "pred_horizon": (
            pred["horizon"]
        ),
        "gt_filters": json.dumps(
            gt["filters"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "pred_filters": json.dumps(
            pred["filters"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "dataset_match": (
            dataset_match
        ),
        "granularity_match": (
            granularity_match
        ),
        "horizon_match": (
            horizon_match
        ),
        "filters_match": (
            filters_match
        ),
        "overall_exact_match": (
            overall
        ),
    }


# ============================================================
# SQL
# ============================================================

def evaluate_sql(
    case_id: str,
    dataset: str,
    gt_history_path: Path,
    sql_result_path: Path,
    system_error: bool,
) -> dict:

    base = {
        "case_id": case_id,
        "dataset": dataset,
        "model": MODEL_NAME,
        "system_error": system_error,
    }

    if not gt_history_path.exists():
        return {
            **base,
            "status": (
                "missing_ground_truth"
            ),
            "sql_result_available": False,
            "execution_accuracy": False,
        }

    # Nếu pipeline không tạo sql_result:
    # Đây là failure đối với SQL ở góc nhìn
    # end-to-end.
    if not sql_result_path.exists():
        return {
            **base,
            "status": (
                "missing_sql_result"
            ),
            "sql_result_available": False,
            "execution_accuracy": False,
        }

    try:
        gt = pd.read_csv(
            gt_history_path
        )

        pred = pd.read_csv(
            sql_result_path
        )

        cmp = compare_ts(
            gt,
            pred,
        )

        return {
            **base,
            "status": "success",
            "sql_result_available": True,
            **cmp,
            "execution_accuracy": (
                cmp["exact_match"]
            ),
        }

    except Exception as e:
        return {
            **base,
            "status": "comparison_error",
            "sql_result_available": True,
            "execution_accuracy": False,
            "error": str(e),
        }


# ============================================================
# HISTORY INTEGRITY
# ============================================================

def history_json_to_df(
    path: Path,
) -> pd.DataFrame:

    obj = read_json(path)

    history = obj.get(
        "history",
        []
    )

    return pd.DataFrame(history)


def evaluate_integrity(
    case_id: str,
    dataset: str,
    gt_history_path: Path,
    gt_train_path: Path,
    gt_test_path: Path,
    output_history_path: Path,
    system_error: bool,
) -> dict:

    evaluation_eligible = (
        gt_test_path.exists()
    )

    base = {
        "case_id": case_id,
        "dataset": dataset,
        "model": MODEL_NAME,
        "system_error": system_error,
        "evaluation_eligible": (
            evaluation_eligible
        ),
    }

    expected_path = (
        gt_train_path
        if evaluation_eligible
        else gt_history_path
    )

    if not output_history_path.exists():
        return {
            **base,
            "status": (
                "missing_history_output"
            ),
            "history_match": False,
            "ground_truth_leakage": (
                np.nan
            ),
        }

    if not expected_path.exists():
        return {
            **base,
            "status": (
                "missing_ground_truth"
            ),
            "history_match": False,
            "ground_truth_leakage": (
                np.nan
            ),
        }

    try:
        expected = pd.read_csv(
            expected_path
        )

        actual = history_json_to_df(
            output_history_path
        )

        cmp = compare_ts(
            expected,
            actual,
        )

        leakage = False

        if evaluation_eligible:

            test = read_json(
                gt_test_path
            )

            test_dates = {
                str(row["ds"])
                for row in test
            }

            actual_norm = normalize_ts(
                actual
            )

            history_dates = set(
                actual_norm[
                    "ds"
                ].tolist()
            )

            leakage = bool(
                test_dates
                & history_dates
            )

        return {
            **base,
            "status": "success",
            **cmp,
            "history_match": (
                cmp["exact_match"]
            ),
            "ground_truth_leakage": (
                leakage
            ),
        }

    except Exception as e:
        return {
            **base,
            "status": "comparison_error",
            "history_match": False,
            "ground_truth_leakage": (
                np.nan
            ),
            "error": str(e),
        }


# ============================================================
# FORECAST
# ============================================================

def load_test(
    path: Path,
) -> pd.DataFrame:

    obj = read_json(path)

    if isinstance(obj, list):
        records = obj

    elif isinstance(obj, dict):
        records = obj.get(
            "test",
            obj.get(
                "actual",
                []
            ),
        )

    else:
        records = []

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(
            columns=["ds", "y"]
        )

    return df[["ds", "y"]]


def load_forecast(
    path: Path,
) -> tuple[pd.DataFrame, dict]:

    obj = read_json(path)

    records = obj.get(
        "forecast",
        []
    )

    df = pd.DataFrame(records)

    if df.empty:
        return (
            pd.DataFrame(
                columns=["ds", "y"]
            ),
            obj,
        )

    if "yhat" not in df.columns:
        raise ValueError(
            "forecast.json does not "
            "contain yhat"
        )

    df = df.rename(
        columns={
            "yhat": "y"
        }
    )

    return (
        df[["ds", "y"]],
        obj,
    )


def mae_rmse(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> tuple[float, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    errors = (
        actual - forecast
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    return (
        float(mae),
        float(rmse),
    )


def evaluate_forecast(
    case_id: str,
    dataset: str,
    gt_request_path: Path,
    gt_test_path: Path,
    forecast_path: Path,
    metadata_path: Path,
    system_error: bool,
    error_info: dict | None,
) -> dict:

    gt_request = (
        safe_json(
            gt_request_path
        )
        or {}
    )

    granularity = gt_request.get(
        "granularity"
    )

    horizon = gt_request.get(
        "horizon"
    )

    eligible = (
        gt_test_path.exists()
    )

    metadata = (
        safe_json(
            metadata_path
        )
        or {}
    )

    base = {
        "case_id": case_id,
        "dataset": dataset,
        "model": MODEL_NAME,
        "granularity": granularity,
        "horizon": horizon,
        "system_error": system_error,
        "error_file": (
            error_info.get(
                "error_file",
                ""
            )
            if error_info
            else ""
        ),
        "error_message": (
            error_info.get(
                "error_message",
                ""
            )
            if error_info
            else ""
        ),
        "evaluation_eligible": (
            eligible
        ),
        "pipeline_status": (
            metadata.get(
                "status"
            )
        ),
    }

    # --------------------------------------------
    # Không có Ground Truth test
    # --------------------------------------------
    if not eligible:

        forecast_available = (
            forecast_path.exists()
        )

        return {
            **base,
            "status": "not_eligible",
            "forecast_available": (
                forecast_available
            ),
            "forecast_valid": np.nan,
            "n_actual": 0,
            "n_forecast": np.nan,
            "horizon_match": np.nan,
            "date_match": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "fallback_used": np.nan,
            "selected_mode": None,
        }

    # --------------------------------------------
    # Eligible nhưng pipeline không có forecast
    # --------------------------------------------
    if not forecast_path.exists():

        status = (
            "system_error"
            if system_error
            else "missing_forecast"
        )

        return {
            **base,
            "status": status,
            "forecast_available": False,
            "forecast_valid": False,
            "n_actual": horizon,
            "n_forecast": 0,
            "horizon_match": False,
            "date_match": False,
            "mae": np.nan,
            "rmse": np.nan,
            "fallback_used": np.nan,
            "selected_mode": None,
        }

    try:

        actual = normalize_ts(
            load_test(
                gt_test_path
            )
        )

        forecast_raw, forecast_obj = (
            load_forecast(
                forecast_path
            )
        )

        forecast = normalize_ts(
            forecast_raw
        )

        n_actual = len(actual)
        n_forecast = len(
            forecast
        )

        horizon_match = (
            n_actual
            == n_forecast
            == int(horizon)
        )

        if not horizon_match:
            return {
                **base,
                "status": (
                    "horizon_mismatch"
                ),
                "forecast_available": True,
                "forecast_valid": False,
                "n_actual": n_actual,
                "n_forecast": n_forecast,
                "horizon_match": False,
                "date_match": False,
                "mae": np.nan,
                "rmse": np.nan,
                "fallback_used": (
                    forecast_obj.get(
                        "fallback_used"
                    )
                ),
                "selected_mode": (
                    forecast_obj.get(
                        "selected_mode"
                    )
                ),
            }

        date_match = (
            actual["ds"].tolist()
            == forecast["ds"].tolist()
        )

        if not date_match:
            return {
                **base,
                "status": (
                    "date_mismatch"
                ),
                "forecast_available": True,
                "forecast_valid": False,
                "n_actual": n_actual,
                "n_forecast": n_forecast,
                "horizon_match": True,
                "date_match": False,
                "mae": np.nan,
                "rmse": np.nan,
                "fallback_used": (
                    forecast_obj.get(
                        "fallback_used"
                    )
                ),
                "selected_mode": (
                    forecast_obj.get(
                        "selected_mode"
                    )
                ),
            }

        yhat = forecast[
            "y"
        ].to_numpy(float)

        finite_ok = bool(
            np.all(
                np.isfinite(yhat)
            )
        )

        nonnegative_ok = bool(
            np.all(
                yhat >= 0
            )
        )

        if not finite_ok:
            return {
                **base,
                "status": (
                    "non_finite_forecast"
                ),
                "forecast_available": True,
                "forecast_valid": False,
                "n_actual": n_actual,
                "n_forecast": n_forecast,
                "horizon_match": True,
                "date_match": True,
                "mae": np.nan,
                "rmse": np.nan,
                "fallback_used": (
                    forecast_obj.get(
                        "fallback_used"
                    )
                ),
                "selected_mode": (
                    forecast_obj.get(
                        "selected_mode"
                    )
                ),
            }

        mae, rmse = mae_rmse(
            actual["y"].to_numpy(
                float
            ),
            yhat,
        )

        return {
            **base,
            "status": "success",
            "forecast_available": True,
            "forecast_valid": True,
            "n_actual": n_actual,
            "n_forecast": n_forecast,
            "horizon_match": True,
            "date_match": True,
            "finite_ok": finite_ok,
            "nonnegative_ok": (
                nonnegative_ok
            ),
            "mae": mae,
            "rmse": rmse,
            "fallback_used": bool(
                forecast_obj.get(
                    "fallback_used",
                    False,
                )
            ),
            "selected_mode": (
                forecast_obj.get(
                    "selected_mode"
                )
            ),
            "short_window_n": (
                forecast_obj
                .get(
                    "window_info",
                    {}
                )
                .get("short_n")
            ),
            "long_window_n": (
                forecast_obj
                .get(
                    "window_info",
                    {}
                )
                .get("long_n")
            ),
        }

    except Exception as e:

        return {
            **base,
            "status": (
                "evaluation_error"
            ),
            "forecast_available": True,
            "forecast_valid": False,
            "mae": np.nan,
            "rmse": np.nan,
            "evaluation_error": str(e),
        }


# ============================================================
# PIPELINE / CASE STATUS
# ============================================================

def evaluate_case_status(
    case_id: str,
    dataset: str,
    case_dir: Path,
    error_info: dict | None,
) -> dict:

    required = [
        "request.json",
        "generated.sql",
        "sql_result.csv",
        "history.json",
        "forecast.json",
        "metadata.json",
    ]

    available = {
        name: (
            case_dir / name
        ).exists()
        for name in required
    }

    system_error = (
        error_info is not None
    )

    complete = all(
        available.values()
    )

    return {
        "case_id": case_id,
        "dataset": dataset,
        "model": MODEL_NAME,
        "system_error": (
            system_error
        ),
        "case_complete": complete,
        "request_available": (
            available[
                "request.json"
            ]
        ),
        "sql_available": (
            available[
                "generated.sql"
            ]
        ),
        "sql_result_available": (
            available[
                "sql_result.csv"
            ]
        ),
        "history_available": (
            available[
                "history.json"
            ]
        ),
        "forecast_available": (
            available[
                "forecast.json"
            ]
        ),
        "metadata_available": (
            available[
                "metadata.json"
            ]
        ),
        "error_file": (
            error_info.get(
                "error_file",
                ""
            )
            if error_info
            else ""
        ),
        "error_message": (
            error_info.get(
                "error_message",
                ""
            )
            if error_info
            else ""
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def rate(
    series: pd.Series,
) -> float:

    values = (
        series
        .dropna()
    )

    if len(values) == 0:
        return np.nan

    return float(
        values.astype(
            float
        ).mean()
    )


def build_summary(
    dataset: str,
    status_df: pd.DataFrame,
    orch_df: pd.DataFrame,
    sql_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> dict:

    n_cases = len(status_df)

    n_system_errors = int(
        status_df[
            "system_error"
        ].sum()
    )

    # ----------------------------
    # End-to-end
    # ----------------------------

    e2e_success = (
        n_cases
        - n_system_errors
    )

    e2e_success_rate = (
        e2e_success / n_cases
        if n_cases
        else np.nan
    )

    # ----------------------------
    # Orchestration
    # ----------------------------

    orchestration_em = rate(
        orch_df[
            "overall_exact_match"
        ]
    )

    # ----------------------------
    # SQL
    #
    # overall:
    # denominator = toàn bộ 63
    #
    # conditional:
    # chỉ case có SQL result
    # ----------------------------

    sql_overall_ex = rate(
        sql_df[
            "execution_accuracy"
        ]
    )

    sql_available = sql_df[
        sql_df[
            "sql_result_available"
        ] == True
    ]

    sql_conditional_ex = (
        rate(
            sql_available[
                "execution_accuracy"
            ]
        )
        if len(sql_available)
        else np.nan
    )

    # ----------------------------
    # Integrity
    # ----------------------------

    integrity_available = (
        integrity_df[
            integrity_df[
                "status"
            ] == "success"
        ]
    )

    history_integrity_rate = (
        rate(
            integrity_available[
                "history_match"
            ]
        )
        if len(
            integrity_available
        )
        else np.nan
    )

    # ----------------------------
    # Forecast
    # ----------------------------

    eligible = forecast_df[
        forecast_df[
            "evaluation_eligible"
        ] == True
    ]

    valid = eligible[
        eligible[
            "forecast_valid"
        ] == True
    ]

    n_eligible = len(eligible)
    n_valid = len(valid)

    forecast_coverage = (
        n_valid / n_eligible
        if n_eligible
        else np.nan
    )

    mean_mae = (
        valid["mae"].mean()
        if n_valid
        else np.nan
    )

    median_mae = (
        valid["mae"].median()
        if n_valid
        else np.nan
    )

    mean_rmse = (
        valid["rmse"].mean()
        if n_valid
        else np.nan
    )

    median_rmse = (
        valid["rmse"].median()
        if n_valid
        else np.nan
    )

    fallback_rate = (
        rate(
            valid[
                "fallback_used"
            ]
        )
        if n_valid
        else np.nan
    )

    direct = valid[
        valid[
            "fallback_used"
        ] == False
    ]

    fallback = valid[
        valid[
            "fallback_used"
        ] == True
    ]

    return {
        "dataset": dataset,
        "model": MODEL_NAME,

        "n_cases": n_cases,

        "n_system_errors": (
            n_system_errors
        ),

        "n_end_to_end_success": (
            e2e_success
        ),

        "end_to_end_success_rate": (
            e2e_success_rate
        ),

        "orchestration_em_rate": (
            orchestration_em
        ),

        "sql_execution_accuracy_overall": (
            sql_overall_ex
        ),

        "n_sql_results": len(
            sql_available
        ),

        "sql_execution_accuracy_conditional": (
            sql_conditional_ex
        ),

        "history_integrity_rate": (
            history_integrity_rate
        ),

        "n_forecast_eligible": (
            n_eligible
        ),

        "n_forecast_valid": (
            n_valid
        ),

        "forecast_coverage": (
            forecast_coverage
        ),

        "mean_mae": mean_mae,
        "median_mae": median_mae,

        "mean_rmse": mean_rmse,
        "median_rmse": median_rmse,

        "n_direct_llm": len(
            direct
        ),

        "n_fallback": len(
            fallback
        ),

        "fallback_rate": (
            fallback_rate
        ),

        "direct_llm_mean_mae": (
            direct["mae"].mean()
            if len(direct)
            else np.nan
        ),

        "direct_llm_mean_rmse": (
            direct["rmse"].mean()
            if len(direct)
            else np.nan
        ),

        "fallback_mean_mae": (
            fallback["mae"].mean()
            if len(fallback)
            else np.nan
        ),

        "fallback_mean_rmse": (
            fallback["rmse"].mean()
            if len(fallback)
            else np.nan
        ),
    }


def build_forecast_breakdown(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast metrics by dataset and granularity.

    Keep every Ground-Truth-eligible case in the denominator. MAE/RMSE are
    calculated only on valid forecasts; invalid/upstream-failed cases remain
    visible through n_invalid and forecast_coverage.
    """
    rows = []
    eligible_df = forecast_df[
        forecast_df["evaluation_eligible"] == True
    ].copy()

    if eligible_df.empty:
        return pd.DataFrame()

    for (dataset, granularity), group in eligible_df.groupby(
        ["dataset", "granularity"], dropna=False
    ):
        valid = group[group["forecast_valid"] == True]
        invalid = group[group["forecast_valid"] != True]
        direct = valid[valid["fallback_used"] == False]
        fallback = valid[valid["fallback_used"] == True]

        n_eligible = len(group)
        n_valid = len(valid)
        n_invalid = len(invalid)

        rows.append({
            "dataset": dataset,
            "granularity": granularity,
            "n_eligible": n_eligible,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "forecast_coverage": (
                n_valid / n_eligible if n_eligible else np.nan
            ),
            "mean_mae": valid["mae"].mean() if n_valid else np.nan,
            "median_mae": valid["mae"].median() if n_valid else np.nan,
            "mean_rmse": valid["rmse"].mean() if n_valid else np.nan,
            "median_rmse": valid["rmse"].median() if n_valid else np.nan,
            "n_direct_llm": len(direct),
            "n_fallback": len(fallback),
            "fallback_rate": (
                len(fallback) / n_valid if n_valid else np.nan
            ),
            "direct_llm_mean_mae": (
                direct["mae"].mean() if len(direct) else np.nan
            ),
            "direct_llm_mean_rmse": (
                direct["rmse"].mean() if len(direct) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_agent_conditional_summary(
    orch_df: pd.DataFrame,
    sql_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Separate end-to-end outcomes from conditional component performance.

    This prevents one upstream Router error from being interpreted as
    independent SQL/History/Forecast-Agent failures.
    """
    rows = []

    for dataset in sorted(orch_df["dataset"].unique()):
        orch = orch_df[orch_df["dataset"] == dataset].copy()
        sql = sql_df[sql_df["dataset"] == dataset].copy()
        integ = integrity_df[integrity_df["dataset"] == dataset].copy()
        fc = forecast_df[forecast_df["dataset"] == dataset].copy()

        # Router
        router_correct_ids = set(
            orch.loc[orch["overall_exact_match"] == True, "case_id"]
        )
        n_total = len(orch)
        n_router_correct = len(router_correct_ids)

        # SQL, conditional on correct Router output
        sql_after_router = sql[sql["case_id"].isin(router_correct_ids)]
        sql_correct_ids = set(
            sql_after_router.loc[
                sql_after_router["execution_accuracy"] == True,
                "case_id",
            ]
        )
        n_sql_input = len(sql_after_router)
        n_sql_correct = len(sql_correct_ids)

        # History, conditional on correct Router + SQL
        integ_after_sql = integ[integ["case_id"].isin(sql_correct_ids)]
        history_correct_ids = set(
            integ_after_sql.loc[
                integ_after_sql["history_match"] == True,
                "case_id",
            ]
        )
        n_history_input = len(integ_after_sql)
        n_history_correct = len(history_correct_ids)

        # Forecast end-to-end denominator: all GT-eligible cases
        fc_eligible_all = fc[fc["evaluation_eligible"] == True]
        fc_valid_all = fc_eligible_all[
            fc_eligible_all["forecast_valid"] == True
        ]

        # Forecast Agent conditional denominator: GT eligible + all upstream OK
        fc_upstream_ok = fc[
            (fc["case_id"].isin(history_correct_ids))
            & (fc["evaluation_eligible"] == True)
        ]
        fc_valid = fc_upstream_ok[
            fc_upstream_ok["forecast_valid"] == True
        ]

        n_forecast_input = len(fc_upstream_ok)
        n_forecast_valid = len(fc_valid)

        rows.append({
            "dataset": dataset,
            "model": MODEL_NAME,
            "n_total_cases": n_total,
            "n_router_correct": n_router_correct,
            "router_exact_match_rate": (
                n_router_correct / n_total if n_total else np.nan
            ),
            "n_sql_input_after_router_correct": n_sql_input,
            "n_sql_correct": n_sql_correct,
            "sql_accuracy_given_router_correct": (
                n_sql_correct / n_sql_input if n_sql_input else np.nan
            ),
            "n_history_input_after_sql_correct": n_history_input,
            "n_history_correct": n_history_correct,
            "history_accuracy_given_upstream_correct": (
                n_history_correct / n_history_input
                if n_history_input else np.nan
            ),
            "n_forecast_eligible_e2e": len(fc_eligible_all),
            "n_forecast_valid_e2e": len(fc_valid_all),
            "forecast_coverage_e2e": (
                len(fc_valid_all) / len(fc_eligible_all)
                if len(fc_eligible_all) else np.nan
            ),
            "n_forecast_input_upstream_correct": n_forecast_input,
            "n_forecast_valid_upstream_correct": n_forecast_valid,
            "forecast_coverage_given_upstream_correct": (
                n_forecast_valid / n_forecast_input
                if n_forecast_input else np.nan
            ),
            "forecast_mean_mae_given_upstream_correct": (
                fc_valid["mae"].mean() if n_forecast_valid else np.nan
            ),
            "forecast_mean_rmse_given_upstream_correct": (
                fc_valid["rmse"].mean() if n_forecast_valid else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN DATASET LOOP
# ============================================================

def evaluate_dataset(
    dataset: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    cfg = CONFIGS[dataset]

    prefix = cfg["prefix"]

    gt_request_dir = (
        cfg["gt_base"]
        / f"{prefix}_forecast_requests"
    )

    gt_history_dir = (
        cfg["gt_base"]
        / f"{prefix}_historical_data"
    )

    gt_train_dir = (
        cfg["gt_base"]
        / f"{prefix}_train_data"
    )

    gt_test_dir = (
        cfg["gt_base"]
        / f"{prefix}_tests"
    )

    case_ids = sorted(
        path.stem
        for path
        in gt_request_dir.glob(
            "*.json"
        )
    )

    error_map = discover_errors(
        dataset,
        cfg,
    )

    print()
    print("=" * 70)
    print(
        f"{dataset} / "
        f"{MODEL_NAME}"
    )
    print("=" * 70)

    print(
        f"Ground Truth cases : "
        f"{len(case_ids)}"
    )

    print(
        f"Detected error cases: "
        f"{len(error_map)}"
    )

    if error_map:
        print(
            "Error case IDs      : "
            + ", ".join(
                sorted(
                    error_map
                )
            )
        )

    status_rows = []
    orch_rows = []
    sql_rows = []
    integrity_rows = []
    forecast_rows = []

    for case_id in case_ids:

        case_dir = (
            cfg["cases_dir"]
            / case_id
        )

        error_info = (
            error_map.get(
                case_id
            )
        )

        system_error = (
            error_info is not None
        )

        gt_request = (
            gt_request_dir
            / f"{case_id}.json"
        )

        gt_history = (
            gt_history_dir
            / f"{case_id}.csv"
        )

        gt_train = (
            gt_train_dir
            / f"{case_id}.csv"
        )

        gt_test = (
            gt_test_dir
            / f"{case_id}.json"
        )

        status_rows.append(
            evaluate_case_status(
                case_id,
                dataset,
                case_dir,
                error_info,
            )
        )

        orch_rows.append(
            evaluate_orchestration(
                case_id=case_id,
                dataset=dataset,
                gt_path=gt_request,
                output_path=(
                    case_dir
                    / "request.json"
                ),
                system_error=(
                    system_error
                ),
            )
        )

        sql_rows.append(
            evaluate_sql(
                case_id=case_id,
                dataset=dataset,
                gt_history_path=(
                    gt_history
                ),
                sql_result_path=(
                    case_dir
                    / "sql_result.csv"
                ),
                system_error=(
                    system_error
                ),
            )
        )

        integrity_rows.append(
            evaluate_integrity(
                case_id=case_id,
                dataset=dataset,
                gt_history_path=(
                    gt_history
                ),
                gt_train_path=(
                    gt_train
                ),
                gt_test_path=(
                    gt_test
                ),
                output_history_path=(
                    case_dir
                    / "history.json"
                ),
                system_error=(
                    system_error
                ),
            )
        )

        forecast_rows.append(
            evaluate_forecast(
                case_id=case_id,
                dataset=dataset,
                gt_request_path=(
                    gt_request
                ),
                gt_test_path=(
                    gt_test
                ),
                forecast_path=(
                    case_dir
                    / "forecast.json"
                ),
                metadata_path=(
                    case_dir
                    / "metadata.json"
                ),
                system_error=(
                    system_error
                ),
                error_info=(
                    error_info
                ),
            )
        )

    status_df = pd.DataFrame(
        status_rows
    )

    orch_df = pd.DataFrame(
        orch_rows
    )

    sql_df = pd.DataFrame(
        sql_rows
    )

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    summary = build_summary(
        dataset,
        status_df,
        orch_df,
        sql_df,
        integrity_df,
        forecast_df,
    )

    return (
        status_df,
        orch_df,
        sql_df,
        integrity_df,
        forecast_df,
        summary,
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def configure_architecture(architecture: str) -> None:
    """Select artifacts and score destination without changing scoring rules."""
    global RESULT_ROOT
    if architecture not in {"multi_agent", "single_agent"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    single = architecture == "single_agent"
    source_root = OUTPUT_ROOT / "single_agent_ablation" if single else OUTPUT_ROOT
    results_base = PROJECT_ROOT / "evaluation" / "results"
    RESULT_ROOT = (
        results_base / "single_agent_ablation" / "gemma"
        if single else results_base / "gemma"
    )
    for dataset, folder in (("M5", "m5_exp"), ("HCMC_BUS", "hcmcbus_exp")):
        model_root = source_root / folder / "gemma4"
        CONFIGS[dataset]["cases_dir"] = model_root / "cases"
        CONFIGS[dataset]["error_dirs"] = (
            [model_root / "errors"] if single
            else [source_root / folder / "errors", model_root / "errors"]
        )


def main():
    parser = argparse.ArgumentParser(description="Score existing Gemma artifacts against ground truth; no LLM calls.")
    parser.add_argument(
        "--architecture", choices=["multi_agent", "single_agent"],
        default="multi_agent", help="Artifact set to score (default: multi_agent).",
    )
    args = parser.parse_args()
    configure_architecture(args.architecture)
    # Refuse an incorrectly configured run before writing any scores.
    for dataset, cfg in CONFIGS.items():
        if not cfg["cases_dir"].is_dir():
            raise SystemExit(f"Missing {dataset} cases directory: {cfg['cases_dir']}")
    print(f"Architecture: {args.architecture}")

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_status = []
    all_orch = []
    all_sql = []
    all_integrity = []
    all_forecast = []
    summaries = []

    for dataset in [
        "M5",
        "HCMC_BUS",
    ]:

        (
            status_df,
            orch_df,
            sql_df,
            integrity_df,
            forecast_df,
            summary,
        ) = evaluate_dataset(
            dataset
        )

        all_status.append(
            status_df
        )

        all_orch.append(
            orch_df
        )

        all_sql.append(
            sql_df
        )

        all_integrity.append(
            integrity_df
        )

        all_forecast.append(
            forecast_df
        )

        summaries.append(
            summary
        )

    status_df = pd.concat(
        all_status,
        ignore_index=True,
    )

    orch_df = pd.concat(
        all_orch,
        ignore_index=True,
    )

    sql_df = pd.concat(
        all_sql,
        ignore_index=True,
    )

    integrity_df = pd.concat(
        all_integrity,
        ignore_index=True,
    )

    forecast_df = pd.concat(
        all_forecast,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    breakdown_df = (
        build_forecast_breakdown(
            forecast_df
        )
    )

    conditional_df = (
        build_agent_conditional_summary(
            orch_df,
            sql_df,
            integrity_df,
            forecast_df,
        )
    )

    status_df.to_csv(
        RESULT_ROOT
        / "case_status.csv",
        index=False,
    )

    orch_df.to_csv(
        RESULT_ROOT
        / "orchestration_per_case.csv",
        index=False,
    )

    sql_df.to_csv(
        RESULT_ROOT
        / "sql_per_case.csv",
        index=False,
    )

    integrity_df.to_csv(
        RESULT_ROOT
        / "integrity_per_case.csv",
        index=False,
    )

    forecast_df.to_csv(
        RESULT_ROOT
        / "forecast_per_case.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULT_ROOT
        / "evaluation_summary.csv",
        index=False,
    )

    breakdown_df.to_csv(
        RESULT_ROOT
        / "forecast_by_granularity.csv",
        index=False,
    )

    conditional_df.to_csv(
        RESULT_ROOT
        / "agent_conditional_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL GEMMA SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("AGENT CONDITIONAL SUMMARY")
    print("=" * 70)
    print(conditional_df.to_string(index=False))

    print()
    print("=" * 70)
    print("FORECAST BY GRANULARITY")
    print("=" * 70)
    print(breakdown_df.to_string(index=False))

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_ROOT
    )


if __name__ == "__main__":
    main()
