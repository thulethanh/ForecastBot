from __future__ import annotations
from pathlib import Path
import json, math
import pandas as pd
import numpy as np

MODEL_DISPLAY = {"gemini":"Gemini 2.5 Pro","gemma":"Gemma4 (gemma4:e4b)"}
DATASET_DISPLAY = {"M5":"M5","HCMC_BUS":"HCMC Bus","HCMC Bus":"HCMC Bus"}
GRANULARITY_DISPLAY = {"day":"Ngày","month":"Tháng","quarter":"Quý","year":"Năm"}

def project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent/"evaluation").exists() and (parent/"outputs").exists():
            return parent
    raise RuntimeError("Không tìm thấy project root (cần có evaluation/ và outputs/).")

ROOT = project_root()
EVAL_RESULTS = ROOT/"evaluation"/"results"
THESIS = ROOT/"evaluation"/"thesis_analysis"
OUT = THESIS/"outputs"
SUPPORT = OUT/"supporting"
OUT.mkdir(parents=True, exist_ok=True)
SUPPORT.mkdir(parents=True, exist_ok=True)

TRACE_CONFIGS = [
    ("M5","Gemini 2.5 Pro", ROOT/"outputs"/"m5_exp"/"gemini_2_5_pro"/"cases"),
    ("M5","Gemma4 (gemma4:e4b)", ROOT/"outputs"/"m5_exp"/"gemma4"/"cases"),
    ("HCMC Bus","Gemini 2.5 Pro", ROOT/"outputs"/"hcmcbus_exp"/"gemini_2_5_pro"/"cases"),
    ("HCMC Bus","Gemma4 (gemma4:e4b)", ROOT/"outputs"/"hcmcbus_exp"/"gemma4"/"cases"),
]

def norm_dataset(x):
    if pd.isna(x): return x
    return DATASET_DISPLAY.get(str(x).strip(), str(x).strip())

def pct(x):
    return np.nan if pd.isna(x) else round(float(x)*100, 2)

def nround(x, n=2):
    return np.nan if pd.isna(x) else round(float(x), n)

def read(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy nguồn: {path}")
    return pd.read_csv(path)

def require(df, cols, label):
    miss=[c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{label} thiếu cột {miss}. Các cột hiện có: {list(df.columns)}")

def save_main(df, name):
    p=OUT/name
    df.to_csv(p,index=False,encoding="utf-8-sig")
    print(f"Saved: {p}")
    print(df.to_string(index=False))
    return p

def save_support(df, name):
    p=SUPPORT/name
    df.to_csv(p,index=False,encoding="utf-8-sig")
    print(f"Supporting: {p}")
    return p

def eval_summary(model_key):
    df=read(EVAL_RESULTS/model_key/"evaluation_summary.csv").copy()
    df["dataset"]=df["dataset"].map(norm_dataset)
    df["model_display"]=MODEL_DISPLAY[model_key]
    return df

def conditional_summary(model_key):
    df=read(EVAL_RESULTS/model_key/"agent_conditional_summary.csv").copy()
    df["dataset"]=df["dataset"].map(norm_dataset)
    df["model_display"]=MODEL_DISPLAY[model_key]
    return df

def explanation_summary(model_key):
    candidates=[
        EVAL_RESULTS/"llm_judge"/model_key/"explanation_summary.csv",
        EVAL_RESULTS/"llm_judge"/"explanation_summary.csv",
    ]
    for p in candidates:
        if not p.exists(): continue
        df=pd.read_csv(p).copy()
        if "target_model_key" in df.columns:
            df=df[df["target_model_key"].astype(str).str.lower()==model_key].copy()
        if "n_cases" in df.columns:
            exact=df[df["n_cases"]==63]
            if len(exact): df=exact
        if len(df):
            df["dataset"]=df["dataset"].map(norm_dataset)
            df["model_display"]=MODEL_DISPLAY[model_key]
            return df
    raise FileNotFoundError(f"Không tìm thấy explanation_summary cho {model_key}")

def forecast_per_case(model_key):
    candidates=[
        EVAL_RESULTS/model_key/"forecast_per_case.csv",
        EVAL_RESULTS/model_key/"forecast_evaluation_per_case.csv",
    ]
    for p in candidates:
        if p.exists():
            df=pd.read_csv(p).copy()
            if "dataset" in df.columns: df["dataset"]=df["dataset"].map(norm_dataset)
            return df
    raise FileNotFoundError(f"Không tìm thấy forecast_per_case cho {model_key}")

def col(df, names, required=True):
    for c in names:
        if c in df.columns: return c
    if required:
        raise ValueError(f"Không tìm thấy cột nào trong {names}. Hiện có: {list(df.columns)}")
    return None

def boolish(s):
    if s.dtype == bool: return s
    return s.astype(str).str.lower().isin(["true","1","yes","y","valid","success"])

def stage_to_agent(stage):
    s=str(stage or "").lower()
    if "orchestration" in s or s=="router": return "Router Agent"
    if s.startswith("sql_") or ".sql_" in s or "sql_generate" in s or "sql_repair" in s: return "SQL Agent"
    if "forecast" in s: return "Forecast Agent"
    if "explanation" in s: return "Explanation Agent"
    return None

def deep_find_numeric(obj, keys):
    """Find first finite numeric value for preferred keys recursively."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                try:
                    v=float(obj[key])
                    if math.isfinite(v): return v
                except Exception: pass
        for v in obj.values():
            got=deep_find_numeric(v,keys)
            if got is not None: return got
    elif isinstance(obj,list):
        for v in obj:
            got=deep_find_numeric(v,keys)
            if got is not None: return got
    return None

def extract_e2e_time(case_dir: Path, trace_sum: float):
    """
    Lấy thời gian end-to-end chính xác được runner.py ghi nhận.

    runner.py đo:
        timings['end_to_end'] = time.perf_counter() - t_all

    và lưu timings vào metadata.json/result.json.
    """

    for fn in ["metadata.json", "result.json"]:
        p = case_dir / fn

        if not p.exists():
            continue

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))

            timings = obj.get("timings", {})

            if isinstance(timings, dict):
                v = timings.get("end_to_end")

                if v is not None:
                    v = float(v)

                    if math.isfinite(v) and v >= 0:
                        return v, f"{fn}:timings.end_to_end"

        except Exception:
            pass

    # Không suy diễn E2E từ tổng llm_time_s.
    return np.nan, "missing_timings.end_to_end"
