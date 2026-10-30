from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

from _common import (
    ROOT,
    EVAL_RESULTS,
    OUT,
    MODEL_DISPLAY,
    norm_dataset,
    forecast_per_case,
    col,
    boolish,
    TRACE_CONFIGS,
    stage_to_agent,
)

APPENDIX = OUT / "appendix"
APPENDIX.mkdir(parents=True, exist_ok=True)

def save_appendix(df: pd.DataFrame, filename: str):
    path = APPENDIX / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved appendix: {path}")
    print(df.to_string(index=False))
    return path

def first_existing(df: pd.DataFrame, names, required=False):
    for c in names:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Không tìm thấy cột nào trong {names}. "
            f"Các cột hiện có: {list(df.columns)}"
        )
    return None

def normalize_model(v):
    s = str(v).strip()
    if s in MODEL_DISPLAY:
        return MODEL_DISPLAY[s]
    if s == "gemini-2.5-pro":
        return "Gemini 2.5 Pro"
    if s == "gemma4:e4b":
        return "Gemma4 (gemma4:e4b)"
    return s
