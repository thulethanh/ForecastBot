#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tạo các bảng trình bày gọn cho Mục 5.2 và Phụ lục B của luận văn ForecastBot.

Nguồn vào:
    evaluation/thesis_analysis/outputs/
        01_overall_summary.csv
        02_router_sql_summary.csv
        03_forecast_summary.csv
        04_explanation_summary.csv
        05_performance_summary.csv
        appendix/
            PL_A1_forecast_by_granularity.csv
            PL_A2_agent_performance.csv
            PL_A3_e2e_performance.csv
            PL_A4_paired_wilcoxon.csv

Đầu ra:
    evaluation/thesis_analysis/outputs/thesis_tables_compact/
        CH5_01_overall_summary_compact.csv
        CH5_02_router_sql_compact.csv
        CH5_03_forecast_compact.csv
        CH5_04_explanation_compact.csv
        CH5_05_e2e_performance_compact.csv
        CH5_06_agent_time_compact.csv
        PL_B1_forecast_by_granularity_compact.csv
        PL_B2a_agent_time_compact.csv
        PL_B2b_agent_token_compact.csv
        PL_B3_e2e_performance_compact.csv
        PL_B4_paired_wilcoxon_compact.csv

Mục tiêu:
- Chỉ biến đổi cách trình bày bảng, KHÔNG tính lại kết quả thực nghiệm.
- Ghép các cột định danh thành "Bộ dữ liệu – Mô hình" hoặc "Bộ dữ liệu – Chỉ số".
- B.1 xoay Ngày/Tháng/Quý/Năm thành cột và chỉ giữ MAE/RMSE trung bình.
- B.2 xoay Router/SQL/Forecast/Explanation thành cột thời gian trung bình.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
import pandas as pd


ROOT = Path("evaluation/thesis_analysis/outputs")
APPENDIX_DIR = ROOT / "appendix"
OUT_DIR = ROOT / "thesis_tables_compact"


def norm(s: str) -> str:
    """Chuẩn hóa tên cột: bỏ dấu tiếng Việt, chữ thường, ký tự đặc biệt -> _."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("đ", "d")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def colmap(df: pd.DataFrame) -> dict[str, str]:
    return {norm(c): c for c in df.columns}


def find_col(df: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    cmap = colmap(df)
    for a in aliases:
        key = norm(a)
        if key in cmap:
            return cmap[key]

    # fallback: contains all tokens
    for a in aliases:
        tokens = [t for t in norm(a).split("_") if t]
        for k, original in cmap.items():
            if all(t in k for t in tokens):
                return original

    if required:
        raise KeyError(
            f"Không tìm thấy cột tương ứng với {aliases}.\n"
            f"Các cột hiện có: {list(df.columns)}"
        )
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def canonical_dataset(v) -> str:
    s = str(v).strip()
    u = s.upper()
    if "HCMC" in u or "BUS" in u:
        return "HCMC Bus"
    if "M5" in u:
        return "M5"
    return s


def canonical_model(v) -> str:
    s = str(v).strip()
    u = s.upper()
    if "GEMINI" in u:
        return "Gemini 2.5 Pro"
    if "GEMMA" in u:
        return "Gemma4 (gemma4:e4b)"
    return s


def canonical_granularity(v) -> str:
    s = str(v).strip().lower()
    mapping = {
        "day": "Ngày", "daily": "Ngày", "ngay": "Ngày", "ngày": "Ngày",
        "month": "Tháng", "monthly": "Tháng", "thang": "Tháng", "tháng": "Tháng",
        "quarter": "Quý", "quarterly": "Quý", "quy": "Quý", "quý": "Quý",
        "year": "Năm", "yearly": "Năm", "nam": "Năm", "năm": "Năm",
    }
    return mapping.get(s, str(v).strip())


def canonical_agent(v) -> str:
    s = str(v).strip().lower()
    if "router" in s:
        return "Router"
    if "sql" in s:
        return "SQL"
    if "forecast" in s:
        return "Forecast"
    if "explanation" in s or "explain" in s:
        return "Explanation"
    return str(v).strip()


def canonical_metric(v) -> str:
    s = str(v).strip().upper()
    if "RMSE" in s:
        return "RMSE"
    if "MAE" in s:
        return "MAE"
    return str(v).strip()


def fmt_pct(v) -> str:
    if pd.isna(v):
        return "–"
    x = float(v)
    # Nếu nguồn đang ở dạng 0-1 thì đổi sang %
    if 0 <= x <= 1:
        x *= 100
    return f"{x:.2f}%"


def fmt_num(v, nd=4) -> str:
    if pd.isna(v):
        return "–"
    return f"{float(v):.{nd}f}"


def fmt_int(v) -> str:
    if pd.isna(v):
        return "–"
    return str(int(round(float(v))))


def id_series(df: pd.DataFrame) -> pd.Series:
    dcol = find_col(df, "dataset", "dataset_name", "bo_du_lieu", "Bộ dữ liệu")
    mcol = find_col(df, "model", "model_name", "mo_hinh", "Mô hình")
    return df[dcol].map(canonical_dataset) + " – " + df[mcol].map(canonical_model)


def ordered(df: pd.DataFrame, id_col="Bộ dữ liệu – Mô hình") -> pd.DataFrame:
    order = [
        "M5 – Gemini 2.5 Pro",
        "M5 – Gemma4 (gemma4:e4b)",
        "HCMC Bus – Gemini 2.5 Pro",
        "HCMC Bus – Gemma4 (gemma4:e4b)",
    ]
    rank = {v: i for i, v in enumerate(order)}
    out = df.copy()
    out["_order"] = out[id_col].map(rank).fillna(999)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def save(df: pd.DataFrame, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {path}")


# ---------------------------------------------------------------------
# CHƯƠNG 5
# ---------------------------------------------------------------------

def build_ch5_overall():
    df = read_csv(ROOT / "01_overall_summary.csv")

    router = find_col(df, "router_em_pct", "router_accuracy_pct", "router_pct", "router", "Router - Tỷ lệ khớp hoàn toàn (%)")
    sql = find_col(df, "sql_accuracy_pct", "sql_conditional_accuracy_pct", "sql_pct", "sql", "SQL - Độ chính xác truy xuất có điều kiện (%)")
    coverage = find_col(df, "forecast_coverage_pct", "forecast_valid_pct", "forecast_coverage", "forecast", "Forecast - Tỷ lệ tạo dự báo hợp lệ (%)")
    explanation = find_col(df, "explanation_overall", "explanation_score", "explanation", "Explanation - Điểm trung bình (1-5)")
    e2e = find_col(df, "e2e_success_pct", "end_to_end_success_pct", "e2e_success", "end_to_end", "ForecastBot - Tỷ lệ hoàn thành toàn bộ quy trình (%)")

    out = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Router": df[router].map(fmt_pct),
        "SQL": df[sql].map(fmt_pct),
        "Dự báo hợp lệ": df[coverage].map(fmt_pct),
        "Giải thích": df[explanation].map(lambda x: "–" if pd.isna(x) else f"{float(x):.2f}/5"),
        "Hoàn thành toàn bộ quy trình": df[e2e].map(fmt_pct),
    })
    save(ordered(out), "CH5_01_overall_summary_compact.csv")


def build_ch5_router_sql():
    df = read_csv(ROOT / "02_router_sql_summary.csv")

    total = find_col(df, "total_cases", "total", "n_cases", "Tổng số yêu cầu", "Tổng số")
    router_ok = find_col(df, "router_correct", "router_correct_cases", "router_ok", "Router đúng")
    router_pct = find_col(df, "router_em_pct", "router_accuracy_pct", "router_pct", "Router - Tỷ lệ khớp hoàn toàn (%)", "Tỷ lệ khớp hoàn toàn (%)")
    sql_in = find_col(df, "sql_conditional_input", "sql_input_cases", "sql_cases", "sql_input", "Số trường hợp vào SQL sau Router đúng", "Trường hợp vào SQL")
    sql_ok = find_col(df, "sql_correct", "sql_correct_cases", "sql_ok", "SQL đúng")
    sql_pct = find_col(df, "sql_accuracy_pct", "sql_conditional_accuracy_pct", "sql_pct", "SQL - Độ chính xác truy xuất có điều kiện (%)", "Độ chính xác truy xuất có điều kiện (%)")

    out = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Tổng số": df[total].map(fmt_int),
        "Router đúng": df[router_ok].map(fmt_int),
        "Tỷ lệ khớp hoàn toàn": df[router_pct].map(fmt_pct),
        "Trường hợp vào SQL": df[sql_in].map(fmt_int),
        "SQL đúng": df[sql_ok].map(fmt_int),
        "Độ chính xác truy xuất": df[sql_pct].map(fmt_pct),
    })
    save(ordered(out), "CH5_02_router_sql_compact.csv")


def build_ch5_forecast():
    df = read_csv(ROOT / "03_forecast_summary.csv")

    # Chỉ lấy dòng tổng hợp, bỏ các dòng granularity nếu file có cả hai
    gcol = find_col(df, "granularity", "time_unit", "Phạm vi", required=False)
    if gcol:
        g = df[gcol].astype(str).str.strip().str.lower()
        mask = (
            g.isin(["total", "overall", "all", "nan", "", "tổng", "tổng hợp", "toàn bộ"])
            | g.str.contains("tong hop|tổng hợp|overall|total", regex=True, na=False)
        )
        if mask.any():
            df = df[mask].copy()

    eligible = find_col(df, "total_eligible", "eligible_cases", "eligible", "Số trường hợp đủ điều kiện", "Đủ điều kiện đánh giá", "Số trường hợp đánh giá")
    valid = find_col(df, "valid_forecasts", "valid_cases", "forecast_valid", "Số dự báo hợp lệ", "Dự báo hợp lệ")
    coverage = find_col(df, "coverage_pct", "forecast_coverage_pct", "coverage", "Tỷ lệ tạo dự báo hợp lệ (%)")
    mae = find_col(df, "mae")
    rmse = find_col(df, "rmse")

    out = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Đủ điều kiện đánh giá": df[eligible].map(fmt_int),
        "Dự báo hợp lệ": df[valid].map(fmt_int),
        "Tỷ lệ tạo dự báo hợp lệ": df[coverage].map(fmt_pct),
        "MAE": df[mae].map(lambda x: fmt_num(x, 4)),
        "RMSE": df[rmse].map(lambda x: fmt_num(x, 4)),
    })
    save(ordered(out), "CH5_03_forecast_compact.csv")


def build_ch5_explanation():
    df = read_csv(ROOT / "04_explanation_summary.csv")

    n = find_col(df, "n_judged", "judged_cases", "n_cases", "cases", "Số trường hợp")
    relevance = find_col(df, "relevance", "relevance_mean", "Tính phù hợp", "Phù hợp")
    faith = find_col(df, "faithfulness", "faithfulness_mean", "Tính trung thực", "Trung thực")
    clarity = find_col(df, "clarity", "clarity_mean", "Tính rõ ràng", "Rõ ràng")
    usefulness = find_col(df, "usefulness", "usefulness_mean", "Tính hữu ích", "Hữu ích")
    overall = find_col(df, "overall", "overall_mean", "average", "Điểm trung bình", "Trung bình")

    out = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Số trường hợp": df[n].map(fmt_int),
        "Phù hợp": df[relevance].map(lambda x: fmt_num(x, 2)),
        "Trung thực": df[faith].map(lambda x: fmt_num(x, 2)),
        "Rõ ràng": df[clarity].map(lambda x: fmt_num(x, 2)),
        "Hữu ích": df[usefulness].map(lambda x: fmt_num(x, 2)),
        "Trung bình": df[overall].map(lambda x: fmt_num(x, 2)),
    })
    save(ordered(out), "CH5_04_explanation_compact.csv")


def build_ch5_performance():
    """
    Tạo 2 bảng hiệu năng cho Chương 5 từ 05_performance_summary.csv.

    Schema thực tế:
    - Bộ dữ liệu
    - Mô hình
    - Phạm vi
    - Thành phần
    - Số trường hợp
    - Thời gian trung bình (s)
    - Thời gian trung vị (s)
    - Token đầu vào TB
    - Token đầu ra TB
    - Tổng token TB
    """
    df = read_csv(ROOT / "05_performance_summary.csv")

    scope_col = find_col(df, "Phạm vi")
    comp_col = find_col(df, "Thành phần")

    # Toàn hệ thống
    scope_norm = df[scope_col].astype(str).str.strip().str.lower()
    overall = df[
        scope_norm.str.contains("toàn hệ thống|toan he thong|overall|end-to-end|end_to_end", regex=True, na=False)
    ].copy()

    if overall.empty:
        raise ValueError("Không tìm thấy dòng 'Toàn hệ thống' trong 05_performance_summary.csv")

    n = find_col(overall, "Số trường hợp")
    avg_t = find_col(overall, "Thời gian trung bình (s)")
    med_t = find_col(overall, "Thời gian trung vị (s)")
    inp = find_col(overall, "Token đầu vào TB")
    outp = find_col(overall, "Token đầu ra TB")
    tot = find_col(overall, "Tổng token TB")

    out_overall = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(overall),
        "Số trường hợp": overall[n].map(fmt_int),
        "Thời gian TB (s)": overall[avg_t].map(lambda x: fmt_num(x, 2)),
        "Trung vị (s)": overall[med_t].map(lambda x: fmt_num(x, 2)),
        "Token đầu vào TB": overall[inp].map(lambda x: fmt_num(x, 2)),
        "Token đầu ra TB": overall[outp].map(lambda x: fmt_num(x, 2)),
        "Tổng token TB": overall[tot].map(lambda x: fmt_num(x, 2)),
    })
    save(ordered(out_overall), "CH5_05_e2e_performance_compact.csv")

    # Theo Agent
    agent_rows = df[
        df[comp_col].astype(str).str.contains("Agent", case=False, na=False)
    ].copy()

    if agent_rows.empty:
        raise ValueError("Không tìm thấy các dòng Agent trong 05_performance_summary.csv")

    tmp = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(agent_rows),
        "Agent": agent_rows[comp_col].map(canonical_agent),
        "Thời gian": pd.to_numeric(agent_rows[avg_t], errors="coerce"),
    })

    piv = tmp.pivot_table(
        index="Bộ dữ liệu – Mô hình",
        columns="Agent",
        values="Thời gian",
        aggfunc="first",
    ).reset_index()

    out_agent = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": piv["Bộ dữ liệu – Mô hình"]
    })

    for a, title in [
        ("Router", "Router (s)"),
        ("SQL", "SQL (s)"),
        ("Forecast", "Forecast (s)"),
        ("Explanation", "Explanation (s)"),
    ]:
        out_agent[title] = (
            piv[a].map(lambda x: fmt_num(x, 2))
            if a in piv.columns else "–"
        )

    save(ordered(out_agent), "CH5_06_agent_time_compact.csv")


# ---------------------------------------------------------------------
# PHỤ LỤC B
# ---------------------------------------------------------------------

def build_pl_b1():
    """
    B.1: xoay kết quả dự báo theo đơn vị thời gian.
    Nguồn: PL_A1_forecast_by_granularity.csv
    """
    df = read_csv(APPENDIX_DIR / "PL_A1_forecast_by_granularity.csv")

    gran = find_col(df, "Đơn vị thời gian")
    mae = find_col(df, "MAE trung bình")
    rmse = find_col(df, "RMSE trung bình")

    tmp = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Đơn vị": df[gran].map(canonical_granularity),
        "MAE": pd.to_numeric(df[mae], errors="coerce"),
        "RMSE": pd.to_numeric(df[rmse], errors="coerce"),
    })

    rows = []
    for ident, g in tmp.groupby("Bộ dữ liệu – Mô hình", sort=False):
        r = {"Bộ dữ liệu – Mô hình": ident}
        for unit in ["Ngày", "Tháng", "Quý", "Năm"]:
            sub = g[g["Đơn vị"] == unit]
            if len(sub):
                r[f"{unit} MAE"] = fmt_num(sub.iloc[0]["MAE"], 4)
                r[f"{unit} RMSE"] = fmt_num(sub.iloc[0]["RMSE"], 4)
            else:
                r[f"{unit} MAE"] = "–"
                r[f"{unit} RMSE"] = "–"
        rows.append(r)

    out = pd.DataFrame(rows, columns=[
        "Bộ dữ liệu – Mô hình",
        "Ngày MAE", "Ngày RMSE",
        "Tháng MAE", "Tháng RMSE",
        "Quý MAE", "Quý RMSE",
        "Năm MAE", "Năm RMSE",
    ])
    save(ordered(out), "PL_B1_forecast_by_granularity_compact.csv")

def build_pl_b2():
    """
    Sinh hai bảng:
    - B.2a: thời gian gọi LLM trung bình theo Agent
    - B.2b: tổng token trung bình theo Agent

    Nguồn: PL_A2_agent_performance.csv
    """
    df = read_csv(APPENDIX_DIR / "PL_A2_agent_performance.csv")

    agent = find_col(df, "Agent")
    avg_time = find_col(df, "Thời gian gọi LLM TB (s)")
    avg_total_tokens = find_col(df, "Tổng token TB")

    tmp = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Agent": df[agent].map(canonical_agent),
        "Thời gian": pd.to_numeric(df[avg_time], errors="coerce"),
        "Tổng token": pd.to_numeric(df[avg_total_tokens], errors="coerce"),
    })

    # B.2a — thời gian
    piv_time = tmp.pivot_table(
        index="Bộ dữ liệu – Mô hình",
        columns="Agent",
        values="Thời gian",
        aggfunc="first",
    ).reset_index()

    out_time = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": piv_time["Bộ dữ liệu – Mô hình"]
    })
    for a, title in [
        ("Router", "Router Agent (s)"),
        ("SQL", "SQL Agent (s)"),
        ("Forecast", "Forecast Agent (s)"),
        ("Explanation", "Explanation Agent (s)"),
    ]:
        out_time[title] = (
            piv_time[a].map(lambda x: fmt_num(x, 2))
            if a in piv_time.columns else "–"
        )
    save(ordered(out_time), "PL_B2a_agent_time_compact.csv")

    # B.2b — token
    piv_tok = tmp.pivot_table(
        index="Bộ dữ liệu – Mô hình",
        columns="Agent",
        values="Tổng token",
        aggfunc="first",
    ).reset_index()

    out_tok = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": piv_tok["Bộ dữ liệu – Mô hình"]
    })
    for a, title in [
        ("Router", "Router Agent"),
        ("SQL", "SQL Agent"),
        ("Forecast", "Forecast Agent"),
        ("Explanation", "Explanation Agent"),
    ]:
        out_tok[title] = (
            piv_tok[a].map(lambda x: fmt_num(x, 2))
            if a in piv_tok.columns else "–"
        )
    save(ordered(out_tok), "PL_B2b_agent_token_compact.csv")

def build_pl_b3():
    """
    B.3: hiệu năng toàn bộ quy trình.
    Nguồn: PL_A3_e2e_performance.csv
    """
    df = read_csv(APPENDIX_DIR / "PL_A3_e2e_performance.csv")

    n = find_col(df, "Số trường hợp")
    avg_t = find_col(df, "Thời gian end-to-end TB (s)")
    med_t = find_col(df, "Thời gian end-to-end trung vị (s)")
    min_t = find_col(df, "Thời gian nhỏ nhất (s)")
    max_t = find_col(df, "Thời gian lớn nhất (s)")
    avg_tok = find_col(df, "Tổng token TB/case")
    med_tok = find_col(df, "Tổng token trung vị/case")

    out = pd.DataFrame({
        "Bộ dữ liệu – Mô hình": id_series(df),
        "Số trường hợp": df[n].map(fmt_int),
        "Thời gian TB (s)": df[avg_t].map(lambda x: fmt_num(x, 2)),
        "Trung vị (s)": df[med_t].map(lambda x: fmt_num(x, 2)),
        "Nhỏ nhất (s)": df[min_t].map(lambda x: fmt_num(x, 2)),
        "Lớn nhất (s)": df[max_t].map(lambda x: fmt_num(x, 2)),
        "Tổng token TB/case": df[avg_tok].map(lambda x: fmt_num(x, 2)),
        "Tổng token trung vị/case": df[med_tok].map(lambda x: fmt_num(x, 2)),
    })
    save(ordered(out), "PL_B3_e2e_performance_compact.csv")

def build_pl_b4():
    """
    B.4: kiểm định Wilcoxon.
    Nguồn: PL_A4_paired_wilcoxon.csv
    """
    df = read_csv(APPENDIX_DIR / "PL_A4_paired_wilcoxon.csv")

    dcol = find_col(df, "Bộ dữ liệu")
    metric = find_col(df, "Chỉ số")
    n = find_col(df, "Số cặp chung hợp lệ")
    gemini_mean = find_col(df, "Gemini 2.5 Pro - trung bình")
    gemma_mean = find_col(df, "Gemma4 (gemma4:e4b) - trung bình")
    gw = find_col(df, "Gemini tốt hơn")
    mw = find_col(df, "Gemma4 tốt hơn")
    ties = find_col(df, "Hòa")
    stat = find_col(df, "Thống kê Wilcoxon")
    pval = find_col(df, "p-value")
    conclusion = find_col(df, "Kết luận (alpha=0.05)")

    out = pd.DataFrame({
        "Bộ dữ liệu – Chỉ số":
            df[dcol].map(canonical_dataset) + " – " + df[metric].map(canonical_metric),
        "Số cặp": df[n].map(fmt_int),
        "Gemini 2.5 Pro TB": df[gemini_mean].map(lambda x: fmt_num(x, 6)),
        "Gemma4 TB": df[gemma_mean].map(lambda x: fmt_num(x, 6)),
        "Gemini tốt hơn": df[gw].map(fmt_int),
        "Gemma4 tốt hơn": df[mw].map(fmt_int),
        "Hòa": df[ties].map(fmt_int),
        "Thống kê Wilcoxon": df[stat].map(lambda x: fmt_num(x, 1)),
        "p-value": df[pval].map(lambda x: fmt_num(x, 6)),
        "Kết luận": df[conclusion].astype(str),
    })

    order = ["M5 – MAE", "M5 – RMSE", "HCMC Bus – MAE", "HCMC Bus – RMSE"]
    rank = {v: i for i, v in enumerate(order)}
    out["_order"] = out["Bộ dữ liệu – Chỉ số"].map(rank).fillna(999)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    save(out, "PL_B4_paired_wilcoxon_compact.csv")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_ch5_overall()
    build_ch5_router_sql()
    build_ch5_forecast()
    build_ch5_explanation()
    build_ch5_performance()

    build_pl_b1()
    build_pl_b2()
    build_pl_b3()
    build_pl_b4()

    print("\nDONE")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
