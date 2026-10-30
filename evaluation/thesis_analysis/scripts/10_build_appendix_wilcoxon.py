from _appendix_summary_common import *
from scipy.stats import wilcoxon

def prepare(model_key):
    df = forecast_per_case(model_key).copy()

    case_id = col(df, ["case_id", "Case_ID", "id"])
    dataset = col(df, ["dataset", "dataset_code"])
    mae = col(df, ["mae", "MAE"])
    rmse = col(df, ["rmse", "RMSE"])

    eligible = first_existing(
        df,
        ["evaluation_eligible", "forecast_eligible", "eligible"]
    )

    valid = first_existing(
        df,
        ["forecast_valid", "is_valid", "valid_forecast", "valid"]
    )

    if eligible:
        df = df[boolish(df[eligible])].copy()

    if valid:
        df = df[boolish(df[valid])].copy()

    q = df[[case_id, dataset, mae, rmse]].copy()
    q.columns = ["case_id", "dataset", "mae", "rmse"]

    q["dataset"] = q["dataset"].map(norm_dataset)
    q["mae"] = pd.to_numeric(q["mae"], errors="coerce")
    q["rmse"] = pd.to_numeric(q["rmse"], errors="coerce")

    return q.dropna()

gemini = prepare("gemini")
gemma = prepare("gemma")

paired = gemini.merge(
    gemma,
    on=["case_id", "dataset"],
    suffixes=("_gemini", "_gemma"),
    how="inner",
)

rows = []

for dataset in ["M5", "HCMC Bus"]:
    d = paired[paired["dataset"] == dataset].copy()

    for metric in ["mae", "rmse"]:
        a = d[f"{metric}_gemini"].astype(float)
        b = d[f"{metric}_gemma"].astype(float)

        statistic, p_value = wilcoxon(
            a,
            b,
            alternative="two-sided",
        )

        rows.append({
            "Bộ dữ liệu": dataset,
            "Chỉ số": metric.upper(),
            "Số cặp chung hợp lệ": int(len(d)),
            "Gemini 2.5 Pro - trung bình": round(float(a.mean()), 6),
            "Gemma4 (gemma4:e4b) - trung bình": round(float(b.mean()), 6),
            "Gemini tốt hơn": int((a < b).sum()),
            "Gemma4 tốt hơn": int((b < a).sum()),
            "Hòa": int((a == b).sum()),
            "Thống kê Wilcoxon": round(float(statistic), 6),
            "p-value": round(float(p_value), 6),
            "Kết luận (alpha=0.05)": (
                "Có ý nghĩa thống kê"
                if p_value < 0.05
                else "Chưa có ý nghĩa thống kê"
            ),
        })

out = pd.DataFrame(rows)

save_appendix(
    out,
    "PL_A4_paired_wilcoxon.csv"
)
