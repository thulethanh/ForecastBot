from _appendix_summary_common import *

rows = []

for mk in ["gemini", "gemma"]:
    df = forecast_per_case(mk).copy()

    ds = col(df, ["dataset", "dataset_code"])
    gran = col(df, ["granularity"])
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

    q = df.copy()

    if eligible:
        q = q[boolish(q[eligible])].copy()

    if valid:
        q = q[boolish(q[valid])].copy()

    q[mae] = pd.to_numeric(q[mae], errors="coerce")
    q[rmse] = pd.to_numeric(q[rmse], errors="coerce")
    q = q[q[mae].notna() & q[rmse].notna()].copy()

    for (dataset, g), d in q.groupby([ds, gran]):
        rows.append({
            "Bộ dữ liệu": norm_dataset(dataset),
            "Mô hình": MODEL_DISPLAY[mk],
            "Đơn vị thời gian": {
                "day": "Ngày",
                "month": "Tháng",
                "quarter": "Quý",
                "year": "Năm",
            }.get(str(g), str(g)),
            "Số dự báo hợp lệ": int(len(d)),
            "MAE trung bình": round(float(d[mae].mean()), 4),
            "MAE trung vị": round(float(d[mae].median()), 4),
            "RMSE trung bình": round(float(d[rmse].mean()), 4),
            "RMSE trung vị": round(float(d[rmse].median()), 4),
        })

order = {"Ngày": 1, "Tháng": 2, "Quý": 3, "Năm": 4}

out = pd.DataFrame(rows)
out["_order"] = out["Đơn vị thời gian"].map(order)

out = (
    out
    .sort_values(["Bộ dữ liệu", "Mô hình", "_order"])
    .drop(columns="_order")
)

save_appendix(
    out,
    "PL_A1_forecast_by_granularity.csv"
)
