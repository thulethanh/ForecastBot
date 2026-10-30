from _appendix_summary_common import *

rows = []

for dataset, model, root in TRACE_CONFIGS:
    if not root.exists():
        raise FileNotFoundError(root)

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metadata_path = case_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        obj = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        timings = obj.get("timings", {})

        if not isinstance(timings, dict):
            raise RuntimeError(
                f"timings không hợp lệ: {metadata_path}"
            )

        e2e = timings.get("end_to_end")

        if e2e is None:
            raise RuntimeError(
                f"Thiếu timings.end_to_end: {metadata_path}"
            )

        rows.append({
            "dataset": dataset,
            "model": model,
            "case_id": case_dir.name,
            "status": obj.get("status", ""),
            "e2e_time_s": float(e2e),
            "total_tokens": pd.to_numeric(
                obj.get("total_tokens"),
                errors="coerce"
            ),
        })

case_df = pd.DataFrame(rows)

if case_df.empty:
    raise RuntimeError("Không tìm thấy metadata case.")

summary = (
    case_df
    .groupby(["dataset", "model"], as_index=False)
    .agg(
        n_cases=("case_id", "nunique"),
        avg_e2e_time_s=("e2e_time_s", "mean"),
        median_e2e_time_s=("e2e_time_s", "median"),
        min_e2e_time_s=("e2e_time_s", "min"),
        max_e2e_time_s=("e2e_time_s", "max"),
        avg_total_tokens=("total_tokens", "mean"),
        median_total_tokens=("total_tokens", "median"),
    )
)

out = summary.rename(columns={
    "dataset": "Bộ dữ liệu",
    "model": "Mô hình",
    "n_cases": "Số trường hợp",
    "avg_e2e_time_s": "Thời gian end-to-end TB (s)",
    "median_e2e_time_s": "Thời gian end-to-end trung vị (s)",
    "min_e2e_time_s": "Thời gian nhỏ nhất (s)",
    "max_e2e_time_s": "Thời gian lớn nhất (s)",
    "avg_total_tokens": "Tổng token TB/case",
    "median_total_tokens": "Tổng token trung vị/case",
})

for c in [
    "Thời gian end-to-end TB (s)",
    "Thời gian end-to-end trung vị (s)",
    "Thời gian nhỏ nhất (s)",
    "Thời gian lớn nhất (s)",
    "Tổng token TB/case",
    "Tổng token trung vị/case",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

save_appendix(
    out,
    "PL_A3_e2e_performance.csv"
)
