from _appendix_summary_common import *

call_rows = []

for dataset, model, root in TRACE_CONFIGS:
    if not root.exists():
        raise FileNotFoundError(root)

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trace_path = case_dir / "trace.json"

        if not trace_path.exists():
            continue

        traces = json.loads(
            trace_path.read_text(encoding="utf-8")
        )

        if not isinstance(traces, list):
            continue

        for t in traces:
            agent = stage_to_agent(t.get("stage"))

            if not agent:
                continue

            call_rows.append({
                "dataset": dataset,
                "model": model,
                "case_id": case_dir.name,
                "agent": agent,
                "stage": t.get("stage"),
                "attempt": t.get("attempt"),
                "status": t.get("status"),
                "llm_time_s": float(t.get("llm_time_s") or 0),
                "input_tokens": float(t.get("input_tokens") or 0),
                "output_tokens": float(t.get("output_tokens") or 0),
                "total_tokens": float(t.get("total_tokens") or 0),
            })

calls = pd.DataFrame(call_rows)

if calls.empty:
    raise RuntimeError("Không tìm thấy trace LLM nào để tổng hợp.")

# Cộng toàn bộ short/long/retry vào đúng Agent trong từng case trước.
case_agent = (
    calls
    .groupby(
        ["dataset", "model", "case_id", "agent"],
        as_index=False
    )
    .agg(
        llm_calls=("stage", "size"),
        error_calls=("status", lambda x: sum(
            str(v).lower() == "error" for v in x
        )),
        llm_time_s=("llm_time_s", "sum"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        total_tokens=("total_tokens", "sum"),
    )
)

summary = (
    case_agent
    .groupby(
        ["dataset", "model", "agent"],
        as_index=False
    )
    .agg(
        cases_invoked=("case_id", "nunique"),
        avg_calls_per_case=("llm_calls", "mean"),
        avg_error_calls_per_case=("error_calls", "mean"),
        avg_time_s=("llm_time_s", "mean"),
        median_time_s=("llm_time_s", "median"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
    )
)

agent_order = {
    "Router Agent": 1,
    "SQL Agent": 2,
    "Forecast Agent": 3,
    "Explanation Agent": 4,
}

summary["_order"] = summary["agent"].map(agent_order)

summary = (
    summary
    .sort_values(["dataset", "model", "_order"])
    .drop(columns="_order")
)

out = summary.rename(columns={
    "dataset": "Bộ dữ liệu",
    "model": "Mô hình",
    "agent": "Agent",
    "cases_invoked": "Số trường hợp Agent được gọi",
    "avg_calls_per_case": "Số lượt gọi LLM TB/case",
    "avg_error_calls_per_case": "Số lượt lỗi TB/case",
    "avg_time_s": "Thời gian gọi LLM TB (s)",
    "median_time_s": "Thời gian gọi LLM trung vị (s)",
    "avg_input_tokens": "Token đầu vào TB",
    "avg_output_tokens": "Token đầu ra TB",
    "avg_total_tokens": "Tổng token TB",
})

for c in [
    "Số lượt gọi LLM TB/case",
    "Số lượt lỗi TB/case",
    "Thời gian gọi LLM TB (s)",
    "Thời gian gọi LLM trung vị (s)",
    "Token đầu vào TB",
    "Token đầu ra TB",
    "Tổng token TB",
]:
    out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

save_appendix(
    out,
    "PL_A2_agent_performance.csv"
)
