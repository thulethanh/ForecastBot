from _common import *

call_rows=[]; case_rows=[]
for dataset,model,root in TRACE_CONFIGS:
    if not root.exists(): raise FileNotFoundError(root)
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        tp=case_dir/"trace.json"
        if not tp.exists(): continue
        traces=json.loads(tp.read_text(encoding="utf-8"))
        if not isinstance(traces,list): continue
        llm_sum=0; inp=out=tot=0
        for t in traces:
            agent=stage_to_agent(t.get("stage"))
            if not agent: continue
            tm=float(t.get("llm_time_s") or 0); i=float(t.get("input_tokens") or 0); o=float(t.get("output_tokens") or 0); tt=float(t.get("total_tokens") or 0)
            llm_sum+=tm; inp+=i; out+=o; tot+=tt
            call_rows.append({"dataset":dataset,"model":model,"case_id":case_dir.name,"agent":agent,"stage":t.get("stage"),"attempt":t.get("attempt"),"status":t.get("status"),"llm_time_s":tm,"input_tokens":i,"output_tokens":o,"total_tokens":tt})
        e2e,source=extract_e2e_time(case_dir,llm_sum)
        case_rows.append({"dataset":dataset,"model":model,"case_id":case_dir.name,"e2e_time_s":e2e,"e2e_source":source,"llm_time_s":llm_sum,"input_tokens":inp,"output_tokens":out,"total_tokens":tot})

calls=pd.DataFrame(call_rows); cases=pd.DataFrame(case_rows)
save_support(calls,"05_agent_llm_calls.csv")
save_support(cases,"05_e2e_per_case.csv")

# Agent per-case then summary
ca=calls.groupby(["dataset","model","case_id","agent"],as_index=False).agg(
    llm_calls=("stage","size"),llm_time_s=("llm_time_s","sum"),
    input_tokens=("input_tokens","sum"),output_tokens=("output_tokens","sum"),total_tokens=("total_tokens","sum"))
save_support(ca,"05_agent_per_case.csv")

ag=ca.groupby(["dataset","model","agent"],as_index=False).agg(
    cases=("case_id","nunique"),avg_time_s=("llm_time_s","mean"),median_time_s=("llm_time_s","median"),
    avg_input_tokens=("input_tokens","mean"),avg_output_tokens=("output_tokens","mean"),avg_total_tokens=("total_tokens","mean"))
save_support(ag,"05_agent_performance_detail.csv")

# End-to-end summary. Exact E2E is mandatory for thesis table.
bad=cases[cases.e2e_time_s.isna()]
if len(bad):
    sample=", ".join(bad.case_id.head(10).astype(str))
    raise RuntimeError(
        f"Không tìm thấy exact E2E time cho {len(bad)} case (ví dụ {sample}). "
        "Script cố ý không dùng tổng llm_time_s thay thế E2E. "
        "Hãy gửi 1 metadata.json/result.json của case để bổ sung key chính xác."
    )

e2e=cases.groupby(["dataset","model"],as_index=False).agg(
    cases=("case_id","nunique"),avg_e2e_time_s=("e2e_time_s","mean"),median_e2e_time_s=("e2e_time_s","median"),
    avg_input_tokens=("input_tokens","mean"),avg_output_tokens=("output_tokens","mean"),avg_total_tokens=("total_tokens","mean"))
save_support(e2e,"05_e2e_performance_detail.csv")

rows=[]
for _,r in e2e.iterrows():
    rows.append({"Bộ dữ liệu":r.dataset,"Mô hình":r.model,"Phạm vi":"Toàn hệ thống","Thành phần":"ForecastBot end-to-end",
                 "Số trường hợp":int(r.cases),"Thời gian trung bình (s)":nround(r.avg_e2e_time_s,2),
                 "Thời gian trung vị (s)":nround(r.median_e2e_time_s,2),"Token đầu vào TB":nround(r.avg_input_tokens,2),
                 "Token đầu ra TB":nround(r.avg_output_tokens,2),"Tổng token TB":nround(r.avg_total_tokens,2)})
order={"Router Agent":1,"SQL Agent":2,"Forecast Agent":3,"Explanation Agent":4}
ag["_o"]=ag.agent.map(order)
for _,r in ag.sort_values(["dataset","model","_o"]).iterrows():
    rows.append({"Bộ dữ liệu":r.dataset,"Mô hình":r.model,"Phạm vi":"Theo Agent","Thành phần":r.agent,
                 "Số trường hợp":int(r.cases),"Thời gian trung bình (s)":nround(r.avg_time_s,2),
                 "Thời gian trung vị (s)":nround(r.median_time_s,2),"Token đầu vào TB":nround(r.avg_input_tokens,2),
                 "Token đầu ra TB":nround(r.avg_output_tokens,2),"Tổng token TB":nround(r.avg_total_tokens,2)})
save_main(pd.DataFrame(rows),"05_performance_summary.csv")
