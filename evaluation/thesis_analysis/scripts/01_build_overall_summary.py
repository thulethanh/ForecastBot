from _common import *
rows=[]
for mk in ["gemini","gemma"]:
    ev=eval_summary(mk); co=conditional_summary(mk); ex=explanation_summary(mk)
    require(ev,["dataset","end_to_end_success_rate","forecast_coverage"],"evaluation_summary")
    require(co,["dataset","router_exact_match_rate","sql_accuracy_given_router_correct"],"agent_conditional_summary")
    require(ex,["dataset","mean_overall_score"],"explanation_summary")
    for ds in ["M5","HCMC Bus"]:
        e=ev[ev.dataset==ds].iloc[0]; c=co[co.dataset==ds].iloc[0]; x=ex[ex.dataset==ds].iloc[0]
        rows.append({
            "Bộ dữ liệu":ds,"Mô hình":MODEL_DISPLAY[mk],
            "Router - Tỷ lệ khớp hoàn toàn (%)":pct(c.router_exact_match_rate),
            "SQL - Độ chính xác truy xuất có điều kiện (%)":pct(c.sql_accuracy_given_router_correct),
            "Forecast - Tỷ lệ tạo dự báo hợp lệ (%)":pct(e.forecast_coverage),
            "Explanation - Điểm trung bình (1-5)":nround(x.mean_overall_score,2),
            "ForecastBot - Tỷ lệ hoàn thành toàn bộ quy trình (%)":pct(e.end_to_end_success_rate),
        })
save_main(pd.DataFrame(rows),"01_overall_summary.csv")
