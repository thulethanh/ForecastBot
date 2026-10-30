from _common import *
rows=[]
for mk in ["gemini","gemma"]:
    ev=eval_summary(mk); co=conditional_summary(mk)
    require(ev,["dataset","n_cases"],"evaluation_summary")
    require(co,["dataset","n_router_correct","router_exact_match_rate","n_sql_input_after_router_correct","n_sql_correct","sql_accuracy_given_router_correct"],"agent_conditional_summary")
    for ds in ["M5","HCMC Bus"]:
        e=ev[ev.dataset==ds].iloc[0]; c=co[co.dataset==ds].iloc[0]
        rows.append({
            "Bộ dữ liệu":ds,"Mô hình":MODEL_DISPLAY[mk],"Tổng số yêu cầu":int(e.n_cases),
            "Router đúng":int(c.n_router_correct),"Router - Tỷ lệ khớp hoàn toàn (%)":pct(c.router_exact_match_rate),
            "Số trường hợp vào SQL sau Router đúng":int(c.n_sql_input_after_router_correct),
            "SQL đúng":int(c.n_sql_correct),"SQL - Độ chính xác truy xuất có điều kiện (%)":pct(c.sql_accuracy_given_router_correct),
        })
save_main(pd.DataFrame(rows),"02_router_sql_summary.csv")
