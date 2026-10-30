from _common import *
rows=[]
for mk in ["gemini","gemma"]:
    ex=explanation_summary(mk)
    require(ex,["dataset","n_cases","n_judged_success","mean_relevance","mean_faithfulness","mean_clarity","mean_usefulness","mean_overall_score"],"explanation_summary")
    for ds in ["M5","HCMC Bus"]:
        r=ex[ex.dataset==ds].iloc[0]
        rows.append({
            "Bộ dữ liệu":ds,"Mô hình":MODEL_DISPLAY[mk],"Số trường hợp":int(r.n_cases),
            "Số giải thích được chấm":int(r.n_judged_success),
            "Tính phù hợp":nround(r.mean_relevance,2),"Tính trung thực":nround(r.mean_faithfulness,2),
            "Tính rõ ràng":nround(r.mean_clarity,2),"Tính hữu ích":nround(r.mean_usefulness,2),
            "Điểm trung bình":nround(r.mean_overall_score,2),
        })
save_main(pd.DataFrame(rows),"04_explanation_summary.csv")
