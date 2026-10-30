from _common import *

# 1) Tự tính chi tiết theo granularity từ forecast_per_case của từng model.
detail=[]
for mk in ["gemini","gemma"]:
    df=forecast_per_case(mk)
    ds=col(df,["dataset","dataset_code"])
    gran=col(df,["granularity"])
    mae=col(df,["mae","MAE"])
    rmse=col(df,["rmse","RMSE"])
    valid=col(df,["forecast_valid","is_valid","valid_forecast","valid"],required=False)
    eligible=col(df,["evaluation_eligible","forecast_eligible","eligible"],required=False)

    work=df.copy()
    if eligible: work=work[boolish(work[eligible])].copy()
    if valid: work=work[boolish(work[valid])].copy()
    work=work[pd.to_numeric(work[mae],errors="coerce").notna() & pd.to_numeric(work[rmse],errors="coerce").notna()].copy()
    work[mae]=pd.to_numeric(work[mae],errors="coerce")
    work[rmse]=pd.to_numeric(work[rmse],errors="coerce")

    for (dataset,g),q in work.groupby([ds,gran]):
        detail.append({
            "dataset":norm_dataset(dataset),"model_display":MODEL_DISPLAY[mk],"granularity":g,
            "n":len(q),"MAE":q[mae].mean(),"RMSE":q[rmse].mean()
        })

det=pd.DataFrame(detail)
save_support(det,"03_forecast_by_granularity_detail.csv")

# 2) Bảng chính: tổng thể từ evaluation_summary + các dòng granularity vừa tái tính.
rows=[]
for mk in ["gemini","gemma"]:
    ev=eval_summary(mk)
    require(ev,["dataset","n_forecast_eligible","n_forecast_valid","forecast_coverage","mean_mae","mean_rmse"],"evaluation_summary")
    for dataset in ["M5","HCMC Bus"]:
        e=ev[ev.dataset==dataset].iloc[0]
        rows.append({
            "Bộ dữ liệu":dataset,"Mô hình":MODEL_DISPLAY[mk],"Phạm vi":"Tổng thể",
            "Số trường hợp đánh giá":int(e.n_forecast_eligible),"Số dự báo hợp lệ":int(e.n_forecast_valid),
            "Tỷ lệ tạo dự báo hợp lệ (%)":pct(e.forecast_coverage),
            "MAE":nround(e.mean_mae,4),"RMSE":nround(e.mean_rmse,4),
        })
        q=det[(det.dataset==dataset)&(det.model_display==MODEL_DISPLAY[mk])].copy()
        order={"day":1,"month":2,"quarter":3,"year":4}
        q["_o"]=q.granularity.map(order)
        for _,r in q.sort_values("_o").iterrows():
            rows.append({
                "Bộ dữ liệu":dataset,"Mô hình":MODEL_DISPLAY[mk],
                "Phạm vi":GRANULARITY_DISPLAY.get(r.granularity,r.granularity),
                "Số trường hợp đánh giá":int(r.n),"Số dự báo hợp lệ":int(r.n),
                "Tỷ lệ tạo dự báo hợp lệ (%)":np.nan,
                "MAE":nround(r.MAE,4),"RMSE":nround(r.RMSE,4),
            })
save_main(pd.DataFrame(rows),"03_forecast_summary.csv")
