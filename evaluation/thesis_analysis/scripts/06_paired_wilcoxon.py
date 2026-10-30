from _common import *
from scipy.stats import wilcoxon
rows=[]
def prep(mk):
    df=forecast_per_case(mk)
    case=col(df,["case_id","Case_ID","id"]); ds=col(df,["dataset","dataset_code"])
    mae=col(df,["mae","MAE"]); rmse=col(df,["rmse","RMSE"])
    valid=col(df,["forecast_valid","is_valid","valid_forecast","valid"],required=False)
    eligible=col(df,["evaluation_eligible","forecast_eligible","eligible"],required=False)
    q=df.copy()
    if eligible:q=q[boolish(q[eligible])]
    if valid:q=q[boolish(q[valid])]
    q=q[[case,ds,mae,rmse]].copy(); q.columns=["case_id","dataset","mae","rmse"]
    q["dataset"]=q.dataset.map(norm_dataset)
    q["mae"]=pd.to_numeric(q.mae,errors="coerce"); q["rmse"]=pd.to_numeric(q.rmse,errors="coerce")
    return q.dropna()
g=prep("gemini"); m=prep("gemma")
p=g.merge(m,on=["case_id","dataset"],suffixes=("_g","_m"))
for ds in ["M5","HCMC Bus"]:
    d=p[p.dataset==ds]
    for metric in ["mae","rmse"]:
        a=d[f"{metric}_g"]; b=d[f"{metric}_m"]
        stat,pv=wilcoxon(a,b,alternative="two-sided")
        rows.append({"dataset":ds,"metric":metric.upper(),"n_pairs":len(d),
                     "gemini_mean":a.mean(),"gemma_mean":b.mean(),
                     "gemini_better":int((a<b).sum()),"gemma_better":int((b<a).sum()),"ties":int((a==b).sum()),
                     "wilcoxon_statistic":stat,"p_value":pv})
out=pd.DataFrame(rows)
save_support(out,"06_paired_wilcoxon.csv")
print(out.to_string(index=False))
