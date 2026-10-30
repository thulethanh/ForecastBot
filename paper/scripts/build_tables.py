"""Recompute CSA2026 Tables 2–6 from saved case-level scores and artifacts.

No LLM, database, network, or manuscript access. Table 1 is not generated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from evaluation.architecture_comparison import analyze as source


# Presentation schema only; all result cells are computed below.
TABLE_COLUMNS = {
    2: ['Dataset – model', 'Router', 'SQL', 'Valid forecast', 'Expl.', 'E2E'],
    3: ['Dataset – model', 'Cases', 'Fallback', 'MAE', 'RMSE', 'MAE w/o fallback'],
    4: ['Dataset – model', 'Relevance', 'Faithfulness', 'Clarity', 'Usefulness', 'Mean'],
    5: ['Dataset – model', 'Router (M / S)', 'SQL (M / S)', 'Valid forecast (M / S)', 'Completion (M / S)'],
    6: ['Dataset – model', 'MAE (M / S)', 'Expl. (M / S)', 'E2E s (M / S)', 'Tokens (M / S)'],
}


def pct(numerator, denominator):
    if denominator == 0:
        raise ValueError('Empty denominator')
    return 100 * numerator / denominator


def label(ds, model):
    return ('M5' if ds == 'M5' else 'Bus') + ' – ' + ('Gemini' if model == 'gemini' else 'Gemma 4')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'paper/outputs')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source.INPUTS.clear()
    detail = pd.concat([source.collect(m) for m in source.MODELS], ignore_index=True)
    for keys, g in detail.groupby(['dataset', 'model', 'architecture']):
        if len(g) != 63 or g.case_id.duplicated().any():
            raise ValueError(f'Invalid scenario coverage: {keys}')
        for field in ['artifact_present','summary_present','scoring_present','judge_present','summary_matches_artifact']:
            if not g[field].all():
                raise ValueError(f'Incomplete or inconsistent input: {keys}/{field}')
    pairs = source.pair_rows(detail)
    stats_rows = []
    for (ds, model), g in pairs.groupby(['dataset','model']):
        for gate, metrics in [('forecast_common',['mae','rmse']), ('forecast_strict_common',['mae','rmse']),
                              ('explanation_common',source.SCORES+['overall_score']),
                              ('explanation_strict_common',['overall_score']),
                              ('performance_common',['end_to_end_s','total_tokens'])]:
            for metric in metrics:
                stats_rows.append(dict(dataset=ds,model=model,**source.paired_stats(g,gate,metric)))
    stats = pd.DataFrame(stats_rows)
    if len(stats) != 48 or not np.isfinite(stats.p_value).all():
        raise ValueError('Paper requires all 48 valid tests for the Holm family')
    stats['p_holm'] = source.holm(stats.p_value.tolist())
    multi = detail[detail.architecture.eq('multi_agent')].copy()
    # Paper Table 3 counts deterministic fallback only; branch recovery stays LLM-based.
    modes = []
    for r in multi.itertuples():
        folder = source.DATASETS[r.dataset][0]
        path = ROOT/'outputs'/folder/source.MODELS[r.model]/'cases'/r.case_id/'forecast.json'
        artifact = source.read(path)
        modes.append(artifact.get('selected_mode', ''))
        if r.forecast_eligible and r.forecast_valid:
            if artifact.get('selected_mode') not in {'ensemble','single','short_only','long_only','deterministic_recent_mean'}:
                raise ValueError(f'Unknown forecast provenance: {path}')
            if not isinstance(artifact.get('fallback_used'), bool) or artifact['fallback_used'] != r.fallback_used:
                raise ValueError(f'Artifact/scorer fallback conflict: {path}')
    multi['selected_mode'] = modes
    multi['deterministic_fallback'] = multi.selected_mode.eq('deterministic_recent_mean')
    common_ids = {}
    for ds in source.DATASETS:
        a = multi[(multi.dataset==ds)&(multi.model=='gemini')].set_index('case_id')
        b = multi[(multi.dataset==ds)&(multi.model=='gemma')].set_index('case_id')
        valid = a.explanation_valid & b.explanation_valid & a.judge_model.eq(b.judge_model) & a.judge_model.ne('')
        common_ids[ds] = set(valid[valid].index)
    stage, forecast, explanation, audits = [], [], [], []
    for ds in ['M5','HCMC_BUS']:
        for model in ['gemini','gemma']:
            g = multi[(multi.dataset==ds)&(multi.model==model)]
            router = g.router_correct
            eligible = g.forecast_eligible
            cond = router & eligible
            valid = cond & g.forecast_valid
            q = g[valid]
            if not np.isfinite(q[['mae','rmse']].to_numpy()).all():
                raise ValueError('Nonfinite forecast error')
            e = g[g.case_id.isin(common_ids[ds])]
            means = e[source.SCORES].mean()
            score = e.overall_score.mean()
            stage.append([label(ds,model),f'{pct(router.sum(),len(g)):.2f}',
                          f'{pct((router & g.sql_correct).sum(),router.sum()):.2f} / {pct(g.sql_correct.sum(),len(g)):.2f}',
                          f'{pct(valid.sum(),cond.sum()):.2f} / {pct((eligible & g.forecast_valid).sum(),eligible.sum()):.2f}',
                          f'{score:.2f}',f'{pct(g.pipeline_success.sum(),len(g)):.2f}'])
            fallback = int(q.deterministic_fallback.sum())
            fb = '0 (0%)' if fallback == 0 else f'{fallback} ({pct(fallback,len(q)):.1f}%)'
            fmt = ',.0f' if ds=='HCMC_BUS' else '.2f'
            forecast.append([label(ds,model),str(len(q)),fb,format(q.mae.mean(),fmt),format(q.rmse.mean(),fmt),
                             format(q.loc[~q.deterministic_fallback,'mae'].mean(),fmt)])
            explanation.append([label(ds,model),*[f'{means[c]:.2f}' for c in source.SCORES],f'{score:.2f}'])
            audits.append(dict(dataset=ds,model=model,n_total=len(g),n_router=int(router.sum()),
                               n_eligible=int(eligible.sum()),n_forecast_conditional=int(cond.sum()),
                               n_forecast_valid=len(q),n_fallback=fallback,n_any_fallback=int(q.fallback_used.sum()),n_branch_recovery=int((q.fallback_used & ~q.deterministic_fallback).sum()),n_explanation=len(e),
                               mae=q.mae.mean(),rmse=q.rmse.mean(),mae_without_fallback=q.loc[~q.deterministic_fallback,'mae'].mean(),
                               explanation_mean=score))
    architecture, quality = [], []
    for ds in ['HCMC_BUS','M5']:
        for model in ['gemini','gemma']:
            values = []
            for arch in ['multi_agent','single_agent']:
                g = detail[(detail.dataset==ds)&(detail.model==model)&(detail.architecture==arch)]
                upstream = g.strict_upstream & g.forecast_eligible
                values.append([pct(g.router_correct.sum(),len(g)),pct((g.router_correct & g.sql_correct).sum(),g.router_correct.sum()),
                               pct((upstream & g.forecast_valid).sum(),upstream.sum()),pct(g.pipeline_success.sum(),len(g))])
            architecture.append([label(ds,model),*[f'{a:.1f} / {b:.1f}' for a,b in zip(*values)]])
            cells = []
            for gate, metric, fmt in [('forecast_common','mae',',.0f' if ds=='HCMC_BUS' else '.1f'),
                                      ('explanation_common','overall_score','.2f'),
                                      ('performance_common','end_to_end_s','.1f'),('performance_common','total_tokens',',.0f')]:
                r = stats[(stats.dataset==ds)&(stats.model==model)&(stats.cohort==gate)&(stats.metric==metric)].iloc[0]
                cells.append(f'{r.multi_mean:{fmt}} / {r.single_mean:{fmt}}'+('*' if r.p_holm<0.05 else ''))
            quality.append([label(ds,model),*cells])
    rows = dict(zip(range(2, 7), [stage, forecast, explanation, architecture, quality]))
    tables = {i: pd.DataFrame(rows[i], columns=TABLE_COLUMNS[i]) for i in rows}
    for i, table in tables.items():
        table.to_csv(out/f'table_{i}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(out/'cohort_audit.csv',index=False)
    detail.drop(columns='holdout').to_csv(out/'case_audit.csv',index=False)
    multi.drop(columns='holdout').to_csv(out/'multi_agent_provenance.csv',index=False)
    pairs.drop(columns=['holdout_multi','holdout_single']).to_csv(out/'pair_audit.csv',index=False)
    stats.to_csv(out/'paired_tests_48.csv',index=False)
    with pd.ExcelWriter(out/'CSA2026_tables.xlsx',engine='openpyxl') as writer:
        for i,t in tables.items():t.to_excel(writer,sheet_name=f'Table {i}',index=False)
        pd.DataFrame(audits).to_excel(writer,sheet_name='Cohorts',index=False)
        stats.to_excel(writer,sheet_name='48 paired tests',index=False)
    lines=['# CSA2026 tables','', 'Tables 2–6 are computed from saved scores and artifacts without a manuscript. Table 1 (literature overview) is not generated.','']
    for i,t in tables.items():
        lines += [f'## Table {i}','','| '+' | '.join(t.columns)+' |','| '+' | '.join(['---']*len(t.columns))+' |']
        lines += ['| '+' | '.join(map(str,row))+' |' for row in t.itertuples(index=False,name=None)]
        lines += ['']
    lines += ['## Verification','', 'Input coverage and provenance checks passed; no manuscript comparison is performed.',
              'See cohort_audit.csv for denominators and paired_tests_48.csv for unrounded values, paired N, raw p and Holm-adjusted p.',
              'Table 2 SQL: correct Router / all cases. Forecast: correct Router and eligible / all eligible cases. Table 3 retains valid forecasts in the conditional cohort, including fallback.',
              'Table 4 uses shared valid explanations across Gemini/Gemma within a dataset (same judge). Table 5 uses architecture-specific upstream cohorts. Table 6 uses common forecast holdouts, common valid explanations, and successful paired pipelines for performance; finite values are filtered per metric.',
              'IMPORTANT: Table 3 Fallback means deterministic_recent_mean only. MAE w/o fallback retains short_only/long_only LLM branch recovery. The broader fallback_used flag counts both kinds; cohort_audit.csv records both. The paper should explicitly define this distinction; Gemini does use branch recovery despite zero deterministic fallback.',
              'Stars use two-sided Wilcoxon and Holm over all 48 tests, including strict-cohort and criterion tests. No hard-coded numerical paper results are used.',
              'Narrative-only Chronos benchmark and latency percentages are outside these experimental tables and are not reproduced by this script.']
    (out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(generated_tables=list(tables),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),inputs=source.INPUTS),indent=2))
    print(f'Output: {out}\nGenerated Tables 2–6; input coverage and provenance checks passed.')
    if (out/'table_1.csv').exists() or (out/'paper_cell_comparison.csv').exists():
        print('Legacy Table 1/manuscript comparison files remain in this directory; they were not regenerated.')


if __name__=='__main__':main()
