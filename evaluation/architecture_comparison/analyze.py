"""Read-only paired architecture comparison; never imports/runs old analysis scripts."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ('multi_agent', 'single_agent')
DATASETS = {'M5': ('m5_exp', 'm5'), 'HCMC_BUS': ('hcmcbus_exp', 'hcmcbus')}
MODELS = {'gemini': 'gemini_2_5_pro', 'gemma': 'gemma4'}
SCORES = ['relevance', 'faithfulness', 'clarity', 'usefulness']
INPUTS = {}


def truth(x):
    return str(x).strip().lower() in {'true', '1', '1.0', 'yes'}


def number(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def read(path, csv=False):
    if not path.exists():
        INPUTS[str(path.relative_to(ROOT))] = {'exists': False}
        return pd.DataFrame() if csv else {}
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = {
        'exists': True, 'sha256': hashlib.sha256(data).hexdigest(),
        'mtime_ns': path.stat().st_mtime_ns,
    }
    return pd.read_csv(path) if csv else json.loads(data)


def indexed(path):
    d = read(path, csv=True)
    if d.empty:
        return {}
    keys = ['dataset', 'case_id'] if 'dataset' in d else ['case_id']
    if d.duplicated(keys).any():
        raise ValueError(f'Duplicate keys: {path}')
    return {tuple(str(row[k]) for k in keys): row for row in d.to_dict('records')}


def valid_explanation(pipeline, judge):
    """Same six gates as thesis_analysis/analyze_explanation_common_valid.py."""
    reasons = []
    checks = {
        'pipeline_status_not_success': pipeline.get('status') == 'success',
        'forecast_invalid': truth(pipeline.get('forecast_valid')),
        'chart_not_success': pipeline.get('chart_status') == 'success',
        'explanation_unavailable': truth(judge.get('explanation_available')),
        'judge_not_success': judge.get('judge_status') == 'success',
        'judge_score_invalid': all(1 <= number(judge.get(k)) <= 5 for k in SCORES),
    }
    for reason, ok in checks.items():
        if not ok:
            reasons.append(reason)
    return not reasons, ';'.join(reasons)


def collect(model):
    rows = []
    for arch in ARCHS:
        single = arch == 'single_agent'
        result_base = ROOT/'evaluation/results'
        artifact_base = ROOT/'outputs'
        if single:
            result_base /= 'single_agent_ablation'
            artifact_base /= 'single_agent_ablation'
        tables = {name: indexed(result_base/model/f'{name}_per_case.csv')
                  for name in ['orchestration', 'sql', 'integrity', 'forecast']}
        judges = indexed(result_base/'llm_judge'/model/'explanation_per_case.csv')
        for ds, (folder, prefix) in DATASETS.items():
            base = artifact_base/folder/MODELS[model]
            pipeline = indexed(base/'summary.csv')
            ids = sorted((ROOT/'evaluation/ground_truth'/prefix/f'{prefix}_forecast_requests').glob('*.json'))
            if not ids:
                raise ValueError(f'No ground-truth case IDs for {ds}')
            for gtpath in ids:
                cid = gtpath.stem
                gt = read(gtpath)
                case = base/'cases'/cid
                result = read(case/'result.json')
                test = read(case/'test.json')
                pr = pipeline.get((ds, cid), pipeline.get((cid,), {}))
                j = judges.get((ds, cid), {})
                f = tables['forecast'].get((ds, cid), {})
                o = tables['orchestration'].get((ds, cid), {})
                s = tables['sql'].get((ds, cid), {})
                h = tables['integrity'].get((ds, cid), {})
                ev, reasons = valid_explanation(pr, j)
                success = result.get('status') == 'success'
                # Missing artifacts must not inherit a success from an old summary.
                ev = ev and success
                if not success:
                    reasons += ';artifact_not_success'
                r = dict(dataset=ds, model=model, architecture=arch, case_id=cid,
                         granularity=gt.get('granularity'), artifact_present=bool(result),
                         pipeline_success=success, summary_present=bool(pr),
                         scoring_present=all([o, s, h, f]), judge_present=bool(j),
                         router_correct=truth(o.get('overall_exact_match')),
                         sql_correct=truth(s.get('execution_accuracy')),
                         history_correct=truth(h.get('history_match')) and not truth(h.get('ground_truth_leakage')),
                         forecast_eligible=truth(f.get('evaluation_eligible')),
                         forecast_valid=truth(f.get('forecast_valid')),
                         fallback_used=truth(f.get('fallback_used')),
                         mae=number(f.get('mae')), rmse=number(f.get('rmse')),
                         explanation_valid=ev, explanation_exclusion_reason=reasons.strip(';'),
                         judge_model=j.get('judge_model', ''),
                         end_to_end_s=number(result.get('timings', {}).get('end_to_end')),
                         total_tokens=number(result.get('total_tokens')),
                         holdout=json.dumps(test.get('test'), sort_keys=True),
                         forecast_score_status=f.get('status', 'missing'),
                         summary_matches_artifact=bool(pr) and pr.get('status') == result.get('status'))
                for k in SCORES:
                    r[k] = number(j.get(k))
                r['overall_score'] = np.mean([r[k] for k in SCORES])
                r['forecast_comparable'] = success and r['forecast_eligible'] and r['forecast_valid'] and all(np.isfinite(r[k]) for k in ['mae', 'rmse'])
                r['strict_upstream'] = r['router_correct'] and r['sql_correct'] and r['history_correct']
                rows.append(r)
    return pd.DataFrame(rows)


def pair_rows(detail):
    keys = ['dataset', 'model', 'case_id']
    a = detail[detail.architecture.eq('multi_agent')].drop(columns='architecture')
    b = detail[detail.architecture.eq('single_agent')].drop(columns='architecture')
    p = a.merge(b, on=keys, suffixes=('_multi', '_single'), validate='one_to_one')
    p['same_holdout'] = p.holdout_multi.eq(p.holdout_single) & ~p.holdout_multi.isin(['null', '[]'])
    p['forecast_common'] = p.forecast_comparable_multi & p.forecast_comparable_single & p.same_holdout
    p['forecast_strict_common'] = p.forecast_common & p.strict_upstream_multi & p.strict_upstream_single
    p['same_judge'] = p.judge_model_multi.eq(p.judge_model_single) & p.judge_model_multi.ne('')
    p['explanation_common'] = p.explanation_valid_multi & p.explanation_valid_single & p.same_judge
    p['explanation_strict_common'] = p.explanation_common & p.strict_upstream_multi & p.strict_upstream_single
    p['performance_common'] = p.pipeline_success_multi & p.pipeline_success_single
    for gate in ['forecast', 'explanation', 'performance']:
        reasons = []
        for _, r in p.iterrows():
            why = []
            for suffix in ['multi', 'single']:
                if gate == 'forecast' and not r[f'forecast_comparable_{suffix}']:
                    why.append(f'{suffix}:pipeline_or_eligibility_or_forecast_invalid')
                elif gate == 'explanation' and not r[f'explanation_valid_{suffix}']:
                    why.append(f"{suffix}:{r[f'explanation_exclusion_reason_{suffix}']}")
                elif gate == 'performance' and not r[f'pipeline_success_{suffix}']:
                    why.append(f'{suffix}:pipeline_not_success_or_missing')
            if gate == 'forecast' and not r.same_holdout:
                why.append('different_or_missing_holdout')
            if gate == 'explanation' and not r.same_judge:
                why.append('different_or_missing_judge_model')
            reasons.append(';'.join(why))
        p[f'{gate}_exclusion_reason'] = reasons
    return p


def paired_stats(g, gate, metric):
    q = g[g[gate]].copy()
    q = q[np.isfinite(q[f'{metric}_multi']) & np.isfinite(q[f'{metric}_single'])]
    a, b = q[f'{metric}_multi'], q[f'{metric}_single']
    diff = b - a
    n = len(q)
    higher = metric in SCORES + ['overall_score']
    statistic = pval = np.nan
    status = 'insufficient_pairs'
    if n >= 2:
        if (diff == 0).all():
            statistic, pval, status = 0.0, 1.0, 'all_ties'
        else:
            statistic, pval = wilcoxon(a, b, alternative='two-sided', method='auto')
            status = 'ok'
    return dict(cohort=gate, metric=metric, n_pairs=n,
                multi_mean=a.mean(), single_mean=b.mean(),
                mean_diff_single_minus_multi=diff.mean(), median_diff_single_minus_multi=diff.median(),
                single_better=int(((b > a) if higher else (b < a)).sum()),
                multi_better=int(((a > b) if higher else (a < b)).sum()), ties=int((a == b).sum()),
                wilcoxon_statistic=statistic, p_value=pval, test_status=status)


def holm(values):
    """Holm correction across all nonmissing tests in this run."""
    out = np.full(len(values), np.nan)
    indices = [i for i, p in enumerate(values) if np.isfinite(p)]
    indices.sort(key=lambda i: values[i])
    last = 0.0
    for rank, i in enumerate(indices):
        last = max(last, min(1.0, (len(indices)-rank)*values[i]))
        out[i] = last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    models = list(dict.fromkeys(args.models))
    detail = pd.concat([collect(m) for m in models], ignore_index=True)
    paired = pair_rows(detail)
    coverage, overall, comparisons = [], [], []
    for (ds, model), g in paired.groupby(['dataset', 'model']):
        complete = bool(g.artifact_present_multi.all() and g.artifact_present_single.all()
                        and g.scoring_present_multi.all() and g.scoring_present_single.all()
                        and g.summary_matches_artifact_multi.all() and g.summary_matches_artifact_single.all())
        judge_complete = complete and bool(g.judge_present_multi.all() and g.judge_present_single.all())
        row = dict(dataset=ds, model=model, n_expected=len(g), artifacts_and_scores_complete=complete,
                   judge_tables_complete=judge_complete)
        for suffix in ['multi', 'single']:
            for key in ['artifact_present', 'pipeline_success', 'scoring_present', 'judge_present', 'explanation_valid']:
                row[f'n_{key}_{suffix}'] = int(g[f'{key}_{suffix}'].sum())
        for gate in ['forecast_common', 'forecast_strict_common', 'explanation_common', 'explanation_strict_common', 'performance_common']:
            row[f'n_{gate}'] = int(g[gate].sum())
        coverage.append(row)
        for gate, metrics in [('forecast_common', ['mae', 'rmse']), ('forecast_strict_common', ['mae', 'rmse']),
                              ('explanation_common', SCORES+['overall_score']),
                              ('explanation_strict_common', ['overall_score']),
                              ('performance_common', ['end_to_end_s', 'total_tokens'])]:
            for metric in metrics:
                comparisons.append(dict(dataset=ds, model=model,
                                        provisional=not (judge_complete if gate.startswith('explanation') else complete),
                                        **paired_stats(g, gate, metric)))
    for (ds, model, arch), g in detail.groupby(['dataset', 'model', 'architecture']):
        row = dict(dataset=ds, model=model, architecture=arch, n_expected=len(g),
                   n_artifacts=int(g.artifact_present.sum()), n_pipeline_success=int(g.pipeline_success.sum()),
                   n_scored=int(g.scoring_present.sum()), n_judged=int(g.judge_present.sum()))
        for k in ['router_correct', 'sql_correct', 'history_correct', 'forecast_comparable', 'explanation_valid']:
            row[f'n_{k}'] = int(g[k].sum())
            row[f'{k}_rate_expected'] = g[k].mean()
        overall.append(row)
    stats = pd.DataFrame(comparisons)
    stats['p_holm_all_tests_in_run'] = holm(stats.p_value.tolist())
    granularity = []
    for (ds, model, gran), g in paired.groupby(['dataset', 'model', 'granularity_multi']):
        for metric in ['mae', 'rmse']:
            r = paired_stats(g, 'forecast_common', metric)
            # Descriptive breakdown only; no extra tests in this table.
            for k in ['p_value', 'wilcoxon_statistic', 'test_status']:
                r.pop(k)
            granularity.append(dict(dataset=ds, model=model, granularity=gran, **r))
    out = Path(__file__).parent/'outputs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    tables = {
        '01_coverage': pd.DataFrame(coverage), '02_overall': pd.DataFrame(overall),
        '03_paired_comparison': stats, '04_forecast_by_granularity': pd.DataFrame(granularity),
        '05_pair_audit': paired.drop(columns=['holdout_multi', 'holdout_single']),
        '06_case_audit': detail.drop(columns='holdout'),
        '07_excluded_explanation': paired[~paired.explanation_common].drop(columns=['holdout_multi', 'holdout_single']),
    }
    for name, frame in tables.items():
        frame.to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(out/'Architecture_Comparison.xlsx', engine='openpyxl') as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    script = Path(__file__)
    (out/'manifest.json').write_text(json.dumps(dict(models=models, created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), inputs=INPUTS), indent=2), encoding='utf-8')
    lines = ['# So sánh single-agent và multi-agent ForecastBot', '',
        'Ghép theo dataset + model + case_id. Hiệu số = single − multi. MAE/RMSE/thời gian/token thấp hơn là tốt hơn; điểm giải thích cao hơn là tốt hơn.', '',
        '## Phạm vi và độ đầy đủ', '']
    for r in coverage:
        lines.append(f"- {r['dataset']} / {r['model']}: artifact single {r['n_artifact_present_single']}/{r['n_expected']}; cặp forecast {r['n_forecast_common']}; cặp explanation {r['n_explanation_common']}; đầy đủ artifact/score: {r['artifacts_and_scores_complete']}; bảng judge đầy đủ: {r['judge_tables_complete']}.")
    lines += ['', '## Các so sánh chính', '']
    for r in comparisons:
        if r['cohort'] not in ['forecast_common', 'explanation_common', 'performance_common'] or r['metric'] in SCORES:
            continue
        lines.append(f"- {r['dataset']} / {r['model']} / {r['metric']}: N={r['n_pairs']}; multi={r['multi_mean']:.4f}; single={r['single_mean']:.4f}; chênh lệch={r['mean_diff_single_minus_multi']:.4f}; tạm thời={r['provisional']}.")
    lines += ['', '## Điều kiện cân nhắc đưa vào luận văn', '',
        '- Bộ còn thiếu artifact hoặc điểm chấm chỉ là phân tích tạm thời. Không tính case chưa chạy thành bằng chứng kiến trúc kém hơn.',
        '- Báo cáo coverage trên toàn bộ tập câu hỏi cùng điểm trên tập cặp hợp lệ; common-valid có thiên lệch lựa chọn về các ca thành công.',
        '- Explanation dùng cùng sáu điều kiện của script cũ; bổ sung xác nhận artifact thành công và cùng model judge. Đây là hợp lệ vận hành, không bảo đảm Router/SQL đúng. Xem cohort strict để yêu cầu Router, SQL và history đều đúng ở cả hai phía.',
        '- Forecast ghép các ca cả hai bên đủ điều kiện, dự báo hợp lệ, sai số hữu hạn và cùng holdout. Fallback được giữ; xem cột fallback trong audit.',
        '- Không gộp MAE/RMSE giữa hai dataset. Thời gian E2E lấy từ result.json, không cộng suy diễn thời gian LLM. Token thiếu giữ là NaN.',
        '- Wilcoxon hai phía mang tính thăm dò; p Holm hiệu chỉnh toàn bộ kiểm định có p hợp lệ trong lần chạy này. Các câu hỏi có thể dùng chuỗi chồng lấn, nên không mặc nhiên xem các cặp độc lập.',
        '- Bản single-agent hợp nhất phân tích yêu cầu và SQL, nhưng vẫn có nhiều lần gọi LLM. Khác biệt prompt, số lượt gọi, fallback và lịch sử đầu vào cần được thảo luận; kết quả không cô lập hoàn toàn tác động của số agent.',
        '- Chỉ so sánh E2E trong điều kiện phần cứng/API/tải tương đương; số token không trực tiếp là chi phí tiền.',
        '- Phân tích đọc bảng scoring/judge hiện có, không xác minh lại toàn bộ phép chấm hay cache. Phải chạy lại scoring sau khi runner thay đổi; summary/artifact khớp trạng thái chưa đủ bảo đảm score còn mới. Manifest lưu hash nguồn để truy vết.',
        '- Không kết luận multi-agent tốt hơn toàn diện chỉ từ một bảng trung bình. Dùng N, coverage, hiệu số ghép cặp và giới hạn thiết kế cùng nhau.',
        '', 'Chi tiết từng cặp và lý do loại: 05_pair_audit.csv, 07_excluded_explanation.csv. Không có file cũ nào được ghi lại.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False))
    print('Output:', out)


if __name__ == '__main__':
    main()
