from pathlib import Path
from evaluation import runner
from forecastbot.demo_chat.translation import translate

ROOT=Path(__file__).resolve().parents[2]/'outputs/demo_chat'
UNITS={'day':'Day','month':'Month','quarter':'Quarter','year':'Year'}


def question_from_config(c):
    subject='sales' if c['dataset']=='M5' else 'passenger counts'
    filters=', '.join(f'{k}={v}' for k,v in c.get('filters',{}).items() if v)
    unit = c["granularity"] + ("s" if c["horizon"] != 1 else "")
    return f"Forecast {subject} for the {c['dataset']} dataset for the next {c['horizon']} {unit}" + (f", with filters {filters}." if filters else ".")


def execute(store,sid,question,config,parent=None):
    tid=store.begin(sid,question,config,parent)
    try:
        # Same entrypoint/policy as the evaluated Multi-Agent runner. No graph or overrides.
        result=runner.run_case(config['dataset'],config['model'],tid,question)
    except Exception as exc:
        result={'case_id':tid,'question':question,'status':'failed','error':str(exc)}
    store.finish(tid,result)  # Save even if file/chart export fails.
    try:
        runner.write_case(ROOT/'sessions'/sid,result)
    except Exception as exc:
        result['demo_export_error']=str(exc)
        store.finish(tid,result)
    if result.get('status') == 'success' and result.get('explanation_result'):
        try:
            translate(ROOT, result['explanation_result'])
        except OSError:
            pass  # Translation persistence must not invalidate a completed forecast.
    return tid
