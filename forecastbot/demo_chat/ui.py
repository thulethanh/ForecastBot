import json
from pathlib import Path
import pandas as pd
import streamlit as st
from forecastbot.demo_chat.store import Store
from forecastbot.demo_chat.translation import read_translation, translate, source_text
from forecastbot.demo_chat.service import ROOT, UNITS, runner, question_from_config, execute

MODELS={'gemini':'Gemini 2.5 Pro','gemma':'Gemma4 (gemma4:e4b)'}


from forecastbot.demo_chat.filters import FIELDS, load_catalog, choices

@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def filter_catalog(dataset):
    return load_catalog(dataset)


def render_result(turn):
    r=turn['result']; req=r.get('forecast_request',{})
    if turn['status']=='running':
        st.info('This request is running or was interrupted. It will not restart automatically.')
        return
    if turn['status']=='interrupted':
        st.warning('This run was marked as interrupted; no completed result is available.')
        return
    if turn['status']=='needs_clarification':
        labels={'dataset':'dataset','granularity':'time unit','horizon':'forecast horizon'}
        missing=', '.join(labels.get(k,k) for k in req.get('missing_fields',[]))
        st.info(req.get('user_message_for_clarification') or f'Please provide: {missing}.')
        return  # Do not show technical result/error panels while asking for clarification.
    elif r.get('status')!='success':
        error=r.get('error','Unknown error.')
        if 'Router dataset=' in error: st.error('The dataset in your question differs from your selection. Edit the question or select the matching dataset.')
        elif 'no historical data' in error: st.error('No historical data found. Check or change the filters.')
        else: st.error('Unable to complete the request. Review the error details, edit the question, or check your connections and retry.')
    else:
        fc=r.get('forecast_result',{})
        if fc.get('fallback_used'): st.warning('Forecast fallback used: '+str(fc.get('selected_mode')))
        split=r.get('split_info',{})
        st.caption('Forecast evaluated on the held-out test period.' if split.get('evaluation_eligible') else 'Insufficient data for holdout evaluation: forecasting from the full history.')
        explanation=r.get('explanation_result',{})
        st.markdown('**Tiếng Việt**')
        if isinstance(explanation,dict):
            st.markdown(str(explanation.get('summary_text','')))
            for item in explanation.get('insights',[]) or []: st.write(item)
        else: st.write(explanation)
        st.markdown('**English**')
        translation=read_translation(ROOT, explanation)
        if translation and translation.get('English'):
            st.markdown(translation['English'])
            st.download_button('Download bilingual explanation',
                json.dumps(translation,ensure_ascii=False,indent=2),
                file_name=f"{turn['id']}_bilingual.json",key='translation_download_'+turn['id'])
        elif source_text(explanation):
            st.info((translation or {}).get('error') or 'English translation is not available yet.')
            if st.button('Translate to English',key='translate_'+turn['id']):
                try:
                    with st.spinner('Translating to English…'):
                        translate(ROOT, explanation)
                    st.rerun()
                except OSError:
                    st.error('Unable to save the translation. Check local storage permissions.')
        if r.get('explanation_fallback_used'): st.info('A fallback explanation is shown.')
        points=[]
        hist=r.get('forecast_history',[])
        for x in hist[-180:]: points.append({'Date':x['ds'],'Value':x['y'],'Series':'History'})
        for x in fc.get('forecast',[]): points.append({'Date':x['ds'],'Value':x['yhat'],'Series':'Forecast'})
        for x in r.get('test_artifact',{}).get('test',[]): points.append({'Date':x['ds'],'Value':x['y'],'Series':'Test actuals'})
        if points:
            data=pd.DataFrame(points);data['Date']=pd.to_datetime(data['Date'])
            st.line_chart(data,x='Date',y='Value',color='Series')
        st.dataframe(fc.get('forecast',[]),hide_index=True)
    with st.expander('Processing and error details'):
        st.write('Model:',MODELS[turn['config']['model']])
        st.json(req)
        st.code(r.get('generated_sql',''),language='sql')
        st.json({k:r.get(k) for k in ['status','error','split_info','timings','total_tokens','demo_export_error']})
        st.json(r.get('traces',[]),expanded=False)
    st.download_button('Download results as JSON',json.dumps(r,ensure_ascii=False,default=str,indent=2),file_name=f"{turn['id']}.json",key='download_'+turn['id'])


def main():
    st.set_page_config(page_title='ForecastBot — Experiment Demo',layout='wide')
    store=Store(ROOT)
    if not store.sessions(): store.create()
    if 'demo_sid' not in st.session_state: st.session_state.demo_sid=store.sessions()[0]['id']
    with st.sidebar:
        st.title('Chat sessions')
        if st.button('＋ New session'):
            st.session_state.demo_sid=store.create();st.rerun()
        sessions=store.sessions();titles={r['id']:r['title'] for r in sessions}
        st.subheader('Recent sessions')
        search=st.text_input('Search sessions',key='session_search')
        visible=[r for r in sessions if search.casefold() in r['title'].casefold()]
        for r in visible[:30]:
            if st.button(r['title'],key='nav_'+r['id'],width='stretch',
                         type='primary' if r['id']==st.session_state.demo_sid else 'secondary',
                         help=r['updated']):
                st.session_state.demo_sid=r['id'];st.rerun()
        if len(visible)>30:
            st.caption('Showing the 30 most recent sessions. Search by name to find older sessions.')
        sid=st.session_state.demo_sid
        session=next(r for r in sessions if r['id']==sid)
        title=st.text_input('Session name',value=session['title'],key='title_'+sid)
        if st.button('Save name'): store.rename(sid,title);st.rerun()
        st.caption('Sessions and results are saved locally. Reopening a session does not call the LLM.')
    cfg=json.loads(session['config']);turns=store.turns(sid)
    running=any(t['status']=='running' for t in turns)
    st.title('ForecastBot')
    st.caption('Python experiment workflow · 63 questions per dataset, 126 questions total · each request runs independently')
    draft='draft_'+sid;pending='parent_'+sid
    for t in turns:
        with st.chat_message('user'): st.write(t['config'].get('_chat_message',t['question']))
        with st.chat_message('assistant'):
            render_result(t)
            if t['status'] in ['failed','interrupted'] and not running:
                if st.button('Edit or retry this question',key='retry_'+t['id']):
                    st.session_state[draft]=t['question'];st.session_state[pending]=t['id'];st.rerun()
            if t['status']=='running':
                if st.button('Mark as interrupted (only if the run has stopped)',key='stop_'+t['id']):
                    store.interrupt(t['id']);st.rerun()
    with st.expander('Build a forecast request',expanded=not bool(turns)):
        cols=st.columns(2)
        dataset=cols[0].selectbox('Dataset',['M5','HCMC_BUS'],index=['M5','HCMC_BUS'].index(cfg.get('dataset','M5')),key='dataset_'+sid)
        model=cols[1].selectbox('LLM',list(MODELS),format_func=MODELS.get,index=list(MODELS).index(cfg.get('model','gemini')),key='model_'+sid)
        st.subheader('Experiment questions')
        cases=runner.load_cases(dataset)
        choice=st.selectbox('Experiment question',cases,format_func=lambda x:f'{x[0]} · {x[1]}',key='case_'+sid+dataset)
        if st.button('Use original experiment question',disabled=running):
            st.session_state[draft]=choice[1];st.session_state.__setitem__(pending, 'new')
            sample_config=dict(dataset=dataset,model=model)
            store.auto_title(sid,choice[1]);store.save_config(sid,sample_config);st.rerun()
        st.divider()
        st.subheader('Custom question')
        gran=st.selectbox('Time unit',list(UNITS),format_func=UNITS.get,index=list(UNITS).index(cfg.get('granularity','day')),key='gran_'+sid)
        horizons=runner.ALLOWED[gran];old=cfg.get('horizon',horizons[0])
        horizon=st.selectbox('Horizon',horizons,index=horizons.index(old) if old in horizons else 0,key='horizon_'+sid+gran)
        filters={}
        if st.button('Refresh filters'):
            filter_catalog.clear()
        try:
            catalog=filter_catalog(dataset)
        except Exception:
            catalog=None
            st.warning('Unable to load filters from the database. Check the connection and refresh, or enter values manually.')
        st.caption('These are the filter fields used in the experiments. Leave a value blank to include all values.')
        for k in FIELDS[dataset]:
            key=f'filter_{sid}_{dataset}_{k}'
            saved=cfg.get('filters',{}).get(k,'') if cfg.get('dataset')==dataset else ''
            if catalog is None:
                filters[k]=st.text_input(k,value=saved,key=key+'_manual')
                continue
            vals=['']+choices(catalog,k,filters)
            if key in st.session_state and st.session_state[key] not in vals:
                st.session_state[key]=''
            if saved not in vals: saved=''
            filters[k]=st.selectbox(k,vals,index=vals.index(saved),key=key,
                                   disabled=len(vals)==1,
                                   help='No values match the filters above.' if len(vals)==1 else None)
        config=dict(dataset=dataset,model=model,granularity=gran,horizon=horizon,filters=filters)
        st.caption('Custom question preview')
        st.write(question_from_config(config))
        if st.button('Insert into chat',disabled=running):
            st.session_state[draft]=question_from_config(config);st.session_state.__setitem__(pending, 'new')
            store.auto_title(sid,st.session_state[draft])
            store.save_config(sid,config)
            st.rerun()
    store.save_config(sid,config)
    waiting = turns[-1] if turns and turns[-1]['status']=='needs_clarification' else None
    if st.session_state.get(pending)=='new':
        waiting=None
    placeholder='Enter additional details…' if waiting else 'Enter a new forecast request…'
    question=st.chat_input(placeholder,key=draft,disabled=running)
    if question and question.strip():
        message=question.strip()
        parent=st.session_state.pop(pending,None)
        effective=message
        active_config=dict(config)
        if waiting:
            parent=waiting['id']
            req=waiting['result'].get('forecast_request',{})
            clarification=req.get('user_message_for_clarification') or ('Please provide: '+', '.join(req.get('missing_fields',[])))
            effective=(waiting['question']+'\nAssistant clarification: '+clarification
                       +'\nAdditional user details: '+message)
            # Continue the same workflow using its original dataset/model configuration.
            active_config=dict(waiting['config'])
            # Router may have identified a dataset different from the builder default
            # before stopping for missing horizon/granularity.
            if req.get('dataset') in ('M5','HCMC_BUS'):
                active_config['dataset']=req['dataset']
        elif parent=='new':
            parent=None
        if turns and turns[-1]['status']=='success' and parent is None:
            # Keep the completed conversation visible until the next submission.
            # That submission starts a fresh persisted session without old context.
            sid=store.create()
            st.session_state.demo_sid=sid
        store.save_config(sid,active_config)
        active_config['_chat_message']=message
        with st.spinner('Processing your request…'):
            execute(store,sid,effective,active_config,parent)
        st.rerun()
