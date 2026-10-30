import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from forecastbot.demo_chat.store import Store
from forecastbot.demo_chat import service


class DemoTests(unittest.TestCase):
    def setUp(self):
        translation_patch = patch.object(service, 'translate')
        translation_patch.start()
        self.addCleanup(translation_patch.stop)

    def test_experiment_filters_and_english_custom_question(self):
        from streamlit.testing.v1 import AppTest
        from forecastbot.demo_chat import ui
        from forecastbot.demo_chat.filters import FIELDS
        root=Path(__file__).resolve().parents[2]
        for dataset, directory in [('M5','m5'),('HCMC_BUS','hcmcbus')]:
            used=set()
            for path in (root/'evaluation/ground_truth'/directory/(directory+'_forecast_requests')).glob('*.json'):
                used.update(k for k,v in json.loads(path.read_text())['filters'].items() if v)
            self.assertEqual(set(FIELDS[dataset]),used)
        with tempfile.TemporaryDirectory() as tmp, patch.object(ui,'filter_catalog',side_effect=RuntimeError('offline')), patch.object(ui,'ROOT',Path(tmp)):
            app=AppTest.from_file(str(root/'app_experiment_chat.py')).run()
            labels=[widget.label for widget in app.selectbox]
            self.assertLess(labels.index('Experiment question'),labels.index('Time unit'))
            self.assertFalse(app.multiselect)
            self.assertTrue(set(FIELDS['M5']).issubset(w.label for w in app.text_input))
            self.assertFalse(any(w.label in ('cat_id','dept_id','id') for w in app.text_input))
            next(w for w in app.text_input if w.label=='item_id').set_value('FOODS_3_228').run()
            self.assertTrue(any('Forecast sales for the M5 dataset' in w.value and 'item_id=FOODS_3_228' in w.value for w in app.markdown))
            next(w for w in app.selectbox if w.label=='Dataset').set_value('HCMC_BUS').run()
            self.assertTrue(set(FIELDS['HCMC_BUS']).issubset(w.label for w in app.text_input))
            self.assertFalse(any(w.label in ('store_id','route_name','start_stop','end_stop') for w in app.text_input))
            self.assertTrue(any('Forecast passenger counts for the HCMC_BUS dataset' in w.value for w in app.markdown))
            self.assertFalse(app.exception)

    def test_persistence_clarification_and_runner_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp));sid=store.create('Test')
            c=dict(dataset='M5',model='gemini',granularity='day',horizon=7,filters={})
            result={'status':'failed','forecast_request':{'needs_clarification':True,'missing_fields':['horizon']}}
            with patch.object(service.runner,'run_case',return_value=result) as run,patch.object(service.runner,'write_case') as write,patch.object(service,'ROOT',Path(tmp)):
                tid=service.execute(store,sid,'question',c)
                run.assert_called_once_with('M5','gemini',tid,'question')
                self.assertEqual(write.call_args.args[0],Path(tmp)/'sessions'/sid)
            reopened=Store(Path(tmp));self.assertEqual(reopened.turns(sid)[0]['status'],'needs_clarification')
            self.assertEqual(reopened.turns(sid)[0]['question'],'question')
            named=reopened.create()
            reopened.auto_title(named,'Tên tự sinh')
            reopened.rename(named,'Tên của tôi')
            reopened.auto_title(named,'Tên khác')
            self.assertEqual(next(r for r in reopened.sessions() if r['id']==named)['title'],'Tên của tôi')
            t=reopened.begin(sid,'complete',c,tid)
            with self.assertRaises(ValueError): reopened.begin(sid,'duplicate',c)
            reopened.finish(t,{'status':'success'})
            self.assertEqual(reopened.turns(sid)[1]['parent_id'],tid)

    def test_ui_no_live_calls(self):
        from streamlit.testing.v1 import AppTest
        from forecastbot.demo_chat import ui
        with tempfile.TemporaryDirectory() as tmp,patch.object(ui,'filter_catalog',side_effect=RuntimeError('offline')), patch.object(ui,'ROOT',Path(tmp)),patch.object(ui,'execute') as execute:
            app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'app_experiment_chat.py')).run(timeout=20)
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label=='Insert into chat').click().run()
            self.assertFalse(app.exception)
            execute.assert_not_called()
            next(b for b in app.button if b.label=='＋ New session').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(Store(Path(tmp)).sessions()),2)

    def test_chat_clarification_then_success(self):
        from streamlit.testing.v1 import AppTest
        from forecastbot.demo_chat import ui
        with tempfile.TemporaryDirectory() as tmp, patch.object(ui,'filter_catalog',side_effect=RuntimeError('offline')), patch.object(ui,'ROOT',Path(tmp)), patch.object(service,'ROOT',Path(tmp)), patch.object(service.runner,'write_case'), patch.object(service.runner,'run_case') as run:
            run.return_value={'status':'failed','forecast_request':{'needs_clarification':True,'missing_fields':['horizon']}}
            app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'app_experiment_chat.py')).run()
            app.chat_input[0].set_value('Dự báo M5').run()
            self.assertFalse(app.exception)
            sid=Store(Path(tmp)).sessions()[0]['id']
            self.assertEqual(Store(Path(tmp)).turns(sid)[0]['status'],'needs_clarification')
            run.return_value={'status':'success','forecast_result':{'forecast':[{'ds':'2026-01-01','yhat':2}]},'forecast_history':[{'ds':'2025-12-31','y':1}],'explanation_result':{'summary_text':'Kết quả'},'split_info':{}}
            app.chat_input[0].set_value('7 ngày').run()
            self.assertIn('Dự báo M5',run.call_args.args[3])
            self.assertIn('7 ngày',run.call_args.args[3])
            self.assertFalse(app.exception)
            turns=Store(Path(tmp)).turns(sid)
            self.assertEqual(turns[1]['parent_id'],turns[0]['id'])
            self.assertEqual(turns[1]['status'],'success')
            calls=run.call_count
            app.run()
            self.assertEqual(run.call_count,calls)
            app.chat_input[0].set_value('Yêu cầu khác').run()
            self.assertEqual(run.call_args.args[3],'Yêu cầu khác')
            new_sid=app.session_state.demo_sid
            self.assertNotEqual(new_sid,sid)
            self.assertEqual(len(Store(Path(tmp)).turns(sid)),2)
            self.assertIsNone(Store(Path(tmp)).turns(new_sid)[-1]['parent_id'])

    def test_bus_clarification_uses_router_dataset_and_dialogue(self):
        from streamlit.testing.v1 import AppTest
        from forecastbot.demo_chat import ui
        with tempfile.TemporaryDirectory() as tmp, patch.object(ui,'filter_catalog',side_effect=RuntimeError('offline')), patch.object(ui,'ROOT',Path(tmp)), patch.object(service,'ROOT',Path(tmp)), patch.object(service.runner,'write_case'), patch.object(service.runner,'run_case') as run:
            clarification='Bạn muốn dự báo theo ngày, tháng, quý, hay năm và trong bao lâu?'
            run.return_value={'status':'failed','forecast_request':{'dataset':'HCMC_BUS','needs_clarification':True,'missing_fields':['granularity','horizon'],'user_message_for_clarification':clarification}}
            app=AppTest.from_file(str(Path(__file__).resolve().parents[2]/'app_experiment_chat.py')).run()
            question='Dự báo lượng hành khách trên bộ dữ liệu HCMC_BUS, với bộ lọc route_id=150, operator=HTX-195.'
            app.chat_input[0].set_value(question).run()
            sid=app.session_state.demo_sid
            # Reload the UI: pending context must survive in the persisted session.
            app.run()
            app.chat_input[0].set_value('30 ngày tới').run()
            self.assertFalse(app.exception)
            self.assertEqual(run.call_args.args[0],'HCMC_BUS')
            for part in (question,clarification,'30 ngày tới'):
                self.assertIn(part,run.call_args.args[3])
            self.assertEqual(app.session_state.demo_sid,sid)
            app.chat_input[0].set_value('Theo ngày nhé').run()
            for part in (question,'30 ngày tới','Theo ngày nhé'):
                self.assertIn(part,run.call_args.args[3])


if __name__=='__main__': unittest.main()
