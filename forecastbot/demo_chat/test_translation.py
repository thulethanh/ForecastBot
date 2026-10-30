import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from forecastbot.demo_chat import translation
from forecastbot.demo_chat import service
from forecastbot.demo_chat.store import Store


class TranslationTests(unittest.TestCase):
    def test_translation_cache_preserves_source(self):
        explanation={'summary_text':'Dự báo 123 lượt khách.', 'insights':['Tăng 5%.']}
        original=json.dumps(explanation)
        response=SimpleNamespace(status='completed', output_text='Forecast: 123 passengers.\n\nIncrease: 5%.', usage=None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'OPENAI_API_KEY':'test'}), patch.object(translation,'OpenAI') as client:
            client.return_value.__enter__.return_value.responses.create.return_value=response
            first=translation.translate(Path(tmp),explanation)
            second=translation.translate(Path(tmp),explanation)
            self.assertEqual(first,second)
            self.assertEqual(first['Tiếng Việt'],'Dự báo 123 lượt khách.\n\nTăng 5%.')
            self.assertIn('123',first['English'])
            api=client.return_value.__enter__.return_value.responses.create
            api.assert_called_once()
            self.assertEqual(api.call_args.kwargs['model'],'gpt-5.6-terra')
            self.assertEqual(json.dumps(explanation),original)

    def test_missing_key_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'OPENAI_API_KEY':''}), patch.object(translation,'OpenAI') as client:
            result=translation.translate(Path(tmp),'Nội dung')
            self.assertEqual(result['English'],'')
            self.assertIn('OPENAI_API_KEY',result['error'])
            client.assert_not_called()
            self.assertEqual(translation.read_translation(Path(tmp),'Nội dung'),result)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'OPENAI_API_KEY':'test'}), patch.object(translation,'OpenAI') as client:
            api=client.return_value.__enter__.return_value.responses.create
            api.side_effect=[RuntimeError('private provider details'),SimpleNamespace(status='completed',output_text='Content',usage=None)]
            failed=translation.translate(Path(tmp),'Nội dung')
            self.assertNotIn('private',failed['error'])
            self.assertEqual(translation.translate(Path(tmp),'Nội dung')['English'],'Content')
            self.assertEqual(api.call_count,2)

    def test_forecast_unchanged_when_translation_storage_fails(self):
        result={'status':'success','explanation_result':{'summary_text':'Nội dung'},'total_tokens':42}
        with tempfile.TemporaryDirectory() as tmp, patch.object(service,'ROOT',Path(tmp)), patch.object(service.runner,'run_case',return_value=result), patch.object(service.runner,'write_case') as export, patch.object(service,'translate',side_effect=OSError('disk')):
            store=Store(Path(tmp)); sid=store.create()
            tid=service.execute(store,sid,'Question',{'dataset':'M5','model':'gemini'})
            self.assertEqual(store.turns(sid)[0]['result'],result)
            self.assertEqual(export.call_args.args[1],result)
            self.assertEqual(store.turns(sid)[0]['id'],tid)
