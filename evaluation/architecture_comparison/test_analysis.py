import unittest
import numpy as np
import pandas as pd
from evaluation.architecture_comparison.analyze import valid_explanation, holm, paired_stats, pair_rows


class ComparisonTests(unittest.TestCase):
    def test_explanation_requires_all_original_gates(self):
        pipeline = {'status': 'success', 'forecast_valid': True, 'chart_status': 'success'}
        judge = dict(explanation_available=True, judge_status='success', relevance=5, faithfulness=4, clarity=5, usefulness=4)
        self.assertTrue(valid_explanation(pipeline, judge)[0])
        self.assertFalse(valid_explanation({**pipeline, 'chart_status': 'failed'}, judge)[0])
        self.assertFalse(valid_explanation(pipeline, {**judge, 'faithfulness': np.nan})[0])
        self.assertFalse(valid_explanation(pipeline, {})[0])

    def test_ties_missing_values_and_direction(self):
        g = pd.DataFrame({'valid': [True, True, True], 'mae_multi': [2, 2, np.nan], 'mae_single': [2, 2, 3]})
        r = paired_stats(g, 'valid', 'mae')
        self.assertEqual((r['n_pairs'], r['p_value'], r['ties']), (2, 1, 2))
        g['mae_single'] = [1, 1, 3]
        self.assertEqual(paired_stats(g, 'valid', 'mae')['single_better'], 2)
        self.assertEqual(paired_stats(g.iloc[:0], 'valid', 'mae')['test_status'], 'insufficient_pairs')

    def test_holm(self):
        np.testing.assert_allclose(holm([.01, .04, np.nan, .03]), [.03, .06, np.nan, .06], equal_nan=True)

    def test_pair_gate_and_case_keys(self):
        base = dict(dataset='M5', model='gemini', case_id='M01', holdout='[{"y":1}]',
                    forecast_comparable=True, strict_upstream=True, judge_model='judge',
                    explanation_valid=True, explanation_exclusion_reason='', pipeline_success=True)
        a = {**base, 'architecture': 'multi_agent'}
        b = {**base, 'architecture': 'single_agent'}
        p = pair_rows(pd.DataFrame([a,b]))
        self.assertTrue(p.iloc[0].explanation_common)
        b.update(holdout='[{"y":2}]', judge_model='other')
        p = pair_rows(pd.DataFrame([a,b]))
        self.assertFalse(p.iloc[0].forecast_common)
        self.assertFalse(p.iloc[0].explanation_common)
        self.assertIn('different_or_missing_judge_model', p.iloc[0].explanation_exclusion_reason)


if __name__ == '__main__':
    unittest.main()
