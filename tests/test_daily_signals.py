import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha.daily_signals import signal_from_response, predict_daily
from jev_alpha.experiment import digest
from jev_alpha.jev import JevRequestError


def response(choice='positive', probability=.7, confidence=.01):
    return {'answers': {'domestic_producer_channel': {'choice': choice, 'confidence': confidence,
                      'probabilities': {'positive': probability}}}}


def fixture():
    request = {'model': 'typesafe/jev-1.13', 'state': {}, 'questions': {}}
    manifest = {'requests': [{'document_id': 'a', 'publication_date': '2026-01-01',
                             'request': request, 'request_sha256': digest(request)}]}
    protocol = {'prediction_manifest_sha256': digest(manifest),
                'selected': [{'document_id': 'a', 'publication_date': '2026-01-01'}],
                'arms': ['jev', 'baseline', 'always_long', 'cash'],
                'model_policy': {'positive_probability_threshold': .70,
                                 'confidence_statistic_used': False, 'provider_phase_budget_usd': 2}}
    return manifest, protocol


class DailySignalTests(unittest.TestCase):
    def test_uses_option_probability_not_confidence(self):
        self.assertEqual(signal_from_response(response())['side'], 'long')
        self.assertEqual(signal_from_response(response(probability=.69, confidence=.99))['side'], 'cash')
        self.assertEqual(signal_from_response(response(choice='mixed', probability=.8))['side'], 'cash')

    def test_changed_input_rejected_before_paid_call(self):
        manifest, protocol = fixture()
        manifest['requests'][0]['publication_date'] = '2026-01-02'
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one') as paid:
            with self.assertRaises(ValueError): predict_daily(None, manifest, protocol, Path(tmp)/'out', live=True)
            paid.assert_not_called()

    def test_failure_and_skips_are_not_cash(self):
        manifest, protocol = fixture()
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one', side_effect=JevRequestError('Synthetic')):
            result = predict_daily(None, manifest, protocol, Path(tmp)/'out', live=True)
        arms = result['signals'][0]['arms']
        self.assertEqual(arms['jev']['status'], 'failed')
        self.assertEqual(arms['baseline']['status'], 'skipped_after_failure')
        self.assertIsNone(arms['jev']['side'])
        self.assertIsNone(arms['baseline']['side'])
        self.assertEqual(arms['cash']['side'], 'cash')

    def test_dry_run_keeps_all_arms_without_calling(self):
        manifest, protocol = fixture()
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one') as paid:
            result = predict_daily(None, manifest, protocol, Path(tmp)/'out')
            paid.assert_not_called()
            self.assertEqual(len(result['signals'][0]['arms']), 4)

    def test_archived_invalid_completion_does_not_skip_next_arm(self):
        manifest, protocol = fixture()
        failure = JevRequestError('Synthetic', validation_reason='Invalid choice')
        failure.response_blob_sha256 = 'diagnostic'
        success = {'response': response(), 'request_hash': 'x', 'cached': False, 'latency_ms': 1}
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one', side_effect=[failure, success]) as paid:
            result = predict_daily(None, manifest, protocol, Path(tmp)/'out', live=True)
            self.assertEqual(paid.call_count, 2)
        self.assertEqual(result['status'], 'complete_with_failures')
        self.assertIsNone(result['signals'][0]['arms']['jev']['side'])
        self.assertEqual(result['signals'][0]['arms']['baseline']['side'], 'long')

    def test_resume_preserves_failed_arm_without_retry(self):
        manifest, protocol = fixture()
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one', side_effect=JevRequestError('Synthetic')):
            old = predict_daily(None, manifest, protocol, Path(tmp)/'old', live=True)
        success = {'response': response(), 'request_hash': 'x', 'cached': False, 'latency_ms': 1}
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_signals.run_one', return_value=success) as paid:
            result = predict_daily(None, manifest, protocol, Path(tmp)/'new', live=True, resume=old)
            self.assertEqual(paid.call_count, 1)
            self.assertEqual(paid.call_args.kwargs['arm'], 'baseline')
        self.assertEqual(result['status'], 'complete_with_failures')
        self.assertEqual(result['resume_signals_sha256'], digest(old))
        self.assertEqual(result['signals'][0]['arms']['jev'], old['signals'][0]['arms']['jev'])


if __name__ == '__main__': unittest.main()
