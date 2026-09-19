import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

from jev_alpha.daily_prices import collect_daily_prices
from jev_alpha.experiment import digest
from jev_alpha.store import Store


def fixture():
    protocol = {'symbols': ['NUE', 'STLD'], 'benchmark': 'SPY',
                'selected': [{'document_id': 'a', 'publication_date': '2026-01-01'}],
                'document_ids': ['a'], 'arms': ['jev', 'baseline', 'always_long', 'cash'],
                'prediction_manifest_sha256': digest({'requests': 'synthetic'})}
    signals = {'protocol_sha256': digest(protocol), 'status': 'complete_with_failures',
               'prediction_manifest_sha256': protocol['prediction_manifest_sha256'],
               'frozen_at': '2026-09-18T20:00:00+00:00', 'signals': [
                   {'document_id': 'a', 'publication_date': '2026-01-01', 'arms': {
                   'jev': {'status': 'completed'}, 'baseline': {'status': 'failed'},
                   'always_long': {'status': 'completed', 'side': 'long'},
                   'cash': {'status': 'completed', 'side': 'cash'}}}]}
    return protocol, signals


class DailyPriceTests(unittest.TestCase):
    def test_direct_collector_rejects_cohort_date_manifest_and_arm_changes_before_network(self):
        for change in ('missing_document', 'duplicate_document', 'changed_date', 'manifest', 'arm'):
            protocol, signals = fixture()
            if change == 'missing_document':
                protocol['selected'].append({'document_id': 'b', 'publication_date': '2026-01-02'})
                protocol['document_ids'].append('b')
                signals['protocol_sha256'] = digest(protocol)
            elif change == 'duplicate_document':
                signals['signals'].append(signals['signals'][0])
            elif change == 'changed_date':
                signals['signals'][0]['publication_date'] = '2026-01-02'
            elif change == 'manifest':
                signals['prediction_manifest_sha256'] = 'changed'
            else:
                del signals['signals'][0]['arms']['cash']
            with self.subTest(change=change), TemporaryDirectory() as tmp:
                with patch('jev_alpha.daily_prices.subprocess.run') as network:
                    with self.assertRaises(ValueError):
                        collect_daily_prices(None, protocol, signals, Path(tmp)/'prices')
                    network.assert_not_called()
                    self.assertFalse((Path(tmp)/'prices').exists())

    def test_unattempted_signals_block_network(self):
        protocol, signals = fixture()
        signals['signals'][0]['arms']['baseline']['status'] = 'skipped_after_failure'
        with TemporaryDirectory() as tmp, patch('jev_alpha.daily_prices.subprocess.run') as network:
            with self.assertRaises(ValueError):
                collect_daily_prices(None, protocol, signals, Path(tmp)/'prices')
            network.assert_not_called()

    def test_denials_and_timeouts_preserve_capture_without_retry(self):
        protocol, signals = fixture()
        responses = [subprocess.TimeoutExpired('curl', 35),
                     subprocess.CompletedProcess([], 22, stdout=b''),
                     subprocess.CompletedProcess([], 0, stdout=b'{"chart":{"error":null,"result":[]}}')]
        with TemporaryDirectory() as tmp, Store(Path(tmp)/'archive') as store:
            with patch('jev_alpha.daily_prices.subprocess.run', side_effect=responses) as network:
                result = collect_daily_prices(store, protocol, signals, Path(tmp)/'prices')
                self.assertEqual(network.call_count, 3)
                self.assertIn('--retry', network.call_args.args[0])
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual([r['status'] for r in result['symbols']], ['failed', 'failed', 'captured'])
            self.assertEqual(result['signals_sha256'], digest(signals))
            self.assertTrue((Path(tmp)/'prices/capture.json').exists())
            self.assertIsNotNone(result['symbols'][-1]['raw_sha256'])


if __name__ == '__main__': unittest.main()
