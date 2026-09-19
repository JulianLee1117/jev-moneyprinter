from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha.claim_experiment import run_claim_screen
from jev_alpha.claims import prepare_claim_screen
from jev_alpha.jev import JevRequestError


def plan():
    return prepare_claim_screen({'episode_manifest': {'document_id': 'x', 'source_sha256': 'abc'},
        'current_source_passages_with_ids': [{'passage_id': 'p1', 'text': 'A refund remains outstanding.'}]})


class ClaimExperimentTests(unittest.TestCase):
    def test_dry_run_never_opens_archive_or_calls_model(self):
        with TemporaryDirectory() as tmp, patch('jev_alpha.claim_experiment.run_one') as paid:
            root = Path(tmp)
            result = run_claim_screen(root/'archive', plan(), root/'out')
            self.assertEqual(result['status'], 'dry_run')
            self.assertFalse((root/'archive').exists())
            paid.assert_not_called()

    def test_failure_keeps_all_evidence_and_no_retry(self):
        with TemporaryDirectory() as tmp, patch('jev_alpha.claim_experiment.run_one', side_effect=JevRequestError('Synthetic')) as paid:
            root = Path(tmp)
            result = run_claim_screen(root/'archive', plan(), root/'out', live=True)
            self.assertEqual(paid.call_count, 1)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['passages_selected'], result['passages_total'])


if __name__ == '__main__': unittest.main()
