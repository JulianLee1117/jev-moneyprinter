import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from jev_alpha.experiment import digest
from jev_alpha.operating_workflow import comparison_manifest
from jev_alpha.store import write_new_json


def cohort():
    text = 'We migrated our paid production Datadog monitoring.'
    return {"records": [{"record_id": "hackernews:1", "native_id": "1", "text": text, "vendor_ids": ["ddog"],
                         "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "url": "https://news.ycombinator.com/item?id=1", "source": "hackernews",
                         "published_at": "2026-08-01T00:00:00Z", "captured_at": "2026-09-18T00:00:00Z"}],
            "splits": {"hackernews:1": "development"}}


class WorkflowTests(unittest.TestCase):
    def test_missing_control_cannot_become_complete_empty_control(self):
        result = comparison_manifest(cohort(), {})
        self.assertEqual(result['arm_status']['keyword']['status'], 'complete')
        self.assertEqual(result['arm_status']['jev']['status'], 'incomplete')
        self.assertEqual(result['selected_pairs']['jev'], [])

    def test_unknowns_remain_retained_and_wrong_hash_fails(self):
        c = cohort()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'
            plan = {'schema_version': 'operating-screen-plan-v1', 'profile_sha256': 'frozen-test-profile',
                    'split': 'development', 'models': {'jev': 'test-jev', 'baseline': 'test-chat'}, 'cohort_sha256': digest(c), 'records': c['records']}
            report = {'cohort_sha256': digest(c), 'plan_sha256': digest(plan), 'split': 'development',
                      'schema_version': 'operating-screen-run-v1', 'summary': {'arm': 'jev', 'status': 'complete'}}
            selection = {'schema_version': 'operating-selection-v1', 'status': 'complete', 'response_coverage_complete': True,
                         'arm': 'jev', 'plan_sha256': digest(plan), 'outcomes': [
                {'record_id': 'hackernews:1', 'vendor_id': 'ddog', 'status': 'unknown'}]}
            report['selection_sha256'] = digest(selection)
            for name, value in [('plan', plan), ('report', report), ('selection', selection)]:
                write_new_json(directory / (name+'.json'), value)
            result = comparison_manifest(c, {'jev': [directory]})
            self.assertEqual(len(result['arm_status']['jev']['unknown_pairs']), 1)
            self.assertEqual(result['selected_pairs']['jev'], [])
            with self.assertRaisesRegex(ValueError, 'Mislabeled'):
                comparison_manifest(c, {'baseline': [directory]})
            changed = copy.deepcopy(c)
            changed['splits']['hackernews:1'] = 'evaluation'
            with self.assertRaisesRegex(ValueError, 'frozen cohort'):
                comparison_manifest(changed, {'jev': [directory]})
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                comparison_manifest(c, {'jev': [directory, directory]})


if __name__ == '__main__':
    unittest.main()
