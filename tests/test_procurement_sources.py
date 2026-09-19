import json
import io
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import patch

from jev_alpha.procurement_sources import (
    Capture, _legistar, annotate_document_dates, association_date, balanced_sample, collect, extract, fdot_award_rows, page_links,
)
from jev_alpha.store import Store


class ProcurementSourcesTests(unittest.TestCase):
    def test_row_dates_and_meeting_urls_do_not_become_publication_times(self):
        raw = b'<html><table><tr><td>December 16, 2025 at 9:30 AM</td><td><a href="/Public/Agenda/1448?meeting=1">Agenda</a></td></tr><tr><td>November 18, 2025</td><td><a href="/Public/Agenda/1448?meeting=2">Agenda</a></td></tr></table></html>'
        links = page_links(raw, 'https://meetings.boardbook.org/Public/Organization/1448')
        self.assertEqual([association_date(x['context']) for x in links], ['2025-12-16', '2025-11-18'])
        self.assertEqual(association_date('https://www.txdot.gov/content/dam/docs/commission/2025/0130/4a.pdf'), '2025-01-30')

    def test_same_url_deduplicates_bytes_but_preserves_multiple_sightings(self):
        with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / 'sources/archive') as store:
            root = Path(tmp)
            capture = Capture(root, store, live=False, maximum=5)
            raw = b'<html><body><p>Recommended infrastructure contract to an identified private company, pending board approval.</p></body></html>'
            sha = store.put_blob(raw)
            receipt = {'sha256': sha, 'blob_path': (store.blobs / sha).relative_to(root).as_posix(), 'captured_at': '2026-09-19T00:00:00Z', 'content_type': 'text/html', 'timings_ms': {'fetch': 2}, 'status': 200}
            with patch.object(capture, 'get', return_value=(raw, receipt)) as get:
                first = capture.document('a', 'https://example.gov/report', '2025-07-01', 'Report', 'report', 'm1')
                second = capture.document('a', 'https://example.gov/report', '2025-08-01', 'Report again', 'report', 'm2')
            self.assertIs(first, second)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(len(capture.sightings), 2)
            self.assertIsNone(first['published_at'])
            self.assertEqual(first['availability'], 'unverified_historical')
            self.assertTrue((root / first['text_path']).exists())

    def test_legistar_pagination_and_exclusion_of_current_outcomes(self):
        events = [{'EventId': n, 'EventDate': '2025-07-01T00:00:00', 'EventBodyName': 'Board'} for n in range(1001)]
        calls = []
        def payload(capture, url):
            calls.append(url)
            if '/EventItems?' in url:
                return [{'EventItemId': 12, 'EventItemTitle': 'Approve contract', 'EventItemActionName': 'Passed later', 'EventItemPassedFlag': 1, 'EventItemMinutesNote': 'Future decision', 'EventItemMatterAttachments': []}]
            return events[1000:] if '%24skip=1000' in url else events[:1000]
        with patch('jev_alpha.procurement_sources._json', side_effect=payload):
            result = _legistar(None, {'id': 'example', 'client': 'example'}, '2025-07-01', '2025-12-31', True)
        self.assertEqual(result['inventory_pages'], 2)
        self.assertEqual(result['packages_in_window'], 1001)
        self.assertEqual(len(result['packages_selected']), 1)
        serialized = json.dumps(result['items'])
        self.assertNotIn('Passed later', serialized)
        self.assertNotIn('Future decision', serialized)

    def test_balanced_sample_preserves_failed_extractions_and_is_order_stable(self):
        documents = [{'source_id': source, 'document_id': str(i) + source, 'kind': 'report', 'extraction_status': 'capture_failed'} for source in ['a', 'b'] for i in range(10)]
        chosen = balanced_sample(documents, 6)
        self.assertEqual(chosen, balanced_sample(list(reversed(documents)), 6))
        self.assertEqual(sum(x['source_id'] == 'a' for x in chosen), 3)
        self.assertTrue(all(x['extraction_status'] == 'capture_failed' for x in chosen))

    def test_embedded_documents_and_letting_heading_do_not_use_later_award_date(self):
        raw = b'''<a href="javascript:void(0)" data-sf-role="toggleLink">July 30, 2025</a>
        <p>Posted August 13, 2025</p><table><tr><td>8/19/2025</td>
        <td><a href="https://ftp.fdot.gov/public/file/opaque/T100-Bid-Tab.pdf">T100</a></td></tr></table>
        <span title="Agenda" data-pdf="https://example.gov/agenda.pdf">Agenda</span>'''
        links = page_links(raw, 'https://www.fdot.gov/archive')
        self.assertEqual(links[0]['package_date'], '2025-07-30')
        self.assertEqual(association_date(links[0]['context']), '2025-08-19')
        self.assertEqual(links[1]['url'], 'https://example.gov/agenda.pdf')

    def test_preflight_cap_is_per_source_and_keeps_attempted_sighting(self):
        with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / 'sources/archive') as store:
            capture = Capture(Path(tmp), store, live=False, maximum=10, source_maximum=1)
            capture.document('a', 'https://example.gov/one', '2025-07-01', 'A', 'report', 'p1')
            with self.assertRaisesRegex(RuntimeError, 'Per-source document cap'):
                capture.document('a', 'https://example.gov/two', '2025-07-01', 'B', 'report', 'p1')
            capture.document('b', 'https://example.gov/three', '2025-07-01', 'C', 'report', 'p2')
            self.assertEqual(len(capture.documents), 2)
            self.assertEqual(len(capture.sightings), 3)

    def test_matter_text_excludes_action_metadata_and_docx_keeps_table_text(self):
        text, status = extract(json.dumps({'MatterTextPlain': 'Proposed construction contract', 'MatterStatusName': 'Passed next month'}).encode(), 'application/json', Path('unused'))
        self.assertEqual(text, 'Proposed construction contract')
        self.assertNotIn('Passed', text)
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Vendor</w:t><w:tab/><w:t>Amount</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
        text, status = extract(raw.getvalue(), 'application/octet-stream', Path('unused'))
        self.assertEqual(text, 'Vendor\tAmount')
        self.assertEqual(status, 'extracted_docx_main_text_layout_unverified')

    def test_concurrent_sources_share_one_url_capture_and_preserve_source_order(self):
        ready = threading.Barrier(2)
        raw = b'<html><body>A public proposed infrastructure contract report for the complete meeting.</body></html>'
        class Response:
            status, url = 200, 'https://example.gov/shared'
            headers = {'Content-Type': 'text/html'}
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _): return raw
        def source(capture, definition, *_):
            ready.wait(timeout=3)
            capture.document(definition['id'], 'https://example.gov/shared', '2025-07-01', 'Shared contract', 'report', 'package')
            return {'source_id': definition['id'], 'inventory_complete': True, 'errors': []}
        with tempfile.TemporaryDirectory() as tmp, patch('jev_alpha.procurement_sources._html_source', side_effect=source), patch('jev_alpha.procurement_sources.urlopen', return_value=Response()) as get:
            protocol = {'sources': [{'id': s, 'kind': 'fixture', 'url': 'https://example.gov/index'} for s in ['z', 'a']]}
            manifest = collect(Path(tmp), protocol, live=True, max_documents=2)
            self.assertEqual(get.call_count, 1)
            self.assertEqual([r['source_id'] for r in manifest['source_results']], ['z', 'a'])
            self.assertEqual(len(manifest['documents']), 2)
            self.assertEqual(manifest['new_requests'], 1)
            self.assertTrue(manifest['complete'])

    def test_reused_fdot_addendum_keeps_issue_bid_and_package_dates_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'text.txt').write_text('October 9, 2025\nADDENDUM NO. 2\nBIDS TO BE RECEIVED: October 29, 2025', encoding='utf-8')
            doc = {'source_id': 'fdot', 'text_path': 'text.txt', 'association_date': '2025-10-29',
                   'association_dates': ['2025-10-29', '2025-12-03'], 'published_at': None}
            annotate_document_dates(root, [doc])
            self.assertEqual(doc['document_stated_date'], '2025-10-09')
            self.assertEqual(doc['stated_bid_receipt_date'], '2025-10-29')
            self.assertTrue(doc['bid_date_differs_from_some_package_dates'])
            self.assertTrue(doc['reused_across_package_dates'])
            self.assertIsNone(doc['published_at'])

    def test_fdot_award_excerpt_preserves_exact_stage_label_and_explicit_prior_link(self):
        raw = b'''<a href="#dec" data-sf-role="toggleLink">December 3, 2025</a><table>
        <tr><th>PROPOSAL/CONTRACT NO. - BID TAB</th><th>INTENT TO AWARD - CONTRACTOR'S NAME</th><th>DATE AWARDED</th></tr>
        <tr><td><a href="https://ftp.fdot.gov/public/file/id/T2A48-Bid-Tab.pdf">T2A48</a></td><td>Private Contractor, Inc.</td><td>12/23/2025</td></tr></table>'''
        url = 'https://www.fdot.gov/contracts/2025'
        rows = fdot_award_rows(raw, url, '2025-07-01', '2025-12-31')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['letting_date'], '2025-12-03')
        self.assertEqual(rows[0]['award_date'], '2025-12-23')
        self.assertIn('INTENT TO AWARD', rows[0]['text'])
        self.assertEqual(rows[0]['related_bid_document_urls'], ['https://ftp.fdot.gov/public/file/id/T2A48-Bid-Tab.pdf'])
        with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / 'sources/archive') as store:
            root = Path(tmp)
            capture = Capture(root, store, live=False, maximum=5)
            sha = store.put_blob(raw)
            receipt = {'sha256': sha, 'blob_path': (store.blobs / sha).relative_to(root).as_posix(), 'captured_at': '2026-09-19T00:00:00Z'}
            doc = capture.award_row('fdot', url, receipt, rows[0])
            self.assertEqual((root / doc['blob_path']).read_bytes(), raw)
            self.assertEqual(doc['parent_source_sha256'], sha)
            self.assertIsNone(doc['published_at'])
            report = root / 'bid.txt'
            report.write_text('Vendor Ranking\nBid Type Bid Status\nDecember 03, 2025 Letting: CT251203\nPage 1 of 1\n12/15/2025Florida Department of Transportation', encoding='utf-8')
            bid = {'source_id': 'fdot', 'url': rows[0]['related_bid_document_urls'][0].replace('Bid-Tab', 'Bid%20Tab'), 'text_path': 'bid.txt',
                   'association_date': '2025-12-03', 'association_dates': ['2025-12-03'], 'published_at': None}
            annotate_document_dates(root, [doc, bid])
            self.assertEqual(bid['association_date'], '2025-12-15')
            self.assertEqual(bid['letting_dates'], ['2025-12-03'])
            self.assertEqual(bid['source_stage'], 'bid')
            self.assertEqual(bid['source_stage_evidence'][0]['quote'], 'Vendor Ranking')
            self.assertNotIn('source_stage', doc)
            self.assertEqual(doc['association_date'], '2025-12-23')

    def test_offline_uncaptured_source_is_incomplete_not_fake_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = {'sources': [{'id': 'seattle', 'kind': 'legistar', 'client': 'seattle', 'url': 'https://seattle.legistar.com/Calendar.aspx'}]}
            result = collect(Path(tmp), protocol, live=False)
            self.assertFalse(result['complete'])
            self.assertEqual(result['new_requests'], 0)
            self.assertEqual(result['documents'], [])
            self.assertTrue(result['source_results'][0]['errors'])
            self.assertTrue((Path(tmp) / result['manifest_path']).exists())


if __name__ == '__main__':
    unittest.main()
