import hashlib
import unittest

from jev_alpha.filing_text import filing_packet


def packet(html):
    raw = html.encode()
    return filing_packet(raw, document_id='issuer-test', symbol='TEST', source_url='https://example.com',
                         role='latest', report_date='2026-06-30', source_sha256=hashlib.sha256(raw).hexdigest())


class FilingTextTests(unittest.TestCase):
    def test_hidden_inline_facts_excluded_without_losing_following_visible_text(self):
        html = '<html><head><style>.x{color:red}</style></head><body><div style="display: none"><ix:hidden>secretfact</ix:hidden></div><div hidden>hiddenfact</div><p>' + ('Visible financial paragraph. ' * 5) + '</p><p>Subsequent refund collected.</p></body></html>'
        result = packet(html)
        text = ' '.join(p['text'] for p in result['current_source_passages_with_ids'])
        self.assertNotIn('secretfact', text)
        self.assertNotIn('hiddenfact', text)
        self.assertIn('Subsequent refund collected.', text)
        self.assertGreater(result['extraction']['hidden_text_characters_excluded'], 0)

    def test_cells_separated_and_unicode_preserved(self):
        html = '<html><body><p>' + ('Context &amp; units € millions. ' * 5) + '</p><table><tr><td>2026</td><td>13.7</td></tr></table><img src="graph.png"/></body></html>'
        result = packet(html)
        text = '\n'.join(p['text'] for p in result['current_source_passages_with_ids'])
        self.assertIn('2026 | 13.7 |', text)
        self.assertIn('Context & units € millions.', text)
        self.assertEqual(result['extraction']['visible_images_not_ocrd'], 1)
        self.assertFalse(result['extraction']['visual_table_fidelity_verified'])

    def test_wrong_hash_rejected(self):
        with self.assertRaises(ValueError):
            filing_packet(b'<html>bad</html>', document_id='x', symbol='X', source_url='https://example.com',
                          role='latest', report_date='2026-01-01', source_sha256='bad')

    def test_source_newlines_and_nested_cell_divs_preserve_whole_row(self):
        html = '<html><body><p>' + ('Context sentence. ' * 8) + '</p><table>\n<tr>\n<td><div>2026</div></td>\n<td><p>$13.7</p></td>\n</tr>\n<tr><td></td><td></td></tr></table></body></html>'
        result = packet(html)
        rows = [p['text'] for p in result['current_source_passages_with_ids']]
        self.assertEqual(len(rows), 2)
        self.assertIn('2026 | $13.7 |', rows[1])


if __name__ == '__main__': unittest.main()
