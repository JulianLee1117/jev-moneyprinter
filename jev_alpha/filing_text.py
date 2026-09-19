"""Deterministic visible-HTML passage extraction; no claim of visual table fidelity."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import re


class _FilingText(HTMLParser):
    blocks = {'p', 'div', 'tr', 'h1', 'h2', 'h3', 'h4', 'li', 'section', 'table'}
    voids = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.images = 0
        self.hidden_text_chars = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        style = re.sub(r'\s+', '', attrs.get('style', '').lower())
        hidden = ((self.stack and self.stack[-1][1]) or tag in {'script', 'style', 'ix:hidden', 'head'}
                  or 'hidden' in attrs or 'display:none' in style or 'visibility:hidden' in style)
        if tag == 'img' and not hidden:
            self.images += 1
        inside_cell = any(t in {'td', 'th'} for t, _ in self.stack)
        if not hidden and (tag in self.blocks or tag in {'br', 'hr'}) and not inside_cell:
            self.parts.append('\n')
        if tag not in self.voids:
            self.stack.append((tag, bool(hidden)))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.voids:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        hidden = self.stack[-1][1] if self.stack else False
        if not hidden:
            if tag in self.blocks and not any(t in {'td', 'th'} for t, _ in self.stack):
                self.parts.append('\n')
            elif tag in {'td', 'th'}:
                self.parts.append(' | ')
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.stack and self.stack[-1][1]:
            self.hidden_text_chars += len(data)
        else:
            # HTML source formatting newlines do not create rendered paragraphs.
            self.parts.append(re.sub(r'\s+', ' ', data))


def filing_packet(raw: bytes, *, document_id: str, symbol: str, source_url: str,
                  role: str, report_date: str, source_sha256: str) -> dict:
    """Include all extracted visible text; never keyword-filter claim passages.

    Inline hidden facts are excluded to avoid duplicating rendered values.
    External CSS, images, column spans and visual cell alignment are not
    interpreted. Any material amount must be checked against the original.
    """
    actual = hashlib.sha256(raw).hexdigest()
    if actual != source_sha256:
        raise ValueError('Filing body hash mismatch')
    html = raw.decode('utf-8-sig')
    if not re.search(r'<html\b', html, re.I):
        raise ValueError('Expected a complete HTML filing body')
    if role not in {'latest', 'prior_comparable'}:
        raise ValueError('Filing role must identify its comparison state')
    parser = _FilingText()
    parser.feed(html)
    parser.close()
    lines = [re.sub(r'\s+', ' ', line).strip() for line in ''.join(parser.parts).splitlines()
             if line.strip(' \t\r\n|\u200b\xa0')]
    if sum(map(len, lines)) < 100:
        raise ValueError('Filing visible text is unexpectedly short')
    passages = [{'passage_id': f'p{i:05d}', 'text': line} for i, line in enumerate(lines, 1)]
    return {'schema_version': 'issuer-filing-packet-v1', 'episode_manifest': {
        'document_id': document_id, 'symbol': symbol, 'source_sha256': actual,
        'source_url': source_url, 'role': role, 'report_date': report_date,
        'title': f'{symbol} {report_date} {role} financial filing',
        'passage_selection': 'all_extracted_visible_html_text',
        'earliest_public_verified': False, 'historical_content_version_verified': False},
        'current_source_passages_with_ids': passages,
        'extraction': {'parser': 'visible-html-v1', 'passages': len(passages),
            'visible_text_characters': sum(len(p['text']) for p in passages),
            'hidden_text_characters_excluded': parser.hidden_text_chars,
            'visible_images_not_ocrd': parser.images,
            'replacement_characters_in_extracted_text': sum(line.count('\ufffd') for line in lines),
            'full_original_html_archived': True,
            'visual_table_fidelity_verified': False,
            'coverage_scope': 'All extracted visible text, not proof of full visual document coverage. No keyword selection.'}}
