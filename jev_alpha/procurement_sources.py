"""Bounded public procurement intake, with observed bytes rather than invented clocks.

Historical association dates are meeting/letting dates, never publication times.
Current API statuses are archived as inventory but excluded from evidence text.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import io
from http.cookiejar import CookieJar
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from xml.etree import ElementTree
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from .filing_text import _FilingText
from .store import Store, canonical_json, utc_now, write_new_json

MAX_DOCUMENTS = 10_000
MAX_BYTES = 25_000_000
PDF_PYTHON = Path(os.environ.get('JEV_PDF_PYTHON', str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')))
SEED = 'procurement-source-census-v1'
PROCUREMENT_PATTERN = r'\b(?:award\w*|contract\w*|bid(?:s|ding)?|procur\w*|purchas\w*|construct\w*|infrastruct\w*|project\w*|capital|agreement\w*|amend\w*|change\s+order\w*|consult\w*|letting|engineering|design|maintenance|repair|rehabilitat\w*|public\s+works|water|wastewater|bridge|roadway|paving|budget)\b'
SELECTION_POLICY = {'version': 'broad-procurement-titles-v2', 'pattern': PROCUREMENT_PATTERN,
                    'all_event_item_titles_retained': True, 'full_agendas_and_minutes': True,
                    'package_order': 'SHA256(seed + package id) across the full frozen period, before evidence caps',
                    'source_document_allocation': 'equal quotas with remainder assigned in frozen source order',
                    'matched_items': 'all linked attachments plus matching-version matter text',
                    'excluded_items': 'inventory only; ten hash-selected exclusions per source require recall audit',
                    'outcome_or_issuer_selection': False}
FOLDER_POLICY = {'version': 'fdot-public-folder-v1', 'max_folders': 10, 'listing_rows': 1000,
                 'max_children_per_folder': 25, 'depth': 1, 'child_order': 'filename ascending',
                 'priority': 'direct bid tabs and award rows before ancillary bid-item folder files',
                 'retrospective_children': 'inventory only; current folder membership does not prove historical association'}


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def normalized_url(url: str) -> str:
    p = urlparse(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password:
        raise ValueError('Public HTTPS URLs without credentials required')
    if p.hostname in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Local URLs are not public procurement sources')
    return urlunparse(p._replace(fragment=''))


class _Page(HTMLParser):
    """Keep anchor labels and row context; never use winning-bid keywords."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.parts, self.rows = [], [], {}
        self.row, self.number, self.anchor, self.hidden = None, 0, None, 0
        self.last_explicit_date = None
        self.package_date = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'script', 'style'}:
            self.hidden += 1
        if tag == 'tr':
            self.number += 1
            self.row = self.number
            self.rows[self.row] = []
        if tag == 'a' and attrs.get('href'):
            self.anchor = {'href': attrs['href'], 'text': '', 'row': self.row, 'position': len(''.join(self.parts)),
                           'section_date': self.last_explicit_date, 'package_date': self.package_date,
                           'date_heading': attrs.get('data-sf-role') == 'toggleLink'}
        if tag in {'iframe', 'embed'} and attrs.get('src'):
            self.links.append({'href': attrs['src'], 'text': 'Embedded document', 'row': self.row,
                               'position': len(''.join(self.parts)), 'section_date': self.last_explicit_date})
        if attrs.get('data-pdf'):
            self.links.append({'href': attrs['data-pdf'], 'text': attrs.get('title', 'Meeting document'), 'row': self.row,
                               'position': len(''.join(self.parts)), 'section_date': self.last_explicit_date})

    def handle_data(self, value):
        if self.hidden:
            return
        value = re.sub(r'\s+', ' ', value)
        if re.search(r'\b20\d\d\b', value):
            parsed_date = association_date(value)
            if parsed_date:
                self.last_explicit_date = parsed_date
        self.parts.append(value + ' ')
        if self.row is not None:
            self.rows[self.row].append(value)
        if self.anchor is not None:
            self.anchor['text'] += value + ' '

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.hidden = max(0, self.hidden - 1)
        if tag == 'a' and self.anchor is not None:
            if self.anchor.get('date_heading'):
                self.package_date = association_date(self.anchor['text']) or self.package_date
            self.links.append(self.anchor)
            self.anchor = None
        if tag == 'tr':
            self.row = None


def page_links(raw: bytes, base: str) -> list[dict]:
    parser = _Page()
    parser.feed(raw.decode('utf-8-sig', errors='replace'))
    text = ''.join(parser.parts)
    result = []
    for link in parser.links:
        href = urljoin(base, link['href'])
        if not href.startswith('https://'):
            continue
        result.append({'url': normalized_url(href), 'title': link['text'].strip(), 'section_date': link['section_date'],
                       'package_date': link.get('package_date'),
                       'context': ' '.join(parser.rows[link['row']]) if link['row'] is not None else
                       text[max(0, link['position'] - 250):link['position'] + 250]})
    return result


def fdot_award_rows(raw: bytes, parent_url: str, start: str, end: str) -> list[dict]:
    """Extract an observed contract/contractor/date row with its exact headers.

    In particular, preserve INTENT TO AWARD alongside DATE AWARDED rather than
    silently converting the mixed source labels into an unconditional award.
    """
    html = raw.decode('utf-8', errors='replace')
    observed_links = page_links(raw, parent_url)
    result = []
    def plain(fragment):
        parser = _Page()
        parser.feed(fragment)
        return re.sub(r'\s+', ' ', ''.join(parser.parts)).strip()
    for table_no, table in enumerate(re.findall(r'<table\b[^>]*>(.*?)</table>', html, re.S | re.I)):
        rows = re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, re.S | re.I)
        if not rows:
            continue
        headers = [plain(cell) for cell in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', rows[0], re.S | re.I)]
        if len(headers) != 3 or 'DATE AWARDED' not in headers[2].upper() or 'CONTRACTOR' not in headers[1].upper():
            continue
        for row_no, row_html in enumerate(rows[1:], 1):
            cells = re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row_html, re.S | re.I)
            values = [plain(cell) for cell in cells]
            if len(values) != 3 or not re.fullmatch(r'[A-Z][A-Z0-9]{3,8}', values[0]):
                continue
            links = page_links(cells[0].encode('utf-8'), parent_url)
            urls = sorted({link['url'] for link in links})
            if not urls:
                continue
            row_text = ' '.join(values)
            letting_days = {link.get('package_date') for link in observed_links
                            if link['url'] in urls and re.sub(r'\s+', ' ', link['context']).strip() == row_text}
            letting_days.discard(None)
            letting_day = next(iter(letting_days)) if len(letting_days) == 1 else None
            award_day = association_date(values[2])
            day = award_day or letting_day
            if not day or not letting_day or not start <= letting_day <= end:
                continue
            text = '\n'.join(f'{header}: {value}' for header, value in zip(headers, values))
            result.append({'project_id': values[0], 'contractor_name': values[1], 'award_date': award_day,
                           'letting_date': letting_day, 'association_date': day,
                           'related_bid_document_urls': urls, 'text': text,
                           'source_table_headers': headers, 'source_row_values': values,
                           'source_table_ordinal': table_no, 'source_row_ordinal': row_no,
                           'source_row_html_sha256': digest(row_html), 'normalized_source_excerpt': True})
    return result


class _FolderConfig(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.csrf, self.config = None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name') == 'csrftoken':
            self.csrf = attrs.get('content')
        if attrs.get('data-cftp'):
            self.config = json.loads(attrs['data-cftp'])


def association_date(value: str, *, year: int | None = None) -> str | None:
    """A date extracted from a meeting URL/label is explicitly not public time."""
    value = unquote(value)
    patterns = [r'(20\d\d)[_/-](\d{2})[_/-](\d{2})', r'/commission/(20\d\d)/(\d{2})(\d{2})/']
    for pattern in patterns:
        m = re.search(pattern, value)
        if m:
            try:
                return date(*map(int, m.groups())).isoformat()
            except ValueError:
                pass
    m = re.search(r'\b(\d{1,2})/(\d{1,2})/(20\d\d)\b', value)
    if m:
        try:
            return date(int(m[3]), int(m[1]), int(m[2])).isoformat()
        except ValueError:
            pass
    m = re.search(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?', value, re.I)
    if m and (m[3] or year):
        month = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'].index(m[1][:3].lower()) + 1
        try:
            return date(int(m[3] or year), month, int(m[2])).isoformat()
        except ValueError:
            pass
    return None


def _kind(title: str, url: str) -> str:
    value = (title + ' ' + url).lower()
    if re.search(r'minute|recap|[?&]m=m(?:&|$)', value):
        return 'minutes'
    if re.search(r'agenda|[?&]m=a(?:&|$)', value):
        return 'agenda'
    return 'report'


def _evidence_link(link: dict) -> bool:
    value = (link['url'] + ' ' + link['title']).lower()
    return bool(re.search(r'\.pdf(?:[?#]|$)|downloadpdf|downloadagenda|/attachments?/|view\.ashx|/public/agenda/|/documents?/', value))


def _pdf_text(path: Path) -> tuple[str, str]:
    exe = shutil.which('pdftotext')
    if exe:
        args = [exe, '-layout', str(path), '-']
    else:
        python = str(PDF_PYTHON) if PDF_PYTHON.exists() else sys.executable
        code = 'import sys; from pypdf import PdfReader; r=PdfReader(sys.argv[1]); print("\\n\\f\\n".join(p.extract_text() or "" for p in r.pages))'
        args = [python, '-X', 'utf8', '-c', code, str(path)]
    run = subprocess.run(args, capture_output=True, timeout=90,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if run.returncode:
        return '', 'pdf_extraction_failed'
    text = run.stdout.decode('utf-8', errors='replace')
    return text, 'extracted_pdf_text_layout_unverified' if len(text.strip()) >= 50 else 'pdf_empty_or_scanned_no_ocr'


def _pdf_links(path: Path, base: str) -> list[dict]:
    """Read public hyperlinks in a commission agenda, without opening any URI."""
    python = str(PDF_PYTHON) if PDF_PYTHON.exists() else sys.executable
    code = ('import sys,json; from pypdf import PdfReader; r=PdfReader(sys.argv[1]); '
            'links=[str(a.get_object().get("/A",{}).get("/URI","")) '
            'for p in r.pages for a in p.get("/Annots",[])]; print(json.dumps(links))')
    try:
        run = subprocess.run([python, '-X', 'utf8', '-c', code, str(path)], capture_output=True,
                             timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('PDF hyperlink extraction unavailable or timed out') from exc
    if run.returncode:
        raise RuntimeError('PDF hyperlink extraction unavailable')
    links = []
    directory = base.rsplit('/', 1)[0] + '/'
    for href in json.loads(run.stdout):
        url = urljoin(base, href)
        # TxDOT agendas link the same meeting's exhibits. Never crawl arbitrary
        # outside websites, email links, or other years mentioned in a PDF.
        if url.startswith(directory) and url.lower().endswith('.pdf') and url != base:
            links.append({'url': normalized_url(url), 'title': url.rsplit('/', 1)[-1]})
    return list({link['url']: link for link in links}.values())


def extract(raw: bytes, content_type: str, path: Path) -> tuple[str, str]:
    if raw.startswith(b'%PDF'):
        try:
            return _pdf_text(path)
        except (OSError, subprocess.TimeoutExpired):
            return '', 'pdf_extraction_unavailable_or_timeout'
    if raw.startswith(b'PK'):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                info = archive.getinfo('word/document.xml')
                if info.file_size > MAX_BYTES:
                    return '', 'docx_xml_exceeds_byte_cap'
                tree = ElementTree.fromstring(archive.read(info))
                ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                text = '\n'.join(''.join(node.text or '' if node.tag == ns + 't' else
                                         '\t' if node.tag == ns + 'tab' else
                                         '\n' if node.tag in {ns + 'br', ns + 'cr'} else ''
                                         for node in paragraph.iter())
                                 for paragraph in tree.iter(ns + 'p'))
                return text, 'extracted_docx_main_text_layout_unverified'
        except (KeyError, zipfile.BadZipFile, ElementTree.ParseError):
            return '', 'unsupported_or_malformed_zip_document'
    if 'json' in content_type.lower():
        try:
            obj = json.loads(raw)
        except ValueError:
            return '', 'invalid_json_document'
        if isinstance(obj, dict) and isinstance(obj.get('MatterTextPlain'), str):
            # Current status/action fields remain in the raw metadata archive;
            # model evidence contains only the requested legislative text.
            return obj['MatterTextPlain'], 'extracted_matter_text_current_version_unverified'
        return '', 'unsupported_json_document'
    if 'html' in content_type.lower() or b'<html' in raw[:2000].lower() or b'<!doctype' in raw[:2000].lower():
        if b'loadDocumentIntoWebViewer' in raw:
            return '', 'html_document_viewer_shell'
        parser = _FilingText()
        parser.feed(raw.decode('utf-8-sig', errors='replace'))
        text = '\n'.join(re.sub(r'\s+', ' ', x).strip() for x in ''.join(parser.parts).splitlines() if x.strip())
        return text, 'extracted_html_current_version_unverified' if len(text) >= 50 else 'html_empty'
    if content_type.startswith('text/'):
        return raw.decode('utf-8-sig', errors='replace'), 'extracted_plain_text'
    return '', 'unsupported_document_type'


class _SharedCapture:
    """Thread coordination only; each worker owns its SQLite connection."""
    def __init__(self):
        self.guard = threading.Lock()
        self.io_lock = threading.RLock()
        self.url_locks, self.host_locks, self.last_host = {}, {}, {}

    def url_lock(self, url):
        with self.guard:
            return self.url_locks.setdefault(url, threading.Lock())

    def wait_for_host(self, host):
        with self.guard:
            lock = self.host_locks.setdefault(host, threading.Lock())
        with lock:
            time.sleep(max(0, self.last_host.get(host, 0) + 1 - time.monotonic()))
            self.last_host[host] = time.monotonic()


class Capture:
    def __init__(self, root: Path, store: Store, *, live: bool, maximum: int, source_maximum: int | None = None,
                 shared: _SharedCapture | None = None):
        self.root, self.store, self.live, self.maximum = root, store, live, maximum
        self.documents, self.errors, self.sightings = [], [], []
        self.last_host, self.seen, self.new_requests = {}, {}, 0
        self.source_maximum = source_maximum
        self.shared = shared or _SharedCapture()
        self.requests = root / 'sources' / 'requests'
        self.requests.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> tuple[bytes, dict]:
        url = normalized_url(url)
        with self.shared.url_lock(url):
            return self._get(url)

    def _get(self, url: str) -> tuple[bytes, dict]:
        target = self.requests / (digest(url) + '.json')
        if target.exists():
            receipt = json.loads(target.read_text(encoding='utf-8'))
            raw = (self.root / receipt['blob_path']).read_bytes()
            if digest(raw) != receipt['sha256']:
                raise ValueError('Cached procurement bytes changed')
            if receipt['status'] != 200:
                raise RuntimeError(f'Previously failed URL retained without retry: HTTP {receipt["status"]}')
            return raw, receipt
        if not self.live:
            raise RuntimeError('Public source not yet captured; live=False')
        if self.new_requests >= max(1000, self.maximum * 5):
            raise RuntimeError('Global request cap reached')
        host = urlparse(url).hostname
        self.shared.wait_for_host(host)
        start = time.monotonic()
        status, content_type, error, final_url = 0, '', None, url
        try:
            req = Request(url, headers={'User-Agent': 'JevAlphaResearch/1.0 (public procurement research)', 'Accept': '*/*'})
            with urlopen(req, timeout=35) as response:
                status, content_type = response.status, response.headers.get('Content-Type', '')
                final_url = normalized_url(response.url)
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    error, status = 'response_exceeds_byte_cap', 0
                    raw = raw[:MAX_BYTES]
        except HTTPError as exc:
            status, raw, error = exc.code, exc.read(MAX_BYTES), f'HTTP {exc.code}'
        except (OSError, URLError, ValueError) as exc:
            raw, error = b'', str(exc)
        self.new_requests += 1
        captured = utc_now()
        with self.shared.io_lock:
            sha = self.store.put_blob(raw)
            observation_id = self.store.observe(url, raw, kind='procurement_public_source', status=status, document_id=None)
        receipt = {'url': url, 'final_url': final_url, 'status': status, 'content_type': content_type,
                   'sha256': sha, 'blob_path': (self.store.blobs / sha).relative_to(self.root).as_posix(),
                   'captured_at': captured, 'observation_id': observation_id, 'bytes': len(raw), 'error': error,
                   'timings_ms': {'fetch': round((time.monotonic() - start) * 1000, 3)}}
        write_new_json(target, receipt)
        if status != 200 or error:
            raise RuntimeError(f'Capture failed: {error or status}')
        return raw, receipt

    def award_row(self, source_id: str, parent_url: str, receipt: dict, row: dict) -> dict:
        """Exact public table excerpt, linked to the immutable full HTML source."""
        identity = parent_url + '|award-row|' + str(row['source_table_ordinal']) + '|' + str(row['source_row_ordinal'])
        key = (source_id, identity)
        if key in self.seen:
            return self.seen[key]
        if len(self.documents) >= self.maximum:
            raise RuntimeError('Per-source document cap reached; award rows incomplete')
        text = row['text']
        text_path = self.root / 'sources/text' / (digest(text) + '.txt')
        with self.shared.io_lock:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            if not text_path.exists():
                text_path.write_bytes(text.encode('utf-8'))
        doc = {**{k: v for k, v in row.items() if k != 'text'},
               'document_id': digest(source_id + '|' + identity), 'source_id': source_id,
               'url': parent_url, 'parent_source_url': parent_url, 'parent_source_sha256': receipt['sha256'],
               'sha256': receipt['sha256'], 'blob_path': receipt['blob_path'], 'captured_at': receipt['captured_at'],
               'published_at': None, 'availability': 'unverified_historical', 'historical_bytes_verified': False,
               'kind': 'award_row', 'title': 'FDOT contract table row ' + row['project_id'],
               'association_dates': [row['association_date']], 'source_package_dates': [row['letting_date']],
               'association_date_basis': 'Explicit DATE AWARDED table cell; current HTML historical version unverified',
               'package': parent_url + '#letting-' + row['letting_date'],
               'text_path': text_path.relative_to(self.root).as_posix(), 'text_characters': len(text),
               'extraction_status': 'extracted_html_exact_table_row', 'timings_ms': {'extract': 0}}
        self.documents.append(doc)
        self.seen[key] = doc
        self.sightings.append({'source_id': source_id, 'url': parent_url, 'document_id': doc['document_id'],
                               'association_date': doc['association_date'], 'package': doc['package']})
        checkpoint = self.root / 'sources/documents' / (doc['document_id'] + '.json')
        if not checkpoint.exists():
            write_new_json(checkpoint, doc)
        return doc

    def folder_listing(self, folder_url: str) -> dict:
        """One-level anonymous FDOT listing; the POST only reads directory data."""
        folder_url = normalized_url(folder_url)
        cache = self.root / 'sources/folder-indexes' / (digest(folder_url) + '.json')
        if cache.exists():
            return json.loads(cache.read_text(encoding='utf-8'))
        if not self.live:
            raise RuntimeError('Public folder listing not captured; live=False')
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        receipts = []
        def read(url, data=None):
            url = normalized_url(url)
            self.shared.wait_for_host(urlparse(url).hostname)
            started = time.monotonic()
            headers = {'User-Agent': 'Mozilla/5.0 (public procurement research)', 'Referer': folder_url}
            if data is not None:
                headers.update({'Content-Type': 'application/x-www-form-urlencoded', 'X-Requested-With': 'XMLHttpRequest'})
            status, error, raw, content_type = 0, None, b'', ''
            try:
                with opener.open(Request(url, data=data, headers=headers), timeout=35) as response:
                    status, content_type = response.status, response.headers.get('Content-Type', '')
                    raw = response.read(MAX_BYTES + 1)
                    if len(raw) > MAX_BYTES:
                        raw, error = raw[:MAX_BYTES], 'response_exceeds_byte_cap'
            except HTTPError as exc:
                status, raw, error = exc.code, exc.read(MAX_BYTES), f'HTTP {exc.code}'
            except (OSError, URLError, ValueError) as exc:
                error = str(exc)
            self.new_requests += 1
            with self.shared.io_lock:
                sha = self.store.put_blob(raw)
                observation = self.store.observe(url, raw, kind='procurement_public_folder_listing', status=status, document_id=None)
            receipt = {'url': url, 'method': 'POST_READ_ONLY' if data is not None else 'GET',
                       'status': status, 'content_type': content_type, 'sha256': sha,
                       'blob_path': (self.store.blobs / sha).relative_to(self.root).as_posix(),
                       'captured_at': utc_now(), 'observation_id': observation,
                       'timings_ms': {'fetch': round((time.monotonic() - started) * 1000, 3)}, 'error': error}
            receipts.append(receipt)
            if status != 200 or error:
                raise RuntimeError(f'Public folder capture failed: {error or status}')
            return raw
        result = {'folder_url': folder_url, 'receipts': receipts, 'files': [], 'errors': [],
                  'policy_version': FOLDER_POLICY['version']}
        try:
            page = _FolderConfig()
            page.feed(read(folder_url).decode('utf-8', errors='replace'))
            if not page.csrf or not page.config or not re.fullmatch(r'[A-Za-z0-9_-]+', page.config.get('fileId', '')):
                raise ValueError('Public folder CSRF/config unavailable; accessible HTML not expanded')
            form = {'cd': '/', 'sEcho': '1', 'iColumns': '5', 'iDisplayStart': '0',
                    'iDisplayLength': str(FOLDER_POLICY['listing_rows']), 'sSearch': '', 'bRegex': 'false',
                    'iSortingCols': '0', 'sColumns': '', 'csrftoken': page.csrf}
            for i, prop in enumerate(['fname', 'type', 'size', 'time', '']):
                form.update({f'mDataProp_{i}': prop, f'bSearchable_{i}': 'true', f'bSortable_{i}': 'true',
                             f'sSearch_{i}': '', f'bRegex_{i}': 'false'})
            endpoint = urljoin(folder_url, '/public/op/' + page.config['fileId'] + '/get_dir')
            payload = json.loads(read(endpoint, urlencode(form).encode('utf-8')))
            if not isinstance(payload, dict):
                raise ValueError('Public folder listing did not return an object')
            rows = payload.get('aaData')
            if not isinstance(rows, list):
                raise ValueError('Public folder returned no aaData array')
            result.update(rows=rows, total_records=payload.get('iTotalRecords'),
                          total_display_records=payload.get('iTotalDisplayRecords'))
            if int(payload.get('iTotalDisplayRecords', len(rows))) > len(rows):
                result['errors'].append('listing_row_cap; unreturned rows remain a coverage gap')
            for row in rows:
                if not isinstance(row, dict):
                    result['errors'].append('unrecognized_public_folder_row')
                    continue
                name = row.get('fname', '')
                if row.get('type') == 'file' and isinstance(name, str) and name not in {'', '.', '..'} and not re.search(r'[/\\]', name):
                    result['files'].append({'name': name, 'url': folder_url.rstrip('/') + '/' + quote(name, safe='') + '?dm=0',
                                             'listed_modified_at_unverified': row.get('time')})
                else:
                    result['errors'].append('nonfile_or_unsafe_child_not_expanded: ' + str(name))
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            result['errors'].append(str(exc))
        write_new_json(cache, result)
        return result

    def document(self, source_id: str, url: str, day: str | None, title: str, kind: str, package: str) -> dict:
        url = normalized_url(url)
        key = (source_id, url)
        self.sightings.append({'source_id': source_id, 'url': url, 'association_date': day, 'package': package})
        if key in self.seen:
            row = self.seen[key]
            days = sorted({d for d in row.get('association_dates', [row.get('association_date')]) + [day] if d})
            row['association_dates'] = days
            row['association_date'] = days[0] if days else None
            return row
        if self.source_maximum is not None and sum(d['source_id'] == source_id for d in self.documents) >= self.source_maximum:
            raise RuntimeError('Per-source document cap reached; package incomplete')
        if len(self.documents) >= self.maximum:
            raise RuntimeError('Global document cap reached; census incomplete')
        checkpoint = self.root / 'sources' / 'documents' / (digest(source_id + '|' + url) + '.json')
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding='utf-8'))
            if row.get('blob_path') and digest((self.root / row['blob_path']).read_bytes()) != row.get('sha256'):
                raise ValueError('Checkpoint source hash mismatch')
            row.update(association_date=day, association_dates=[day] if day else [], package=package)
            # Preserve old checkpoints while correcting derived extraction in
            # this new immutable manifest. No public URL is fetched again.
            if row.get('blob_path') and (row.get('extraction_status') == 'unsupported_document_type' or '/documents/fileviewerorpublic/' in url.lower()):
                raw, receipt = self.get(url)
                start = time.monotonic()
                content_type = 'text/csv' if urlparse(url).path.lower().endswith('.csv') else receipt['content_type']
                text, status = extract(raw, content_type, self.root / receipt['blob_path'])
                text_path = self.root / 'sources/text' / (digest(text) + '.txt')
                with self.shared.io_lock:
                    if not text_path.exists():
                        text_path.write_bytes(text.encode('utf-8'))
                row.update(text_path=text_path.relative_to(self.root).as_posix(), extraction_status=status,
                           text_characters=len(text), derived_extraction_version=2,
                           timings_ms={**receipt['timings_ms'], 'extract': round((time.monotonic() - start) * 1000, 3)})
            self.documents.append(row)
            self.seen[key] = row
            return row
        row = {'document_id': digest(source_id + '|' + url), 'source_id': source_id, 'url': url,
               'association_date': day, 'published_at': None, 'availability': 'unverified_historical',
               'association_dates': [day] if day else [],
               'kind': kind, 'title': title, 'package': package,
               'historical_bytes_verified': False, 'retrospective_current_status_excluded': True}
        try:
            raw, receipt = self.get(url)
            start = time.monotonic()
            content_type = 'text/csv' if urlparse(url).path.lower().endswith('.csv') else receipt['content_type']
            text, status = extract(raw, content_type, self.root / receipt['blob_path'])
            text_path = self.root / 'sources' / 'text' / (digest(text) + '.txt')
            text_path.parent.mkdir(parents=True, exist_ok=True)
            with self.shared.io_lock:
                if not text_path.exists():
                    with text_path.open('xb') as stream:
                        stream.write(text.encode('utf-8'))
            row.update({k: receipt[k] for k in ('sha256', 'blob_path', 'captured_at')})
            row.update(text_path=text_path.relative_to(self.root).as_posix(), extraction_status=status,
                       text_characters=len(text), timings_ms={**receipt['timings_ms'], 'extract': round((time.monotonic() - start) * 1000, 3)})
        except (RuntimeError, ValueError) as exc:
            row.update(sha256=None, blob_path=None, text_path=None, captured_at=utc_now(),
                       extraction_status='capture_failed', error=str(exc), timings_ms={})
            # Failed transport bytes/receipt remain archived even when not evidence.
            receipt_path = self.requests / (digest(url) + '.json')
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                row.update({k: receipt[k] for k in ('sha256', 'blob_path', 'captured_at', 'timings_ms')})
        self.documents.append(row)
        self.seen[key] = row
        write_new_json(checkpoint, row)
        if len(self.documents) % 25 == 0:
            print(f'procurement capture {source_id}: {len(self.documents)} documents, {self.new_requests} new public read requests', flush=True)
        return row


def _json(capture: Capture, url: str):
    raw, _ = capture.get(url)
    return json.loads(raw)


def _choose(packages: list[dict], preflight: bool) -> list[dict]:
    ordered = sorted(packages, key=lambda p: digest(SEED + '|' + p['id']))
    return ordered[:1] if preflight else ordered


def _legistar(capture: Capture, source: dict, start: str, end: str, preflight: bool) -> dict:
    base = 'https://webapi.legistar.com/v1/' + source['client']
    stop = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    events, pages, ids = [], 0, set()
    for page in range(100):
        params = {'$filter': f"EventDate ge datetime'{start}' and EventDate lt datetime'{stop}'", '$orderby': 'EventId asc', '$top': '1000', '$skip': str(page * 1000)}
        payload = _json(capture, base + '/Events?' + urlencode(params))
        if not isinstance(payload, list):
            raise ValueError('Legistar events did not return an array')
        pages += 1
        for event in payload:
            event_id = event.get('EventId')
            if event_id in ids:
                raise ValueError('Legistar repeated event across pages; inventory unstable')
            ids.add(event_id)
            day = str(event.get('EventDate', ''))[:10]
            if not start <= day <= end:
                raise ValueError('Legistar returned event outside frozen dates')
            events.append({'id': str(event_id), 'date': day, 'event': event})
        if len(payload) < 1000:
            break
    else:
        raise RuntimeError('Legistar pagination cap; inventory incomplete')
    result = {'source_id': source['id'], 'inventory_pages': pages, 'packages_in_window': len(events),
              'packages_selected': [], 'items': [], 'errors': [], 'inventory_complete': True,
              'inventory_packages': [{'id': e['id'], 'date': e['date'], 'title': e['event'].get('EventBodyName')} for e in events],
              'current_action_statuses_archived_but_not_interpreted': True}
    # Cancelled meetings stay in the inventory; preflight tests an available
    # package. Selection uses existence of files, never issuer or award content.
    testable = [p for p in events if p['event'].get('EventAgendaFile') or p['event'].get('EventMinutesFile')]
    if not events:
        result['errors'].append({'reason': 'no_events_in_frozen_window; source_gap_requires_audit'})
    evidence_capped = False
    for package in _choose(testable if preflight and testable else events, preflight):
        event, event_id, day = package['event'], package['id'], package['date']
        package_id = source['id'] + ':event:' + event_id
        result['packages_selected'].append({'id': package_id, 'date': day, 'title': event.get('EventBodyName')})
        try:
            items = _json(capture, base + '/Events/' + event_id + '/EventItems?AgendaNote=1&MinutesNote=0&Attachments=1')
            if not isinstance(items, list):
                raise ValueError('Legistar event items are not an array')
            for item in items:
                # DO NOT copy current action/passed/status/minutes notes into text inputs.
                result['items'].append({'package': package_id, 'association_date': day,
                                       'item_id': item.get('EventItemId'), 'matter_id': item.get('EventItemMatterId'),
                                       'title': item.get('EventItemTitle'), 'title_current_version_unverified': True,
                                       'procurement_title_match': bool(re.search(PROCUREMENT_PATTERN, item.get('EventItemTitle') or '', re.I))})
            if evidence_capped:
                continue
            for key, kind in [('EventAgendaFile', 'agenda'), ('EventMinutesFile', 'minutes')]:
                if event.get(key):
                    capture.document(source['id'], event[key], day, event.get('EventBodyName', '') + ' ' + kind, kind, package_id)
                else:
                    result['errors'].append({'package': package_id, 'reason': 'missing_' + kind})
            for item in items:
                if not re.search(PROCUREMENT_PATTERN, item.get('EventItemTitle') or '', re.I):
                    continue
                matter_id, version = item.get('EventItemMatterId'), str(item.get('EventItemVersion') or '')
                if matter_id:
                    try:
                        versions = _json(capture, f'{base}/Matters/{matter_id}/Versions')
                        matches = [v for v in versions if str(v.get('Value')) == version]
                        if not matches:
                            result['errors'].append({'package': package_id, 'matter_id': matter_id, 'reason': 'matching_event_item_text_version_unavailable'})
                        for match in matches:
                            capture.document(source['id'], f'{base}/Matters/{matter_id}/Texts/{match["Key"]}', day,
                                             item.get('EventItemTitle') or '', 'matter_text', package_id)
                    except (RuntimeError, ValueError, TypeError) as exc:
                        if 'cap reached' in str(exc):
                            raise
                        result['errors'].append({'package': package_id, 'matter_id': matter_id, 'reason': 'matter_text: ' + str(exc)})
                for attachment in item.get('EventItemMatterAttachments') or []:
                    href = attachment.get('MatterAttachmentHyperlink')
                    if not href and item.get('EventItemMatterId') and attachment.get('MatterAttachmentId'):
                        href = f"{base}/Matters/{item['EventItemMatterId']}/Attachments/{attachment['MatterAttachmentId']}/File"
                    if href:
                        capture.document(source['id'], href, day, attachment.get('MatterAttachmentName', ''),
                                         _kind(attachment.get('MatterAttachmentName', ''), href), package_id)
        except (RuntimeError, ValueError, TypeError) as exc:
            result['errors'].append({'package': package_id, 'reason': str(exc)})
            if 'cap reached' in str(exc):
                evidence_capped = True
    excluded = [item for item in result['items'] if not item['procurement_title_match']]
    result['title_exclusions_count'] = len(excluded)
    result['title_exclusion_audit_sample'] = sorted(excluded, key=lambda item: digest(SEED + '|' + str(item['item_id'])))[:10]
    result['selection_policy_version'] = SELECTION_POLICY['version']
    result['scope_caveat'] = 'Broad title screen can miss procurement hidden behind generic titles; excluded-item recall audit required'
    return result


def _html_packages(capture: Capture, source: dict, start: str, end: str) -> tuple[list[dict], list[dict]]:
    """Explicit source-specific index navigation; no unrestricted recursive crawler."""
    if start[:4] != end[:4]:
        if int(end[:4]) - int(start[:4]) > 1:
            raise ValueError('HTML inventory supports at most two calendar years per bounded collection')
        packages, indexes = [], []
        for y in range(int(start[:4]), int(end[:4]) + 1):
            child, index = _html_packages(capture, source, max(start, f'{y}-01-01'), min(end, f'{y}-12-31'))
            packages.extend(child)
            indexes.extend(index)
        return list({p['id']: p for p in packages}.values()), list({i['url']: i for i in indexes}.values())
    url, kind = source['url'], source['kind'].lower().replace('-', '_')
    year = int(start[:4])
    if 'port' in kind and 'seattle' in kind:
        url = url.split('?')[0] + '?' + urlencode({'filter[meeting_year]': year})
    raw, _ = capture.get(url)
    links = page_links(raw, url)
    indexes = [{'url': url, 'links': len(links)}]
    if 'txdot' in kind:
        # Current page links the agency's searchable historical archive. If the
        # year is unavailable without interactive search, retain that gap.
        year_links = [x for x in links if (str(year) in x['title'] or 'meeting archive' in x['title'].lower()) and urlparse(x['url']).hostname.endswith('txdot.gov')]
        for link in year_links[:3]:
            body, _ = capture.get(link['url'])
            links.extend(page_links(body, link['url']))
            indexes.append({'url': link['url']})
    elif 'fdot' in kind:
        year_links = [x for x in links if str(year) in x['title'] and 'letting' in x['title'].lower()]
        for link in year_links[:2]:
            body, _ = capture.get(link['url'])
            links.extend(page_links(body, link['url']))
            indexes.append({'url': link['url']})
        # FDOT uses monthly accordion headings; award-table dates underneath
        # them are later milestones and must not change the letting association.
        monthly = {}
        for link in links:
            day = link.get('package_date')
            host = urlparse(link['url']).hostname or ''
            if day and start <= day <= end and host in {'ftp.fdot.gov', 'bidletting.fdot.gov'}:
                package = monthly.setdefault(day, {'id': indexes[-1]['url'] + '#letting-' + day,
                    'date': day, 'url': indexes[-1]['url'], 'title': 'FDOT letting ' + day,
                    'index_only': True, 'links': []})
                package['links'].append(link)
        for package in monthly.values():
            package['links'] = list({link['url']: link for link in package['links']}.values())
        return list(monthly.values()), indexes
    packages = {}
    for link in links:
        href = link['url']
        if 'boardbook' in kind:
            relevant = '/public/agenda/' in href.lower()
        elif 'seattle' in kind:
            relevant = '/meeting/' in href or '/accessible/' in href
        elif 'txdot' in kind:
            relevant = f'/commission/{year}/' in href and ('agenda' in href.lower() or href.lower().endswith('.html'))
        elif 'fdot' in kind:
            relevant = 'letting' in (href + link['title']).lower() and (str(year) in (href + link['context']))
        else:
            relevant = False
        if not relevant:
            continue
        day = association_date(href) or association_date(link['context'], year=year) or link.get('section_date')
        if day and start <= day <= end:
            packages.setdefault(href, {'id': href, 'date': day, 'url': href, 'title': link['title']})
    return list(packages.values()), indexes


def _html_source(capture: Capture, source: dict, start: str, end: str, preflight: bool) -> dict:
    packages, indexes = _html_packages(capture, source, start, end)
    result = {'source_id': source['id'], 'indexes': indexes, 'packages_in_window': len(packages),
              'packages_selected': [], 'items': [], 'errors': [], 'inventory_complete': False,
              'inventory_packages': packages,
              'inventory_caveat': 'HTML archive enumeration requires manual gap audit; no complete-census claim'}
    if not packages:
        result['errors'].append({'reason': 'no_dated_packages_discovered; archive_access_or_parser_gap'})
    selected_packages = _choose(packages, preflight)
    folders = {}
    if source['kind'].lower() == 'fdot':
        selected_dates = {package['date'] for package in selected_packages}
        for parent_url in sorted({package['url'] for package in selected_packages}):
            raw, receipt = capture.get(parent_url)
            rows = fdot_award_rows(raw, parent_url, start, end)
            for row in rows:
                if row['letting_date'] in selected_dates:
                    try:
                        capture.award_row(source['id'], parent_url, receipt, row)
                    except RuntimeError as exc:
                        result['errors'].append({'url': parent_url, 'reason': str(exc)})
                        break
    for package in selected_packages:
        package_id, day, url = package['id'], package['date'], package['url']
        result['packages_selected'].append({'id': package_id, 'date': day, 'title': package['title']})
        try:
            if package.get('index_only'):
                for link in package['links']:
                    result['items'].append({'package': package_id, 'association_date': day, **link})
                for link in package['links']:
                    document = capture.document(source['id'], link['url'], day, link['title'], _kind(link['title'], link['url']), package_id)
                    if '/folder/' in link['url']:
                        folder = folders.setdefault(link['url'], {'document': document, 'sightings': []})
                        folder['sightings'].append({'package': package_id, 'date': day, 'title': link['title']})
                continue
            raw, receipt = capture.get(url)
            if raw.startswith(b'%PDF'):
                capture.document(source['id'], url, day, package['title'], _kind(package['title'], url), package_id)
                if source['kind'].lower() == 'txdot':
                    links = _pdf_links(capture.root / receipt['blob_path'], url)
                    for link in links:
                        result['items'].append({'package': package_id, 'association_date': day, **link})
                    for link in links:
                        capture.document(source['id'], link['url'], day, link['title'], _kind(link['title'], link['url']), package_id)
                continue
            capture.document(source['id'], url, day, package['title'], 'agenda_page', package_id)
            # Only one additional evidence-link level. Unfollowed child pages
            # remain explicitly visible in inventory rather than silent gaps.
            links = [link for link in page_links(raw, url) if _evidence_link(link)]
            if 'boardbook' in source['kind'].lower():
                parsed = urlparse(url)
                meeting = parse_qs(parsed.query).get('meeting', [None])[0]
                if meeting:
                    packet = re.sub('/Public/Agenda/', '/Public/DownloadAgenda/', url, flags=re.I)
                    links.insert(0, {'url': packet, 'title': 'Full agenda packet'})
            for link in links:
                result['items'].append({'package': package_id, 'association_date': day, 'title': link['title'], 'url': link['url']})
            for link in links:
                capture.document(source['id'], link['url'], day, link['title'], _kind(link['title'], link['url']), package_id)
                if '/documents/fileviewerorpublic/' in link['url'].lower():
                    viewer, _ = capture.get(link['url'])
                    children = page_links(viewer, link['url'])
                    html = viewer.decode('utf-8', errors='replace')
                    fmt = re.search(r"loadUrlFormat\s*=\s*['\"]([^'\"]+)['\"]", html)
                    file_id = re.search(r"loadDocumentIntoWebViewer\(['\"](\d+)['\"]", html)
                    if fmt and file_id:
                        children.append({'url': urljoin(link['url'], fmt[1].replace('TBR_AAAA_', file_id[1])), 'title': link['title']})
                    for child in children:
                        if _evidence_link(child) and child['url'] != link['url']:
                            result['items'].append({'package': package_id, 'association_date': day, **child})
                            capture.document(source['id'], child['url'], day, link['title'], _kind(link['title'], child['url']), package_id)
        except (RuntimeError, ValueError) as exc:
            result['errors'].append({'package': package_id, 'reason': str(exc)})
            if 'cap reached' in str(exc):
                break
    for index, (folder_url, folder) in enumerate(sorted(folders.items())):
        if index >= FOLDER_POLICY['max_folders']:
            result['errors'].append({'url': folder_url, 'reason': 'public_folder_count_cap; accessible format not expanded'})
            continue
        try:
            listing = capture.folder_listing(folder_url)
            result.setdefault('folder_listings', []).append(listing)
            folder['document']['folder_listing_captured'] = bool(listing.get('rows') is not None)
            for error in listing['errors']:
                result['errors'].append({'url': folder_url, 'reason': error})
            files = sorted(listing['files'], key=lambda item: item['name'])
            if len(files) > FOLDER_POLICY['max_children_per_folder']:
                result['errors'].append({'url': folder_url, 'reason': 'public_folder_child_cap; remaining files retained in inventory'})
            for child in files[:FOLDER_POLICY['max_children_per_folder']]:
                for sighting in folder['sightings']:
                    document = capture.document(source['id'], child['url'], sighting['date'],
                                                sighting['title'] + ': ' + child['name'], 'folder_child', sighting['package'])
                    document.update(parent_folder_url=folder_url, folder_listed_modified_at_unverified=child.get('listed_modified_at_unverified'))
                    document.update(evidence_role='inventory_only', historical_folder_membership_verified=False,
                                    folder_child_extraction_status=document['extraction_status'],
                                    folder_child_kind='folder_child', kind='inventory')
        except (RuntimeError, ValueError) as exc:
            result['errors'].append({'url': folder_url, 'reason': str(exc)})
            if 'cap reached' in str(exc):
                break
    return result


def balanced_sample(documents: list[dict], limit: int = 100, seed: str = SEED) -> list[dict]:
    """Stable round-robin hash sample across sources, including extraction failures."""
    if limit < 0:
        raise ValueError('Nonnegative sample limit required')
    groups = defaultdict(list)
    for doc in documents:
        if doc.get('kind') != 'inventory':
            groups[doc['source_id']].append(doc)
    for group in groups.values():
        group.sort(key=lambda d: digest(seed + '|' + d['document_id']))
    chosen = []
    ordered = sorted(groups, key=lambda source: digest(seed + '|' + source))
    while len(chosen) < limit and any(groups.values()):
        for source in ordered:
            if groups[source] and len(chosen) < limit:
                chosen.append(groups[source].pop(0))
    return chosen


def annotate_document_dates(root: Path, documents: list[dict]) -> None:
    """Keep FDOT's document dates distinct from all linked letting packages.

    Rescheduled projects can reuse older addenda in later letting sections.
    Neither an issue date nor a scheduled bid date proves first publication.
    """
    for document in documents:
        if '/documents/fileviewerorpublic/' in document.get('url', '').lower():
            document.update(kind='inventory', evidence_role='inventory_only',
                            extraction_status='html_document_viewer_shell')
            continue
        if document.get('source_id') != 'fdot' or not document.get('text_path'):
            continue
        if document.get('kind') == 'award_row':
            continue
        if document.get('parent_folder_url'):
            continue
        if '/folder/' in document.get('url', ''):
            document.update(kind='inventory', evidence_role='inventory_only',
                            extraction_status='html_folder_inventory_expanded' if document.get('folder_listing_captured') else 'html_folder_inventory_unexpanded')
            continue
        text = (root / document['text_path']).read_text(encoding='utf-8')
        stated = association_date(text[:2000])
        bid = re.search(r'\bBIDS\s+TO\s+BE\s+RECEIVED\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+20\d\d)', text[:4000], re.I)
        bid_day = association_date(bid[1]) if bid else None
        days = sorted({day for day in document.get('association_dates', []) if day})
        document.update(document_stated_date=stated, document_stated_date_is_publication_proof=False,
                        stated_bid_receipt_date=bid_day, source_package_dates=days,
                        association_date_basis='Earliest official letting-section association; repeated package dates retained',
                        reused_across_package_dates=len(days) > 1,
                        bid_date_differs_from_some_package_dates=bool(bid_day and any(day != bid_day for day in days)))
        if re.search(r'bid[-_\s]*tab', unquote(document.get('url', '')), re.I):
            first_page = text.split('\f')[0]
            ranking = re.search(r'Vendor Ranking', first_page)
            bid_status = re.search(r'Bid\s+Type\s+Bid\s+Status', first_page)
            if ranking and bid_status and not re.search(r'\baward(?:ed)?\b', text, re.I):
                document.update(source_stage='bid',
                    source_stage_evidence=[{'start': match.start(), 'end': match.end(), 'quote': match[0]}
                                           for match in (ranking, bid_status)],
                    source_stage_basis='FDOT bid-tab URL plus explicit first-page vendor/bid-ranking headers; '
                                       'Winning bid is a ranking label, not an independently observed award decision')
            footer_dates = sorted({day for value in re.findall(
                r'(\d{1,2}/\d{1,2}/20\d\d)\s*Florida Department of Transportation', text)
                                   if (day := association_date(value))})
            document.update(letting_dates=days, report_dates=footer_dates)
            if footer_dates:
                report_date = footer_dates[-1]
                document.update(report_date=report_date, report_date_basis='Explicit date immediately before Florida Department of Transportation footer',
                                association_date=max([report_date] + days),
                                association_date_basis='Later of explicit bid-report footer date and associated letting dates; publication still unverified')


def collect(root: Path, protocol: dict, *, live: bool = False, max_documents: int | None = None) -> dict:
    """Archive bounded public reads; a failed/incomplete source is never replaced."""
    root = Path(root).resolve()
    maximum = min(MAX_DOCUMENTS, max_documents if max_documents is not None else protocol.get('max_documents', MAX_DOCUMENTS))
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError('Positive integer max_documents required')
    start, end = protocol.get('historical_start', '2025-07-01'), protocol.get('historical_end', '2025-12-31')
    if date.fromisoformat(end) < date.fromisoformat(start):
        raise ValueError('Invalid historical window')
    sources = protocol.get('sources', [])
    if not sources or len({s['id'] for s in sources}) != len(sources) or len(sources) > 10:
        raise ValueError('Expected one to ten distinct frozen source definitions')
    preflight_mode = protocol.get('collection_mode') == 'preflight'
    target = root / 'sources'
    target.mkdir(parents=True, exist_ok=True)
    selection_path = target / ('selection-' + digest(canonical_json(SELECTION_POLICY)) + '.json')
    if not selection_path.exists():
        write_new_json(selection_path, SELECTION_POLICY)
    folder_policy_path = target / ('folder-policy-' + digest(canonical_json(FOLDER_POLICY)) + '.json')
    if not folder_policy_path.exists():
        write_new_json(folder_policy_path, FOLDER_POLICY)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = target / f'{"preflight" if preflight_mode else "census"}-{run_id}.json'
    lock = target / 'collection.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(run_id)
    try:
        shared = _SharedCapture()
        base_quota, remainder = divmod(maximum, len(sources))
        quotas = {s['id']: base_quota + (index < remainder) for index, s in enumerate(sources)}
        # Create the schema once before workers open their own connections.
        with Store(target / 'archive'):
            pass

        def collect_source(source):
            with shared.io_lock:
                store = Store(target / 'archive')
                store.db.execute('PRAGMA busy_timeout = 30000')
            with store:
                quota = quotas[source['id']]
                capture = Capture(root, store, live=live, maximum=quota, source_maximum=quota, shared=shared)
                try:
                    if source.get('client') or 'legistar' in source['kind'].lower():
                        result = _legistar(capture, source, start, end, preflight_mode)
                    else:
                        result = _html_source(capture, source, start, end, preflight_mode)
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                    result = {'source_id': source['id'], 'inventory_complete': False, 'packages_selected': [], 'items': [], 'errors': [{'reason': str(exc)}]}
                docs = capture.documents
                annotate_document_dates(root, docs)
                result['document_count'] = len(docs)
                result['inventory_document_count'] = sum(d.get('kind') == 'inventory' for d in docs)
                result['substantive_document_count'] = len(docs) - result['inventory_document_count']
                result['document_cap'] = quota
                result['extraction_success_count'] = sum(d.get('extraction_status', '').startswith('extracted_') for d in docs)
                result['complete'] = result.get('inventory_complete', False) and not result['errors'] and all(d.get('extraction_status', '').startswith('extracted_') for d in docs)
                write_new_json(target / 'source-results' / f'{run_id}-{source["id"]}.json', result)
                print(f'procurement source {source["id"]}: {len(docs)} documents, {len(result["errors"])} gaps', flush=True)
                return result, capture.documents, capture.sightings, capture.new_requests

        # map preserves frozen source order in the manifest regardless of which
        # network worker finishes first. Public starts remain <=1/second/host.
        with ThreadPoolExecutor(max_workers=min(4, len(sources))) as pool:
            batches = list(pool.map(collect_source, sources))
        results = [batch[0] for batch in batches]
        documents = [doc for batch in batches for doc in batch[1]]
        sightings = [item for batch in batches for item in batch[2]]
        sample = balanced_sample(documents)
        manifest = {'schema_version': 'procurement-source-manifest-v1', 'created_at': utc_now(),
                    'protocol_sha256': digest(canonical_json(protocol)), 'collection_mode': 'preflight' if preflight_mode else 'census',
                    'selection_policy': SELECTION_POLICY, 'selection_policy_path': selection_path.relative_to(root).as_posix(),
                    'folder_policy': FOLDER_POLICY, 'folder_policy_path': folder_policy_path.relative_to(root).as_posix(),
                    'historical_start': start, 'historical_end': end, 'live': live, 'max_documents': maximum,
                    'new_requests': sum(batch[3] for batch in batches), 'documents': documents, 'sightings': sightings,
                    'per_source_preflight_document_cap': base_quota if preflight_mode else None,
                    'per_source_document_cap': base_quota, 'source_document_quotas': quotas, 'concurrent_sources': min(4, len(sources)),
                    'source_results': results, 'balanced_sample_document_ids': [d['document_id'] for d in sample],
                    'complete': not preflight_mode and all(r['complete'] for r in results),
                    'publication_clock_verified': False, 'manifest_path': path.relative_to(root).as_posix()}
        write_new_json(path, manifest)
        return manifest
    finally:
        lock.unlink()


def preflight(root: Path, protocol: dict, *, live: bool = False, max_documents: int = 100) -> dict:
    """One frozen hash-selected historical package per source, never a full census."""
    return collect(root, {**protocol, 'collection_mode': 'preflight'}, live=live, max_documents=max_documents)
