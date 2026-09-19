import tempfile
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from jev_alpha.sources import FederalRegister, SourceError
from jev_alpha.store import Store


def record(number, day="2025-01-01"):
    return {"document_number": number, "title": "Corrosion-resistant steel", "publication_date": day}


class SourceTests(unittest.TestCase):
    def test_large_query_is_partitioned_before_pagination_limit(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            obs = store.observe("https://example.com", b"fixture", kind="test")
            client = FederalRegister(store)
            def response(url, **kwargs):
                query = parse_qs(urlparse(url).query)
                lo, hi = query["conditions[publication_date][gte]"][0], query["conditions[publication_date][lte]"][0]
                if lo != hi:
                    return {"count": 2001, "results": []}, obs
                return {"count": 1, "results": [record("2025-10001" if lo.endswith("01") else "2025-10002", lo)]}, obs
            with patch.object(client, "fetch_json", side_effect=response) as mock:
                result = client.discover("steel", "2025-01-01", "2025-01-02")
            self.assertEqual(result["documents_retrieved"], 2)
            self.assertEqual(mock.call_count, 3)

    def test_truncated_discovery_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            obs = store.observe("https://example.com", b"fixture", kind="test")
            client = FederalRegister(store)
            with patch.object(client, "fetch_json", return_value=({"count": 2, "results": [record("2025-10001")]}, obs)):
                with self.assertRaisesRegex(SourceError, "truncated"):
                    client.discover("steel", "2025-01-01", "2025-01-01")

    def test_daily_overflow_stops_instead_of_recursing_forever(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = FederalRegister(store)
            with patch.object(client, "fetch_json", return_value=({"count": 2001, "results": []}, 1)):
                with self.assertRaisesRegex(SourceError, "Daily search"):
                    client.discover("steel", "2025-01-01", "2025-01-01")

    def test_invalid_source_url_fails_without_network(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = FederalRegister(store)
            with patch("jev_alpha.sources.urlopen") as mock:
                for url in ("http://www.federalregister.gov/a", "https://example.com/a", "file:///etc/passwd"):
                    with self.assertRaises(SourceError):
                        client.fetch(url, kind="test")
                mock.assert_not_called()

    def test_search_second_page_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            obs = store.observe("https://example.com", b"fixture", kind="test")
            client = FederalRegister(store)
            first = [record(f"2025-{i:05d}") for i in range(1000)]
            with patch.object(client, "fetch_json", side_effect=[({"count": 1001, "results": first}, obs),
                                                                  ({"count": 1001, "results": [record("2025-10000")]}, obs)]) as mock:
                result = client.discover("steel", "2025-01-01", "2025-01-01")
            self.assertEqual(result["documents_retrieved"], 1001)
            self.assertIn("page=2", mock.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
