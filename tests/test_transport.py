"""Offline curl transport tests. No network requests or real credentials."""

import json
import subprocess
import unittest
from unittest.mock import patch

from jev_alpha.transport import (CurlJSONTransport, MAX_RESPONSE_BYTES,
                                 TransportError, _STATUS_MARKER)

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def unquote_config(value):
    """Decode documented curl-config escapes for round-trip fixture assertions."""
    assert value[0] == value[-1] == '"'
    result = []
    index = 1
    escapes = {"\\": "\\", '"': '"', "t": "\t", "n": "\n", "r": "\r", "v": "\v"}
    while index < len(value) - 1:
        character = value[index]
        if character == "\\":
            index += 1
            character = escapes[value[index]]
        result.append(character)
        index += 1
    return "".join(result)


def completed(body=b'{"ok":true}', status=b"200", returncode=0):
    return subprocess.CompletedProcess(["curl"], returncode, body + _STATUS_MARKER + status)


class CurlTransportTests(unittest.TestCase):
    @patch("jev_alpha.transport.shutil.which", return_value="C:/curl.exe")
    @patch("jev_alpha.transport.subprocess.run")
    def test_secret_only_in_stdin_and_json_round_trips(self, run, which):
        body = json.dumps({"text": 'Quotes " and slash \\ plus\nnewline\ttab 鋼板',
                           "attempt": '\nheader = "Authorization: other"'},
                          ensure_ascii=False, indent=2).encode("utf-8")
        key = 'synthetic-key-with-"quote\\slash'
        run.return_value = completed()
        result = CurlJSONTransport().post(ENDPOINT, body, key, timeout=12)
        self.assertEqual(b'{"ok":true}', result)
        args, kwargs = run.call_args
        self.assertEqual(["C:/curl.exe", "-q", "--config", "-"], args[0])
        self.assertNotIn(key, str(args))
        self.assertNotIn("env", kwargs)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(17, kwargs["timeout"])
        self.assertEqual(subprocess.DEVNULL, kwargs["stderr"])
        config = kwargs["input"].decode("utf-8")
        lines = config.splitlines()
        data_line = next(line for line in lines if line.startswith("data-binary = "))
        self.assertEqual(body.decode("utf-8"), unquote_config(data_line.split(" = ", 1)[1]))
        headers = [unquote_config(line.split(" = ", 1)[1])
                   for line in lines if line.startswith("header = ")]
        self.assertEqual(1, len([header for header in headers if header.startswith("Authorization:")]))
        self.assertIn("Authorization: Bearer " + key, headers)
        self.assertIn("no-location", lines)
        self.assertIn("retry = 0", lines)
        self.assertIn("max-filesize = 2000000", lines)
        self.assertIn('proto = "=https"', lines)
        self.assertIn('connect-timeout = "10"', lines)
        run.assert_called_once()

    @patch("jev_alpha.transport.shutil.which", return_value="curl")
    @patch("jev_alpha.transport.subprocess.run")
    def test_embedded_status_marker_and_newlines_do_not_confuse_trailer(self, run, which):
        payload = b'{\n"text":"literal"}\n' + _STATUS_MARKER + b"418\n"
        run.return_value = completed(payload)
        self.assertEqual(payload, CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key"))

    @patch("jev_alpha.transport.subprocess.run")
    def test_rejects_unapproved_endpoints_and_injected_keys_before_subprocess(self, run):
        for url in ("http://openrouter.ai/api/alpha/decisions", ENDPOINT + "?x=1",
                    "https://evil.invalid/api/alpha/decisions", ENDPOINT + "/", None):
            with self.subTest(url=url), self.assertRaises(TransportError) as caught:
                CurlJSONTransport().post(url, b"{}", "synthetic-key")
            self.assertFalse(caught.exception.outcome_uncertain)
        for key in ("", " space", "bad\nheader", "bad\rheader", "bad\x00", "é", None):
            with self.subTest(key=key), self.assertRaises(TransportError):
                CurlJSONTransport().post(ENDPOINT, b"{}", key)
        run.assert_not_called()

    @patch("jev_alpha.transport.subprocess.run")
    def test_rejects_non_json_file_references_bad_utf8_and_timeouts(self, run):
        for body in (b"@private-file", b"\xff", b"[]", b'{"n":NaN}', b'{"x":"\x00"}', "{}"):
            with self.subTest(body=body), self.assertRaises(TransportError):
                CurlJSONTransport().post(ENDPOINT, body, "synthetic-key")
        for timeout in (0, -1, True, float("nan"), float("inf"), 121, 10 ** 1000):
            with self.subTest(timeout=timeout), self.assertRaises(TransportError):
                CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key", timeout=timeout)
        run.assert_not_called()

    @patch("jev_alpha.transport.shutil.which", return_value=None)
    @patch("jev_alpha.transport.subprocess.run")
    def test_missing_curl_does_not_start(self, run, which):
        with self.assertRaises(TransportError) as caught:
            CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")
        self.assertFalse(caught.exception.outcome_uncertain)
        run.assert_not_called()

    @patch("jev_alpha.transport.shutil.which", return_value="curl")
    @patch("jev_alpha.transport.subprocess.run")
    def test_http_error_and_redirect_body_redacted_no_retry(self, run, which):
        for status in (b"302", b"401", b"429", b"500"):
            run.reset_mock()
            run.return_value = completed(b"private error body", status)
            with self.subTest(status=status), self.assertRaises(TransportError) as caught:
                CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")
            self.assertEqual(int(status), caught.exception.status)
            self.assertTrue(caught.exception.outcome_uncertain)
            self.assertNotIn("private", str(caught.exception))
            run.assert_called_once()

    @patch("jev_alpha.transport.shutil.which", return_value="curl")
    @patch("jev_alpha.transport.subprocess.run")
    def test_timeout_process_failure_and_missing_trailer_are_redacted(self, run, which):
        run.side_effect = subprocess.TimeoutExpired(["curl"], 50, output=b"private output")
        with self.assertRaises(TransportError) as caught:
            CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")
        self.assertTrue(caught.exception.outcome_uncertain)
        self.assertNotIn("private", str(caught.exception))
        run.side_effect = None
        for fixture in (completed(status=b"000", returncode=28),
                        subprocess.CompletedProcess(["curl"], 0, b"private response without trailer"),
                        completed(status=b"20x"), completed(status=b"200\n")):
            run.return_value = fixture
            with self.assertRaises(TransportError):
                CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")

    @patch("jev_alpha.transport.shutil.which", return_value="curl")
    @patch("jev_alpha.transport.subprocess.run")
    def test_response_size_limit(self, run, which):
        run.return_value = completed(b"x" * MAX_RESPONSE_BYTES)
        self.assertEqual(MAX_RESPONSE_BYTES,
                         len(CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")))
        run.return_value = completed(b"x" * (MAX_RESPONSE_BYTES + 1))
        with self.assertRaises(TransportError):
            CurlJSONTransport().post(ENDPOINT, b"{}", "synthetic-key")


if __name__ == "__main__":
    unittest.main()
