"""Opt-in curl transport for the two explicitly supported OpenRouter APIs.

Credentials and request JSON are sent through stdin as curl configuration, never
through arguments, temporary files, or environment changes. No retries, redirects,
shell execution, default curl configuration, or stderr/body error disclosures.
Config escaping reference: https://curl.se/docs/manpage.html#-K
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess

ALLOWED_ENDPOINTS = frozenset({
    "https://openrouter.ai/api/alpha/decisions",
    "https://openrouter.ai/api/v1/chat/completions",
})
MAX_RESPONSE_BYTES = 2_000_000
_STATUS_MARKER = b"\n__JEV_HTTP_STATUS__:"


class TransportError(RuntimeError):
    """Redacted failure; uncertain paid attempts must not be retried implicitly."""

    def __init__(self, message: str, *, status: int | None = None,
                 outcome_uncertain: bool = True):
        super().__init__(message)
        self.status = status
        self.outcome_uncertain = outcome_uncertain


def _config_quote(value: str) -> str:
    # curl config double-quoted parameters recognize these exact escape forms.
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\t", "\\t").replace("\n", "\\n")
               .replace("\r", "\\r").replace("\v", "\\v"))
    return '"' + escaped + '"'


def _invalid_constant(_value: str):
    raise ValueError("Nonfinite JSON")


class CurlJSONTransport:
    """One POST using an installed curl executable; construction performs no I/O."""

    def post(self, url: str, body: bytes, api_key: str, timeout: float = 45) -> bytes:
        if not isinstance(url, str) or url not in ALLOWED_ENDPOINTS:
            raise TransportError("Unsupported OpenRouter endpoint.", outcome_uncertain=False)
        if (not isinstance(api_key, str) or not api_key
                or any(ord(character) < 33 or ord(character) > 126 for character in api_key)):
            raise TransportError("API key contains invalid header characters.", outcome_uncertain=False)
        try:
            valid_timeout = (type(timeout) in (int, float)
                             and math.isfinite(timeout) and 0.1 <= timeout <= 120)
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise TransportError("Timeout must be finite and between 0.1 and 120 seconds.",
                                 outcome_uncertain=False)
        if not isinstance(body, bytes):
            raise TransportError("Request body must be UTF-8 JSON bytes.", outcome_uncertain=False)
        try:
            text = body.decode("utf-8")
            parsed = json.loads(text, parse_constant=_invalid_constant)
            if not isinstance(parsed, dict):
                raise ValueError("Object required")
        except (ValueError, UnicodeError, RecursionError):
            raise TransportError("Request body must be a finite UTF-8 JSON object.",
                                 outcome_uncertain=False) from None
        # Requiring an object also prevents data-binary's special @file behavior.
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if executable is None:
            raise TransportError("curl is unavailable; install it or use the urllib transport.",
                                 outcome_uncertain=False)
        config = "\n".join([
            "url = " + _config_quote(url),
            'request = "POST"',
            'proto = "=https"',
            'proto-redir = "=https"',
            "no-location",
            "max-redirs = 0",
            "retry = 0",
            "silent",
            "max-time = " + _config_quote(str(timeout)),
            "connect-timeout = " + _config_quote(str(min(timeout, 10))),
            "max-filesize = " + str(MAX_RESPONSE_BYTES),
            'header = "Content-Type: application/json"',
            'header = "Accept: application/json"',
            'user-agent = "jev-alpha-research/0.1"',
            "header = " + _config_quote("Authorization: Bearer " + api_key),
            "data-binary = " + _config_quote(text),
            "write-out = " + _config_quote(_STATUS_MARKER.decode("ascii") + "%{http_code}"),
            "",
        ]).encode("utf-8")
        try:
            result = subprocess.run(
                [executable, "-q", "--config", "-"],
                input=config, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=timeout + 5, check=False, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise TransportError(
                "curl timed out; no retry was attempted. Billing outcome is uncertain.") from None
        except OSError:
            raise TransportError("curl could not start; no request was completed.",
                                 outcome_uncertain=False) from None
        raw = result.stdout
        if not isinstance(raw, bytes):
            raise TransportError("curl returned invalid transport output; no retry was attempted.")
        # curl enforces max-filesize during transfer; also reject oversize output
        # here in case an older curl or unusual endpoint ignored that guard.
        if len(raw) > MAX_RESPONSE_BYTES + len(_STATUS_MARKER) + 3:
            raise TransportError("curl response exceeded the size limit; no retry was attempted.")
        parts = raw.rsplit(_STATUS_MARKER, 1)
        status = None
        response = b""
        if (len(parts) == 2 and len(parts[1]) == 3
                and all(48 <= digit <= 57 for digit in parts[1])):
            response, digits = parts
            value = int(digits)
            status = value if 100 <= value <= 599 else None
        if result.returncode != 0:
            raise TransportError(
                "curl transfer failed; no retry was attempted. Billing outcome is uncertain.",
                status=status)
        if status is None:
            raise TransportError("curl returned no valid HTTP status; no retry was attempted.")
        if not 200 <= status < 300:
            raise TransportError(
                f"OpenRouter returned HTTP {status}; no retry was attempted. Billing outcome may be uncertain.",
                status=status)
        if len(response) > MAX_RESPONSE_BYTES:
            raise TransportError("curl response exceeded the size limit; no retry was attempted.",
                                 status=status)
        return response
