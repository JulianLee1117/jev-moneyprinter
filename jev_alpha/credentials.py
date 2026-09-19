"""Read one explicitly named local credential; never echo or interpolate it."""

from __future__ import annotations

import os
from pathlib import Path


ALLOWED_KEYS = frozenset({"OPENROUTER_API_KEY", "SCRY_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"})


def read_key(name: str, path: str | Path = ".env") -> str | None:
    """Read only an allowlisted credential, without interpolation or env mutation."""
    if not isinstance(name, str) or name not in ALLOWED_KEYS:
        raise ValueError("Unsupported credential name")
    label = "OpenRouter" if name == "OPENROUTER_API_KEY" else name
    current = os.environ.get(name)
    if current and current.strip():
        return current.strip()
    file = Path(path)
    if not file.is_file():
        return None
    if file.stat().st_size > 100_000:
        raise ValueError("Local environment file exceeds size limit")
    found = None
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        entry_name, separator, value = line.partition("=")
        if not separator or entry_name.strip() != name:
            continue
        value = value.strip()
        if value.startswith(("\"", "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith("#")):
                raise ValueError(f"Malformed local {label} credential entry")
            value = value[1:end]
        else:
            value = value.split(" #", 1)[0].strip()
        if found is not None:
            raise ValueError(f"Duplicate local {label} credential entries")
        found = value
    if found and any(ord(c) < 33 or ord(c) > 126 for c in found):
        raise ValueError(f"Local {label} credential has invalid characters")
    return found or None


def openrouter_key(path: str | Path = ".env") -> str | None:
    """Backward-compatible OpenRouter credential entry point."""
    return read_key("OPENROUTER_API_KEY", path)
