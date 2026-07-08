"""Shared file I/O helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any, *, default: Any = str) -> None:
    """Atomically write JSON — tmp + os.replace (same pattern as live/control.py).

    ``default`` handles non-JSON-native values (pd.Timestamp → str).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, default=default)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
