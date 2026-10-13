"""common_io — shared I/O helpers for FlightDock tools."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Union


def ensure_output_dir(path: Union[str, Path]) -> None:
    """Create parent directories for *path* if they don't already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def timestamp_str() -> str:
    """Return an ISO-8601 timestamp string (local time, no timezone)."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Union[str, Path], data: Any) -> None:
    """Write *data* as pretty-printed JSON to *path*, creating dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
