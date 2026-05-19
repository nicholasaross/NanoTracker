"""JSON / JSONL output: EventLog (crash-safe append) and save_json helpers.

Ported from NanoTracker's `EventLog` (nano_tracker.py line ~953) and
`save_json` (line ~943). Behaviour is unchanged so existing pulled
sessions remain readable by the new analysers and vice versa.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
from pathlib import Path
from typing import Any

from streettracker.common.schema import SessionMeta, TrackRecord


def format_wall(unix_ts: float) -> str:
    """Local-time ISO-8601 with timezone offset, e.g.
    ``2026-05-17T14:32:05+01:00``. Matches NanoTracker's `format_wall()`.
    """
    return (
        datetime.datetime.fromtimestamp(unix_ts)
        .astimezone()
        .isoformat(timespec="seconds")
    )


class EventLog:
    """Append-on-finalize JSONL event writer.

    Each finalized track is written as a single JSON line and immediately
    flushed + fsynced. A crash loses only tracks still active in memory
    at crash time — never tracks already finalized.

    The file handle is opened in append mode so re-running a session with
    the same output directory cleanly extends the existing log.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        # SIM115: file handle is intentionally long-lived (one open per session).
        self._fh = open(str(path), "a", encoding="utf-8")  # noqa: SIM115
        self.count = 0

    def append(self, record: TrackRecord | dict[str, Any]) -> None:
        d = record.to_json_dict() if isinstance(record, TrackRecord) else record
        self._fh.write(json.dumps(d, separators=(",", ":")) + "\n")
        self._fh.flush()
        # fsync can fail on tmpfs / NFS / etc. — not fatal.
        with contextlib.suppress(OSError):
            os.fsync(self._fh.fileno())
        self.count += 1

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_data_json(records: list[TrackRecord | dict[str, Any]], path: Path) -> None:
    """Write the records as a bare top-level array (for jq / pandas / SQL
    ingestion). Pretty-printed; size is small enough (~150 B/row) that
    even a 24h session is a few MB.
    """
    payload = [
        r.to_json_dict() if isinstance(r, TrackRecord) else r
        for r in records
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_meta_json(meta: SessionMeta | dict[str, Any], path: Path) -> None:
    payload = meta.to_json_dict() if isinstance(meta, SessionMeta) else meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_json(
    records: list[TrackRecord | dict[str, Any]],
    meta: SessionMeta | dict[str, Any],
    data_path: Path,
    meta_path: Path,
) -> None:
    """Convenience wrapper: write both data and meta JSON in one call.
    Matches NanoTracker's `save_json` signature."""
    save_data_json(records, data_path)
    save_meta_json(meta, meta_path)


def read_data_json(path: Path) -> list[TrackRecord]:
    """Read a `{session}_data.json` produced by `save_data_json`. Tolerates
    extra fields (forward-compatible) and missing optional ones."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TrackRecord.from_json_dict(r) for r in raw]


def read_events_jsonl(path: Path) -> list[TrackRecord]:
    """Read a `{session}_events.jsonl` produced by `EventLog`. Skips blank
    lines so partial-write tails are tolerated."""
    out: list[TrackRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(TrackRecord.from_json_dict(json.loads(line)))
    return out
