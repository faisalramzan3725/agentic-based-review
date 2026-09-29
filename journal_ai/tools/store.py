"""Manuscript repository (SQLite). Holds every manuscript version plus an
append-only event log that doubles as the editorial audit trail."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas import Manuscript

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manuscripts (
    manuscript_id TEXT NOT NULL,
    version       INTEGER NOT NULL,
    title         TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (manuscript_id, version)
);
CREATE TABLE IF NOT EXISTS status (
    manuscript_id TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    manuscript_id TEXT NOT NULL,
    ts            TEXT NOT NULL,
    agent         TEXT NOT NULL,
    event         TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ManuscriptStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ ids
    def next_manuscript_id(self, year: int | None = None) -> str:
        year = year or datetime.now(timezone.utc).year
        key = f"ms-{year}"
        with self._lock:
            row = self._conn.execute("SELECT value FROM counters WHERE name=?", (key,)).fetchone()
            value = (row["value"] if row else 0) + 1
            self._conn.execute(
                "INSERT INTO counters(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()
        return f"MS-{year}-{value:05d}"

    def next_sequence(self, name: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
            value = (row["value"] if row else 0) + 1
            self._conn.execute(
                "INSERT INTO counters(name, value) VALUES(?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, value),
            )
            self._conn.commit()
        return value

    # ---------------------------------------------------------- manuscripts
    def save_version(self, manuscript: Manuscript) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO manuscripts(manuscript_id, version, title, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    manuscript.manuscript_id,
                    manuscript.version,
                    manuscript.title,
                    manuscript.model_dump_json(),
                    _now(),
                ),
            )
            self._conn.commit()

    def get(self, manuscript_id: str, version: int | None = None) -> Manuscript | None:
        sql = "SELECT data FROM manuscripts WHERE manuscript_id=?"
        args: tuple[Any, ...] = (manuscript_id,)
        if version is not None:
            sql += " AND version=?"
            args += (version,)
        sql += " ORDER BY version DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return Manuscript.model_validate_json(row["data"]) if row else None

    def all_latest(self, exclude: str | None = None) -> list[Manuscript]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.data FROM manuscripts m JOIN ("
                " SELECT manuscript_id, MAX(version) v FROM manuscripts GROUP BY manuscript_id"
                ") x ON m.manuscript_id=x.manuscript_id AND m.version=x.v"
            ).fetchall()
        out = [Manuscript.model_validate_json(r["data"]) for r in rows]
        return [m for m in out if m.manuscript_id != exclude]

    def set_status(self, manuscript_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO status(manuscript_id, status, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(manuscript_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
                (manuscript_id, status, _now()),
            )
            self._conn.commit()

    def get_status(self, manuscript_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM status WHERE manuscript_id=?", (manuscript_id,)
            ).fetchone()
        return row["status"] if row else None

    def list_manuscripts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.manuscript_id, MAX(m.version) AS version, m.title, s.status "
                "FROM manuscripts m LEFT JOIN status s ON s.manuscript_id=m.manuscript_id "
                "GROUP BY m.manuscript_id ORDER BY m.manuscript_id"
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- events
    def log_event(self, manuscript_id: str, agent: str, event: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(manuscript_id, ts, agent, event, payload) VALUES (?, ?, ?, ?, ?)",
                (manuscript_id, _now(), agent, event, json.dumps(payload or {}, default=str)),
            )
            self._conn.commit()

    def events(self, manuscript_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, agent, event, payload FROM events WHERE manuscript_id=? ORDER BY id",
                (manuscript_id,),
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
