"""Loader for the bundled sample manuscripts (data/samples/<name>/)."""

from __future__ import annotations

import json
from pathlib import Path

from .documents import extract_title_and_abstract
from .schemas import ManuscriptSubmission, RevisionSubmission

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def available() -> list[str]:
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if (p / "meta.json").exists())


def load_submission(name: str) -> ManuscriptSubmission:
    d = SAMPLES_DIR / name
    meta = json.loads(_read(d / "meta.json"))
    body = _read(d / "manuscript.md")
    _, abstract = extract_title_and_abstract(body)
    return ManuscriptSubmission(**meta, abstract=meta.get("abstract") or abstract or "", body=body)


def load_revision(name: str, n: int) -> RevisionSubmission:
    d = SAMPLES_DIR / name
    body = _read(d / f"revision_{n}.md")
    _, abstract = extract_title_and_abstract(body)
    resp = d / f"response_{n}.md"
    return RevisionSubmission(body=body, abstract=abstract, response_to_reviewers=_read(resp) if resp.exists() else "")
