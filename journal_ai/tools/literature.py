"""Literature search: a local TF-IDF corpus (offline, deterministic) with an
optional live OpenAlex backend for real novelty checking."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .text import TfIdfIndex

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def _corpus() -> tuple[list[dict[str, Any]], TfIdfIndex]:
    path = Path(os.getenv("JOURNAL_LITERATURE_CORPUS", DATA_DIR / "literature_corpus.json"))
    papers = json.loads(path.read_text(encoding="utf-8"))
    index = TfIdfIndex({p["id"]: f"{p['title']} {p['abstract']}" for p in papers})
    return papers, index


def search_local(query: str, k: int = 5) -> list[dict[str, Any]]:
    papers, index = _corpus()
    by_id = {p["id"]: p for p in papers}
    return [
        {**by_id[pid], "similarity": round(score, 4)}
        for pid, score in index.query(query, k)
        if score > 0
    ]


def search_openalex(query: str, k: int = 5) -> list[dict[str, Any]]:  # pragma: no cover - network
    import httpx

    resp = httpx.get(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": k, "select": "id,doi,title,publication_year,abstract_inverted_index"},
        timeout=20.0,
    )
    resp.raise_for_status()
    out = []
    for w in resp.json().get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((pos, word) for word, poss in inv.items() for pos in poss))
        out.append(
            {
                "id": w.get("id"),
                "doi": w.get("doi"),
                "title": w.get("title") or "",
                "year": w.get("publication_year"),
                "abstract": " ".join(word for _, word in words),
                "similarity": None,
            }
        )
    return out


def search_literature(query: str, k: int = 5) -> list[dict[str, Any]]:
    if os.getenv("JOURNAL_LITERATURE_LIVE", "").lower() in {"1", "true", "yes"}:
        try:
            return search_openalex(query, k)
        except Exception:  # fall back to the local corpus rather than failing a review
            pass
    return search_local(query, k)
