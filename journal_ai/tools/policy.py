"""Journal policy database (author guidelines as data, not prompt text)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def _policies() -> dict[str, Any]:
    path = Path(os.getenv("JOURNAL_POLICY_FILE", DATA_DIR / "journal_policy.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def get_policy(article_type: str = "research_article", domain: str | None = None) -> dict[str, Any]:
    """Effective policy = defaults <- article-type overrides <- domain overrides."""
    db = _policies()
    policy: dict[str, Any] = json.loads(json.dumps(db["default"]))
    for layer in (db.get("article_types", {}).get(article_type), db.get("domains", {}).get(domain or "")):
        if not layer:
            continue
        for key, value in layer.items():
            if isinstance(value, list) and isinstance(policy.get(key), list):
                policy[key] = list(dict.fromkeys(policy[key] + value))
            else:
                policy[key] = value
    policy["article_type"] = article_type
    policy["domain"] = domain
    return policy


def journal_scope() -> dict[str, list[str]]:
    return _policies()["scope"]
