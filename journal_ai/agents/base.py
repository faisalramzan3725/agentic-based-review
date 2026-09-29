from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings
from ..llm import ClaudeLLM
from ..tools import JournalTools


@dataclass
class AgentContext:
    """Everything an agent needs: configuration, its tools and (optionally) an LLM."""

    settings: Settings
    tools: JournalTools
    llm: ClaudeLLM | None = None

    @property
    def mode(self) -> str:
        return "claude" if self.llm else "heuristic"


def has_any(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def count_references(text: str) -> int:
    """Count entries in a references section: '[1] ...', '1. ...', '- ...' lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entries = [ln for ln in lines if re.match(r"^(\[\d+\]|\d+[.)]\s|[-*]\s)", ln)]
    return len(entries) if entries else len([ln for ln in lines if len(ln) > 30])


def numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?\s?%|\d+\.\d+", text)
