"""LLM backends.

* ``ClaudeLLM`` - Claude via the official Anthropic SDK, structured outputs
  (``messages.parse`` with a Pydantic schema), adaptive thinking, prompt caching
  of the manuscript so parallel reviewers share one cached prefix, and
  server-side refusal fallbacks.
* ``None`` backend - agents fall back to their deterministic rule-based
  implementation. That keeps the whole pipeline runnable and testable offline,
  and doubles as a transparent baseline for the research evaluation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMRefusalError(RuntimeError):
    pass


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage) -> None:  # anthropic Usage object
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def snapshot(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


SYSTEM_PROMPT = (
    "You are part of an editorial office for a peer-reviewed scientific journal. "
    "You act as one specialised agent in a multi-agent workflow; your output is consumed "
    "by other agents, so it must be precise, evidence-based and grounded in the manuscript text. "
    "Cite the section (and, where possible, the sentence or table) for every issue you raise. "
    "Do not invent content that is not in the manuscript. Severity scale: 'critical' means the "
    "conclusions are not supported or the work cannot be published without it; 'major' means a "
    "substantive change is required; 'minor' means a presentation-level fix."
)


class ClaudeLLM:
    def __init__(self, settings: Settings, client=None):
        import anthropic

        self.settings = settings
        self.client = client or anthropic.Anthropic()
        self.usage = UsageMeter()

    def structured(
        self,
        *,
        context: str,
        instruction: str,
        schema: type[T],
        effort: str | None = None,
    ) -> T:
        """One structured call. ``context`` (the manuscript) is cached and shared
        across agents; ``instruction`` is the agent-specific task."""
        response = self.client.beta.messages.parse(
            model=self.settings.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            output_format=schema,
            output_config={"effort": effort or self.settings.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage.add(response.usage)
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"Model declined: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise RuntimeError(f"Structured output incomplete (stop_reason={response.stop_reason})")
        return response.parsed_output


def _has_claude_credentials() -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE"):
        return True
    if shutil.which("ant"):
        try:
            out = subprocess.run(["ant", "auth", "status"], capture_output=True, text=True, timeout=10)
            return out.returncode == 0 and "no active" not in (out.stdout + out.stderr).lower()
        except Exception:
            return False
    return False


def build_llm(settings: Settings) -> ClaudeLLM | None:
    backend = settings.llm_backend.lower()
    if backend == "heuristic":
        return None
    if backend == "claude" or (backend == "auto" and _has_claude_credentials()):
        return ClaudeLLM(settings)
    return None
