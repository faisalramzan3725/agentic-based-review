"""Runtime configuration, read from environment variables (12-factor style so the
same image runs locally, in docker-compose and in Kubernetes)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # "auto" -> Claude when credentials exist, otherwise the offline heuristic backend.
    llm_backend: str = field(default_factory=lambda: os.getenv("JOURNAL_LLM_BACKEND", "auto"))
    model: str = field(default_factory=lambda: os.getenv("JOURNAL_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.getenv("JOURNAL_EFFORT", "high"))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("JOURNAL_DATA_DIR", "./var")))
    checkpointer: str = field(default_factory=lambda: os.getenv("JOURNAL_CHECKPOINTER", "sqlite"))
    postgres_url: str | None = field(default_factory=lambda: os.getenv("JOURNAL_POSTGRES_URL"))

    # Tools are called in-process by default; "mcp" routes them through the MCP server.
    tool_transport: str = field(default_factory=lambda: os.getenv("JOURNAL_TOOL_TRANSPORT", "local"))
    mcp_server_url: str | None = field(default_factory=lambda: os.getenv("JOURNAL_MCP_URL"))

    require_human_approval: bool = field(default_factory=lambda: _bool("JOURNAL_HUMAN_APPROVAL", True))
    max_revision_rounds: int = field(default_factory=lambda: int(os.getenv("JOURNAL_MAX_ROUNDS", "2")))
    min_reviewers: int = field(default_factory=lambda: int(os.getenv("JOURNAL_MIN_REVIEWERS", "3")))
    max_reviewers: int = field(default_factory=lambda: int(os.getenv("JOURNAL_MAX_REVIEWERS", "6")))
    # Seed for stochastic tie-breaking among equally qualified reviewers (None = random).
    assignment_seed: int | None = field(
        default_factory=lambda: int(os.environ["JOURNAL_ASSIGNMENT_SEED"])
        if os.getenv("JOURNAL_ASSIGNMENT_SEED")
        else 7
    )

    n8n_webhook_url: str | None = field(default_factory=lambda: os.getenv("N8N_WEBHOOK_URL"))
    doi_prefix: str = field(default_factory=lambda: os.getenv("JOURNAL_DOI_PREFIX", "10.5555"))
    journal_name: str = field(default_factory=lambda: os.getenv("JOURNAL_NAME", "Journal of Agentic Science"))
    journal_issn: str = field(default_factory=lambda: os.getenv("JOURNAL_ISSN", "0000-0000"))

    def ensure_dirs(self) -> None:
        for sub in ("", "pdf", "uploads"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "journal.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"
