from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from journal_ai.agents import AgentContext, AgentTeam  # noqa: E402
from journal_ai.config import Settings  # noqa: E402
from journal_ai.service import JournalService  # noqa: E402
from journal_ai.tools import LocalTools, ManuscriptStore  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.llm_backend = "heuristic"
    s.data_dir = tmp_path / "var"
    s.checkpointer = "memory"
    s.tool_transport = "local"
    s.require_human_approval = True
    s.max_revision_rounds = 2
    s.min_reviewers = 3
    s.max_reviewers = 6
    s.assignment_seed = 7
    s.n8n_webhook_url = None
    s.ensure_dirs()
    return s


@pytest.fixture
def store(settings) -> ManuscriptStore:
    return ManuscriptStore(settings.db_path)


@pytest.fixture
def ctx(settings, store) -> AgentContext:
    return AgentContext(settings=settings, tools=LocalTools(store), llm=None)


@pytest.fixture
def team(ctx) -> AgentTeam:
    return AgentTeam.build(ctx)


@pytest.fixture
def svc(settings) -> JournalService:
    return JournalService(settings, llm=None)


@pytest.fixture
def intake(team):
    """Parse a bundled sample into a Manuscript (id MS-TEST-<name>)."""
    from journal_ai import samples

    def _load(name: str):
        return team.editorial_manager.intake(samples.load_submission(name), f"MS-TEST-{name}")

    return _load
