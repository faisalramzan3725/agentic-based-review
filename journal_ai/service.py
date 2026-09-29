"""Application service: wires settings, store, tools, LLM, agents, graph and
notifications together, and exposes the journal operations used by the REST
API, the CLI and the tests."""

from __future__ import annotations

import logging
import threading
from typing import Any

from langgraph.types import Command

from .agents import AgentContext, AgentTeam
from .config import Settings
from .graph import build_checkpointer, build_graph
from .llm import build_llm
from .schemas import HumanDecision, ManuscriptSubmission, RevisionSubmission
from .tools import LocalTools, MCPTools, ManuscriptStore

log = logging.getLogger(__name__)


class N8nNotifier:
    """Posts lifecycle events to an n8n webhook (e-mails, CRM updates, editor alerts).

    Failures are logged and swallowed: notifications must never break the
    editorial workflow.
    """

    def __init__(self, url: str | None):
        self.url = url
        self.sent: list[dict[str, Any]] = []

    def __call__(self, manuscript_id: str, status: str, payload: dict[str, Any]) -> None:
        event = {"manuscript_id": manuscript_id, "status": status, "payload": payload}
        self.sent.append(event)
        if not self.url:
            return
        try:
            import httpx

            httpx.post(self.url, json=event, timeout=5.0)
        except Exception as exc:  # pragma: no cover - network
            log.warning("n8n notification failed for %s: %s", manuscript_id, exc)


class JournalService:
    def __init__(self, settings: Settings | None = None, llm=..., tools=None, checkpointer=None):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.store = ManuscriptStore(self.settings.db_path)
        if tools is None:
            if self.settings.tool_transport == "mcp" and self.settings.mcp_server_url:
                tools = MCPTools(self.settings.mcp_server_url)
            else:
                tools = LocalTools(self.store)
        self.tools = tools
        self.llm = build_llm(self.settings) if llm is ... else llm
        self.ctx = AgentContext(settings=self.settings, tools=self.tools, llm=self.llm)
        self.team = AgentTeam.build(self.ctx)
        self.notifier = N8nNotifier(self.settings.n8n_webhook_url)
        self.checkpointer = checkpointer or build_checkpointer(self.settings)
        self.graph = build_graph(self.team, self.store, self.checkpointer, self.notifier)
        self._locks: dict[str, threading.Lock] = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _config(manuscript_id: str) -> dict:
        return {"configurable": {"thread_id": manuscript_id}}

    def _lock(self, manuscript_id: str) -> threading.Lock:
        return self._locks.setdefault(manuscript_id, threading.Lock())

    def pending_action(self, manuscript_id: str) -> dict[str, Any] | None:
        snap = self.graph.get_state(self._config(manuscript_id))
        for task in snap.tasks:
            for intr in task.interrupts:
                return intr.value
        return None

    def state(self, manuscript_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(manuscript_id)).values)

    # ------------------------------------------------------------- operations
    def new_manuscript_id(self) -> str:
        return self.store.next_manuscript_id()

    def submit(self, submission: ManuscriptSubmission, manuscript_id: str | None = None) -> str:
        ms_id = manuscript_id or self.new_manuscript_id()
        with self._lock(ms_id):
            self.graph.invoke({"submission": submission.model_dump(mode="json"), "manuscript_id": ms_id}, self._config(ms_id))
        return ms_id

    def _resume(self, manuscript_id: str, expected: str, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock(manuscript_id):
            pending = self.pending_action(manuscript_id)
            if not pending or pending.get("type") != expected:
                raise ValueError(
                    f"Manuscript {manuscript_id} is not awaiting '{expected}' "
                    f"(pending: {pending.get('type') if pending else 'nothing'})"
                )
            self.graph.invoke(Command(resume=value), self._config(manuscript_id))
        return self.summary(manuscript_id)

    def submit_revision(self, manuscript_id: str, revision: RevisionSubmission) -> dict[str, Any]:
        return self._resume(manuscript_id, "revision_request", revision.model_dump(mode="json"))

    def withdraw(self, manuscript_id: str) -> dict[str, Any]:
        return self._resume(manuscript_id, "revision_request", {"withdraw": True})

    def editor_decision(self, manuscript_id: str, decision: HumanDecision) -> dict[str, Any]:
        return self._resume(manuscript_id, "editor_approval", decision.model_dump(mode="json"))

    # ---------------------------------------------------------------- queries
    def summary(self, manuscript_id: str) -> dict[str, Any]:
        st = self.state(manuscript_id)
        if not st:
            raise KeyError(manuscript_id)
        ms = st.get("manuscript", {})
        return {
            "manuscript_id": manuscript_id,
            "title": ms.get("title"),
            "version": ms.get("version"),
            "domain": ms.get("domain"),
            "corresponding_author": next(
                (a.get("email") for a in ms.get("authors", []) if a.get("corresponding") and a.get("email")), None
            ),
            "status": st.get("status"),
            "round": st.get("round"),
            "pending_action": self.pending_action(manuscript_id),
            "screening": st.get("screening"),
            "assignments": st.get("assignments"),
            "aggregated": st.get("aggregated"),
            "decision": st.get("decision"),
            "human_decision": st.get("human_decision"),
            "reviews": st.get("reviews", []),
            "resolutions": st.get("resolutions", []),
            "publication": {k: v for k, v in (st.get("publication") or {}).items() if not k.endswith("_xml")} or None,
            "production": st.get("production"),
            "metrics": st.get("metrics"),
        }

    def audit_trail(self, manuscript_id: str) -> list[dict[str, Any]]:
        return self.store.events(manuscript_id)

    def concern_matrix(self, manuscript_id: str) -> list[dict[str, Any]]:
        """Concern x round resolution matrix - the core artefact of the revision cycle."""
        st = self.state(manuscript_id)
        rows: dict[str, dict[str, Any]] = {}
        for rep in st.get("reviews", []):
            for c in rep["concerns"]:
                rows[c["concern_id"]] = {
                    "concern_id": c["concern_id"],
                    "reviewer_id": c["reviewer_id"],
                    "severity": c["severity"],
                    "description": c["description"],
                    "rounds": {},
                }
        for r in st.get("resolutions", []):
            if r["concern_id"] in rows:
                rows[r["concern_id"]]["rounds"][str(r["round"])] = r["status"]
        return list(rows.values())
