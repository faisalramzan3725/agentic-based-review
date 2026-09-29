"""MCP server for the agentic journal.

Two groups of tools:

* **Workflow tools** - let any MCP client (Claude Code, Claude Desktop, ...) run the
  journal inside a conversation: one-shot review of a paper, full submission ->
  review -> editor decision -> revision -> publication lifecycle, status and the
  concern-resolution matrix.
* **Service tools** - the low-level tools the agents themselves use (literature
  search, policy DB, duplicate check, manuscript DB, DOI registry, PDF validation,
  audit log). Agents call these over MCP when JOURNAL_TOOL_TRANSPORT=mcp.

Run as a stdio server (what MCP clients launch; see .mcp.json):
    python -m journal_ai.mcp_server --transport stdio
or as a network service (Kubernetes sidecar / docker-compose):
    python -m journal_ai.mcp_server --transport streamable-http --port 8765

Client config (Claude Desktop's claude_desktop_config.json or Claude Code's .mcp.json):
    {
      "mcpServers": {
        "journal-ai": {
          "command": "python",
          "args": ["-m", "journal_ai.mcp_server", "--transport", "stdio"],
          "env": {
            "PYTHONPATH": "C:\\\\Users\\\\Faisal Ramzan\\\\Desktop\\\\Journal Agentic AI",
            "JOURNAL_DATA_DIR": "C:\\\\Users\\\\Faisal Ramzan\\\\Desktop\\\\Journal Agentic AI\\\\var"
          }
        }
      }
    }
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import Settings
from .tools import LocalTools, ManuscriptStore


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    """Readable view of a manuscript's state for an MCP client."""
    decision = summary.get("decision") or {}
    reviews = summary.get("reviews") or []
    last_round = max((r["round"] for r in reviews), default=1)
    out = {
        "manuscript_id": summary["manuscript_id"],
        "title": summary.get("title"),
        "version": summary.get("version"),
        "domain": summary.get("domain"),
        "status": summary.get("status"),
        "round": summary.get("round"),
        "pending_action": summary.get("pending_action"),
        "screening_failures": [
            f"{c['name']}: {c['detail']}" for c in (summary.get("screening") or {}).get("checks", []) if not c["passed"]
        ],
        "reviewers": [a["reviewer_id"] for a in summary.get("assignments") or []],
        "reviews": [
            {
                "reviewer": r["reviewer_id"],
                "recommendation": r["recommendation"],
                "confidence": r["confidence"],
                "concerns": [f"{c['concern_id']} [{c['severity']}] {c['description']}" for c in r["concerns"]],
            }
            for r in reviews
            if r["round"] == last_round
        ],
        "agreement": (summary.get("aggregated") or {}).get("agreement"),
        "recommendation": decision.get("recommendation"),
        "rationale": decision.get("rationale"),
        "resolutions_this_round": [
            f"{r['concern_id']}: {r['status']}" + (f" - {r['remaining_issue']}" if r.get("remaining_issue") else "")
            for r in summary.get("resolutions") or []
            if r["round"] == summary.get("round")
        ],
    }
    if summary.get("publication"):
        out["doi"] = summary["publication"]["doi"]
    if summary.get("production"):
        out["pdf_path"] = summary["production"]["pdf_path"]
        out["print_ready"] = summary["production"]["print_ready"]
    return out


def build_server(settings: Settings | None = None, store: ManuscriptStore | None = None, service=None) -> MCPServer:
    settings = settings or Settings()
    settings.ensure_dirs()
    if store is None:
        store = service.store if service is not None else ManuscriptStore(settings.db_path)
    tools = LocalTools(store)
    server = MCPServer(
        name="journal-ai",
        instructions=(
            "Agentic scientific journal. Use review_manuscript for a quick multi-agent review of a paper "
            "(nothing is stored). Use submit_manuscript / submit_manuscript_file to start the full editorial "
            "workflow, then follow 'pending_action': editor_decision when it is 'editor_approval', "
            "submit_revision when it is 'revision_request'. get_concern_matrix shows how each reviewer "
            "concern was resolved across revision rounds."
        ),
    )

    # ------------------------------------------------------------ workflow
    state: dict[str, Any] = {"svc": service}
    lock = threading.Lock()

    def svc():
        with lock:
            if state["svc"] is None:
                from .service import JournalService

                inner = copy.copy(settings)
                inner.tool_transport = "local"  # the server's own agents call tools in-process
                state["svc"] = JournalService(inner)
            return state["svc"]

    def _submission(title, body, abstract, keywords, authors, article_type, figures, cover_letter):
        from .documents import extract_title_and_abstract
        from .schemas import ManuscriptSubmission

        _, found_abstract = extract_title_and_abstract(body)
        return ManuscriptSubmission(
            title=title,
            body=body,
            abstract=abstract or found_abstract or "",
            keywords=keywords or [],
            authors=authors or [],
            article_type=article_type,
            figures=figures or [],
            cover_letter=cover_letter,
        )

    def _load_file(path: str, metadata_path: str | None) -> tuple[str, dict[str, Any]]:
        from .documents import extract_title_and_abstract, load_manuscript_text

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"manuscript file not found: {path}")
        body = load_manuscript_text(p)
        meta: dict[str, Any] = {}
        if metadata_path:
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
        elif (p.parent / "meta.json").exists():
            meta = json.loads((p.parent / "meta.json").read_text(encoding="utf-8-sig"))
        title, _ = extract_title_and_abstract(body)
        meta.setdefault("title", title or p.stem)
        return body, meta

    def _one_shot_review(sub) -> dict[str, Any]:
        s = svc()
        team = s.team
        with lock:
            preview_id = f"PREVIEW-{s.store.next_sequence('preview'):05d}"
        ms = team.editorial_manager.intake(sub, preview_id)
        screening = team.editorial_manager.screen(ms)
        result: dict[str, Any] = {
            "title": ms.title,
            "domain": ms.domain,
            "screening_passed": screening.passed,
            "screening_failures": [f"{c.name}: {c.detail}" for c in screening.checks if not c.passed],
            "llm_mode": s.ctx.mode,
        }
        if not screening.passed:
            result["recommendation"] = "desk_reject"
            return result
        assignments = team.editorial_manager.assign_reviewers(ms)
        reports = [team.reviewers[a.reviewer_id].review(ms) for a in assignments]
        agg = team.editorial_manager.aggregate(ms, reports, 1)
        decision = team.decision.decide(ms, 1, agg, reports, None)
        result.update(
            {
                "reviewers": {a.reviewer_id: a.reason for a in assignments},
                "reviews": [
                    {
                        "reviewer": r.reviewer_id,
                        "recommendation": r.recommendation.value,
                        "confidence": r.confidence,
                        "concerns": [
                            {
                                "id": c.concern_id,
                                "severity": c.severity.value,
                                "issue": c.description,
                                "required_change": c.required_change,
                                "evidence": [f"{e.section}: {e.issue}" for e in c.evidence],
                            }
                            for c in r.concerns
                        ],
                    }
                    for r in reports
                ],
                "agreement": agg.agreement,
                "common_concerns": [cl.representative for cl in agg.clusters if cl.is_common],
                "conflicts": [c.detail for c in agg.conflicts],
                "recommendation": decision.recommendation.value,
                "rationale": decision.rationale,
                "decision_letter": decision.decision_letter,
            }
        )
        return result

    @server.tool(description=(
        "Quick multi-agent peer review of a paper WITHOUT starting a workflow or storing it: screening, "
        "domain classification, capability-based reviewer assignment, specialised reviews with concern ids, "
        "aggregation (agreement/conflicts) and the editorial recommendation with a decision letter."
    ))
    def review_manuscript(
        title: str,
        body: str,
        abstract: str = "",
        keywords: list[str] | None = None,
        article_type: str = "research_article",
    ) -> dict[str, Any]:
        return _one_shot_review(_submission(title, body, abstract, keywords, None, article_type, None, ""))

    @server.tool(description=(
        "Quick multi-agent review of a manuscript file (.md, .txt or .pdf) without storing it. "
        "metadata_path may point to a JSON file with title/authors/keywords; a meta.json next to the file is used automatically."
    ))
    def review_manuscript_file(path: str, metadata_path: str | None = None) -> dict[str, Any]:
        from .schemas import ManuscriptSubmission

        body, meta = _load_file(path, metadata_path)
        sub = ManuscriptSubmission(**meta, body=body)
        if not sub.abstract:
            sub = _submission(sub.title, body, "", sub.keywords, [a.model_dump() for a in sub.authors],
                              sub.article_type, sub.figures, sub.cover_letter)
        return _one_shot_review(sub)

    @server.tool(description=(
        "Submit a manuscript and start the full editorial workflow (screening -> parallel review -> editor gate). "
        "authors: list of {name, email, affiliation, corresponding}. Returns the manuscript id and pending_action."
    ))
    def submit_manuscript(
        title: str,
        body: str,
        abstract: str = "",
        keywords: list[str] | None = None,
        authors: list[dict[str, Any]] | None = None,
        article_type: str = "research_article",
        figures: list[str] | None = None,
        cover_letter: str = "",
    ) -> dict[str, Any]:
        s = svc()
        ms_id = s.submit(_submission(title, body, abstract, keywords, authors, article_type, figures, cover_letter))
        return _compact(s.summary(ms_id))

    @server.tool(description="Submit a manuscript file (.md/.txt/.pdf) and start the full editorial workflow.")
    def submit_manuscript_file(path: str, metadata_path: str | None = None) -> dict[str, Any]:
        from .schemas import ManuscriptSubmission

        body, meta = _load_file(path, metadata_path)
        s = svc()
        sub = ManuscriptSubmission(**meta, body=body)
        if not sub.abstract:
            from .documents import extract_title_and_abstract

            sub.abstract = extract_title_and_abstract(body)[1] or ""
        return _compact(s.summary(s.submit(sub)))

    @server.tool(description="List the bundled sample manuscripts (for demos).")
    def list_sample_manuscripts() -> list[str]:
        from . import samples

        return samples.available()

    @server.tool(description="Submit one of the bundled sample manuscripts and start its workflow.")
    def submit_sample_manuscript(name: str) -> dict[str, Any]:
        from . import samples

        s = svc()
        return _compact(s.summary(s.submit(samples.load_submission(name))))

    @server.tool(description="Current state of a manuscript: status, pending_action, reviews, recommendation, resolutions, DOI/PDF.")
    def get_manuscript_status(manuscript_id: str) -> dict[str, Any]:
        return _compact(svc().summary(manuscript_id))

    @server.tool(description=(
        "Handling editor's decision when pending_action is 'editor_approval'. decision: accept | minor_revision | "
        "major_revision | reject. It may differ from the agents' recommendation (the override is audited)."
    ))
    def editor_decision(manuscript_id: str, decision: str, editor: str = "handling-editor", notes: str = "") -> dict[str, Any]:
        from .schemas import HumanDecision

        return _compact(svc().editor_decision(manuscript_id, HumanDecision(decision=decision, editor=editor, notes=notes)))

    @server.tool(description=(
        "Author revision when pending_action is 'revision_request'. Give the revised text as body or body_path "
        "(.md/.txt/.pdf), and the response to reviewers (cite concern ids) as response_to_reviewers or response_path."
    ))
    def submit_revision(
        manuscript_id: str,
        body: str | None = None,
        body_path: str | None = None,
        response_to_reviewers: str = "",
        response_path: str | None = None,
    ) -> dict[str, Any]:
        from .documents import extract_title_and_abstract, load_manuscript_text
        from .schemas import RevisionSubmission

        if body_path:
            body = load_manuscript_text(body_path)
        if not body:
            raise ValueError("provide body or body_path")
        if response_path:
            response_to_reviewers = Path(response_path).read_text(encoding="utf-8-sig")
        rev = RevisionSubmission(body=body, abstract=extract_title_and_abstract(body)[1],
                                 response_to_reviewers=response_to_reviewers)
        return _compact(svc().submit_revision(manuscript_id, rev))

    @server.tool(description="Author withdraws the manuscript (only while a revision is requested).")
    def withdraw_manuscript(manuscript_id: str) -> dict[str, Any]:
        return _compact(svc().withdraw(manuscript_id))

    @server.tool(description="Concern x round matrix: every reviewer concern and its addressed / partially_addressed / not_addressed status per revision round.")
    def get_concern_matrix(manuscript_id: str) -> list[dict[str, Any]]:
        return svc().concern_matrix(manuscript_id)

    # ------------------------------------------------------ service tools
    @server.tool(description="Search related literature (local corpus, or OpenAlex when enabled). Returns title, year, doi, abstract, similarity.")
    def search_literature(query: str, k: int = 5) -> list[dict[str, Any]]:
        return tools.search_literature(query, k)

    @server.tool(description="Find previously submitted manuscripts whose text is similar above a threshold (duplicate submission check).")
    def check_duplicate(manuscript_id: str, text: str, threshold: float = 0.8) -> list[dict[str, Any]]:
        return tools.check_duplicate(manuscript_id, text, threshold)

    @server.tool(description="Effective journal policy (required sections, declarations, limits) for an article type and domain.")
    def get_policy(article_type: str = "research_article", domain: str | None = None) -> dict[str, Any]:
        return tools.get_policy(article_type, domain)

    @server.tool(description="Journal scope: keyword vocabulary per subject domain.")
    def journal_scope() -> dict[str, list[str]]:
        return tools.journal_scope()

    @server.tool(description="Fetch a manuscript (latest version unless version is given) from the manuscript database.")
    def get_manuscript(manuscript_id: str, version: int | None = None) -> dict[str, Any] | None:
        ms = store.get(manuscript_id, version)
        return ms.model_dump(mode="json") if ms else None

    @server.tool(description="List manuscripts with their latest version and workflow status.")
    def list_manuscripts() -> list[dict[str, Any]]:
        return store.list_manuscripts()

    @server.tool(description="Register (mint) a DOI for an accepted manuscript.")
    def register_doi(manuscript_id: str, prefix: str) -> dict[str, Any]:
        return tools.register_doi(manuscript_id, prefix)

    @server.tool(description="Validate a generated PDF: page count, extractable text, metadata, uniform page size.")
    def validate_pdf(path: str) -> dict[str, Any]:
        return tools.validate_pdf(path)

    @server.tool(description="Append an event to the manuscript's audit trail.")
    def log_event(manuscript_id: str, agent: str, event: str, payload: dict[str, Any] | None = None) -> str:
        tools.log_event(manuscript_id, agent, event, payload or {})
        return "ok"

    @server.tool(description="Audit trail (all events) for a manuscript.")
    def get_audit_trail(manuscript_id: str) -> list[dict[str, Any]]:
        return store.events(manuscript_id)

    return server


def main() -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        from mcp.server.transport_security import TransportSecuritySettings

        # inside the cluster the Service DNS name is the Host header; restrict via NetworkPolicy instead
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        server.run(args.transport, host=args.host, port=args.port, transport_security=security)


if __name__ == "__main__":  # pragma: no cover
    main()
