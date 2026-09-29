"""LangGraph orchestration of the manuscript lifecycle.

    intake -> screening --(fail)--> desk_reject
                 |
               assign --Send x N--> review (parallel) --> aggregate --> decide --> editor_gate
                                                                          ^            |
    verify (parallel, per reviewer) <--Send x N-- author_revision <-------+-- revision |
                                                                                       |
                                     rejected <-- reject --------------------------------+
                                     publication --> production --> END  <-- accept ---+

``author_revision`` and ``editor_gate`` are human-in-the-loop interrupts; the
checkpointer persists state between them, so a manuscript can wait for weeks
and resume in a different process.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from .agents import AgentTeam
from .documents import attach_sections
from .schemas import (
    AggregatedReview,
    Concern,
    ConcernResolution,
    EditorialDecision,
    HumanDecision,
    Manuscript,
    ManuscriptSubmission,
    Recommendation,
    ResolutionStatus,
    ReviewReport,
    RevisionSubmission,
)
from .tools import ManuscriptStore


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)) and k.endswith("_s"):
            out[k] = round(out[k] + v, 4)
        else:
            out[k] = v
    return out


class JournalState(TypedDict, total=False):
    manuscript_id: str
    submission: dict
    manuscript: dict
    previous_manuscript: dict
    round: int
    status: str
    screening: dict
    assignments: list[dict]
    reviews: Annotated[list[dict], operator.add]
    resolutions: Annotated[list[dict], operator.add]
    open_concerns: list[dict]
    aggregated: dict
    decision: dict
    decisions: Annotated[list[dict], operator.add]
    human_decision: dict
    response_letters: Annotated[list[dict], operator.add]
    publication: dict
    production: dict
    audit: Annotated[list[dict], operator.add]
    metrics: Annotated[dict, merge_dicts]


class ReviewTask(TypedDict):
    reviewer_id: str
    manuscript: dict
    round: int


class VerifyTask(TypedDict):
    reviewer_id: str
    manuscript: dict
    previous_manuscript: dict
    concerns: list[dict]
    response_letter: str
    round: int


Notifier = Callable[[str, str, dict[str, Any]], None]


def build_graph(team: AgentTeam, store: ManuscriptStore, checkpointer=None, notifier: Notifier | None = None):
    ctx = team.editorial_manager.ctx
    settings = ctx.settings

    def audit(ms_id: str, agent: str, event: str, payload: dict | None = None) -> dict:
        ctx.tools.log_event(ms_id, agent, event, payload or {})
        return {"agent": agent, "event": event, "payload": payload or {}, "ts": time.time()}

    def set_status(ms_id: str, status: str, payload: dict | None = None) -> None:
        store.set_status(ms_id, status)
        if notifier:
            notifier(ms_id, status, payload or {})

    def timed(name: str, t0: float) -> dict:
        return {f"{name}_s": round(time.perf_counter() - t0, 4)}

    # ------------------------------------------------------------------ nodes
    def intake(state: JournalState) -> dict:
        t0 = time.perf_counter()
        sub = ManuscriptSubmission.model_validate(state["submission"])
        ms_id = state.get("manuscript_id") or store.next_manuscript_id()
        ms = team.editorial_manager.intake(sub, ms_id)
        store.save_version(ms)
        set_status(ms_id, "submitted", {"title": ms.title})
        return {
            "manuscript_id": ms_id,
            "manuscript": ms.model_dump(mode="json"),
            "round": 1,
            "status": "submitted",
            "audit": [audit(ms_id, "editorial_manager", "manuscript_received", {"version": 1, "domain": ms.domain})],
            "metrics": {**timed("intake", t0), "started_at": time.time(), "llm_mode": ctx.mode},
        }

    def screening(state: JournalState) -> dict:
        t0 = time.perf_counter()
        ms = Manuscript.model_validate(state["manuscript"])
        result = team.editorial_manager.screen(ms)
        status = "screening_passed" if result.passed else "screening_failed"
        store.set_status(ms.manuscript_id, status)
        failed = [c.name for c in result.checks if not c.passed]
        return {
            "screening": result.model_dump(mode="json"),
            "status": status,
            "audit": [audit(ms.manuscript_id, "editorial_manager", "screening", {"passed": result.passed, "failed": failed})],
            "metrics": timed("screening", t0),
        }

    def desk_reject(state: JournalState) -> dict:
        ms_id = state["manuscript_id"]
        failed = [c for c in state["screening"]["checks"] if not c["passed"] and c["blocking"]]
        decision = EditorialDecision(
            manuscript_id=ms_id,
            round=0,
            recommendation=Recommendation.REJECT,
            rationale=[f"Returned before peer review: {c['name']} ({c['detail']})" for c in failed],
            requires_human_approval=False,
        )
        set_status(ms_id, "desk_rejected", {"reasons": decision.rationale})
        return {
            "decision": decision.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json")],
            "status": "desk_rejected",
            "audit": [audit(ms_id, "editorial_manager", "desk_reject", {"reasons": decision.rationale})],
        }

    def assign(state: JournalState) -> dict:
        ms = Manuscript.model_validate(state["manuscript"])
        assignments = team.editorial_manager.assign_reviewers(ms)
        set_status(ms.manuscript_id, "under_review", {"reviewers": [a.reviewer_id for a in assignments]})
        return {
            "assignments": [a.model_dump(mode="json") for a in assignments],
            "status": "under_review",
            "audit": [
                audit(ms.manuscript_id, "editorial_manager", "reviewers_assigned", {"assignments": [a.model_dump() for a in assignments]})
            ],
        }

    def review(task: ReviewTask) -> dict:
        t0 = time.perf_counter()
        ms = Manuscript.model_validate(task["manuscript"])
        report = team.reviewers[task["reviewer_id"]].review(ms, task["round"])
        return {
            "reviews": [report.model_dump(mode="json")],
            "audit": [
                audit(
                    ms.manuscript_id, task["reviewer_id"], "review_submitted",
                    {"round": task["round"], "recommendation": report.recommendation.value, "concerns": len(report.concerns)},
                )
            ],
            "metrics": {f"review_{task['reviewer_id']}_s": round(time.perf_counter() - t0, 4)},
        }

    def aggregate(state: JournalState) -> dict:
        ms = Manuscript.model_validate(state["manuscript"])
        reports = [ReviewReport.model_validate(r) for r in state["reviews"] if r["round"] == 1]
        agg = team.editorial_manager.aggregate(ms, reports, 1)
        return {
            "aggregated": agg.model_dump(mode="json"),
            "audit": [
                audit(ms.manuscript_id, "editorial_manager", "reviews_aggregated",
                      {"agreement": agg.agreement, "common": len(agg.common_concerns), "conflicts": len(agg.conflicts)})
            ],
        }

    def decide(state: JournalState) -> dict:
        t0 = time.perf_counter()
        ms = Manuscript.model_validate(state["manuscript"])
        rnd = state["round"]
        reports = [ReviewReport.model_validate(r) for r in state["reviews"] if r["round"] == 1]
        agg = AggregatedReview.model_validate(state["aggregated"])
        resolutions = [ConcernResolution.model_validate(r) for r in state.get("resolutions", []) if r["round"] == rnd]
        decision = team.decision.decide(ms, rnd, agg, reports, resolutions if rnd > 1 else None)

        # concerns that stay open for the next revision round
        if rnd == 1:
            open_c = [c.model_dump(mode="json") for r in reports for c in r.concerns]
        else:
            still = {x.concern_id for x in resolutions if x.status != ResolutionStatus.ADDRESSED}
            open_c = [c for c in state.get("open_concerns", []) if c["concern_id"] in still]
        # notify the handling editor that a recommendation awaits approval
        set_status(ms.manuscript_id, "decision_pending",
                   {"round": rnd, "recommendation": decision.recommendation.value})
        return {
            "decision": decision.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json")],
            "open_concerns": open_c,
            "status": "decision_pending",
            "audit": [
                audit(ms.manuscript_id, "editorial_decision", "recommendation",
                      {"round": rnd, "recommendation": decision.recommendation.value, "rationale": decision.rationale})
            ],
            "metrics": timed(f"decision_round{rnd}", t0),
        }

    def editor_gate(state: JournalState) -> dict:
        decision = EditorialDecision.model_validate(state["decision"])
        ms_id = state["manuscript_id"]
        if settings.require_human_approval:
            answer = interrupt(
                {
                    "type": "editor_approval",
                    "manuscript_id": ms_id,
                    "round": decision.round,
                    "recommendation": decision.recommendation.value,
                    "rationale": decision.rationale,
                    "decision_letter": decision.decision_letter,
                }
            )
            human = HumanDecision.model_validate(answer)
        else:
            human = HumanDecision(decision=decision.recommendation, editor="auto", notes="human approval disabled")
        overridden = human.decision != decision.recommendation
        set_status(ms_id, f"decision_{human.decision.value}",
                   {"round": decision.round, "letter": decision.decision_letter, "overridden": overridden})
        return {
            "human_decision": human.model_dump(mode="json"),
            "status": f"decision_{human.decision.value}",
            "audit": [
                audit(ms_id, "human_editor" if settings.require_human_approval else "editorial_decision",
                      "final_decision",
                      {"decision": human.decision.value, "editor": human.editor, "notes": human.notes,
                       "agent_recommendation": decision.recommendation.value, "overridden": overridden})
            ],
        }

    def author_revision(state: JournalState) -> dict:
        ms_id = state["manuscript_id"]
        decision = state["decision"]
        answer = interrupt(
            {
                "type": "revision_request",
                "manuscript_id": ms_id,
                "round": state["round"],
                "decision_letter": decision["decision_letter"],
                "open_concerns": [
                    {k: c[k] for k in ("concern_id", "reviewer_id", "severity", "description", "required_change")}
                    for c in state.get("open_concerns", [])
                ],
            }
        )
        if isinstance(answer, dict) and answer.get("withdraw"):
            set_status(ms_id, "withdrawn")
            return {"status": "withdrawn", "audit": [audit(ms_id, "author", "withdrawn")]}
        rev = RevisionSubmission.model_validate(answer)
        prev = Manuscript.model_validate(state["manuscript"])
        new = prev.model_copy(
            update={
                "version": prev.version + 1,
                "body": rev.body,
                "abstract": rev.abstract if rev.abstract is not None else prev.abstract,
                "title": rev.title or prev.title,
                "sections": [],
            }
        )
        attach_sections(new)
        if rev.abstract is None:
            sec = new.section("abstract")
            if sec:
                new.abstract = sec.text
        store.save_version(new)
        rnd = state["round"] + 1
        set_status(ms_id, "revision_received", {"version": new.version})
        return {
            "manuscript": new.model_dump(mode="json"),
            "previous_manuscript": prev.model_dump(mode="json"),
            "round": rnd,
            "status": "revision_received",
            "response_letters": [{"round": rnd, "text": rev.response_to_reviewers}],
            "audit": [audit(ms_id, "author", "revision_submitted", {"version": new.version, "round": rnd})],
        }

    def verify(task: VerifyTask) -> dict:
        t0 = time.perf_counter()
        agent = team.reviewers[task["reviewer_id"]]
        res = agent.verify(
            Manuscript.model_validate(task["previous_manuscript"]),
            Manuscript.model_validate(task["manuscript"]),
            [Concern.model_validate(c) for c in task["concerns"]],
            task["response_letter"],
        )
        res = [r.model_copy(update={"round": task["round"]}) for r in res]
        ms_id = task["manuscript"]["manuscript_id"]
        return {
            "resolutions": [r.model_dump(mode="json") for r in res],
            "audit": [
                audit(ms_id, task["reviewer_id"], "concerns_verified",
                      {"round": task["round"], "statuses": {r.concern_id: r.status.value for r in res}})
            ],
            "metrics": {f"verify_{task['reviewer_id']}_r{task['round']}_s": round(time.perf_counter() - t0, 4)},
        }

    def rejected(state: JournalState) -> dict:
        set_status(state["manuscript_id"], "rejected")
        return {"status": "rejected", "metrics": {"finished_at": time.time()}}

    def publication(state: JournalState) -> dict:
        ms = Manuscript.model_validate(state["manuscript"])
        record = team.publication.publish(ms)
        store.set_status(ms.manuscript_id, "in_production")
        return {
            "publication": record.model_dump(mode="json"),
            "status": "in_production",
            "audit": [audit(ms.manuscript_id, "publication", "doi_registered",
                            {"doi": record.doi, "metadata_issues": record.metadata_issues})],
        }

    def production(state: JournalState) -> dict:
        from .schemas import PublicationRecord

        ms = Manuscript.model_validate(state["manuscript"])
        result = team.production.produce(ms, PublicationRecord.model_validate(state["publication"]))
        status = "published" if result.print_ready else "production_issues"
        set_status(ms.manuscript_id, status, {"doi": state["publication"]["doi"], "pdf": result.pdf_path})
        usage = ctx.llm.usage.snapshot() if ctx.llm else {"llm_calls": 0}
        return {
            "production": result.model_dump(mode="json"),
            "status": status,
            "audit": [audit(ms.manuscript_id, "production", "production_complete",
                            {"print_ready": result.print_ready, "failed": [c.name for c in result.checks if not c.passed]})],
            "metrics": {"finished_at": time.time(), **usage},
        }

    # ---------------------------------------------------------------- routing
    def after_screening(state: JournalState) -> str:
        return "assign" if state["screening"]["passed"] else "desk_reject"

    def fan_out_reviews(state: JournalState) -> list[Send]:
        return [
            Send("review", {"reviewer_id": a["reviewer_id"], "manuscript": state["manuscript"], "round": 1})
            for a in state["assignments"]
        ]

    def after_editor(state: JournalState) -> str:
        d = state["human_decision"]["decision"]
        if d == Recommendation.ACCEPT.value:
            return "publication"
        if d == Recommendation.REJECT.value:
            return "rejected"
        return "author_revision"

    def fan_out_verify(state: JournalState) -> list[Send] | str:
        if state.get("status") == "withdrawn":
            return END
        letter = state["response_letters"][-1]["text"] if state.get("response_letters") else ""
        by_reviewer: dict[str, list[dict]] = {}
        for c in state.get("open_concerns", []):
            by_reviewer.setdefault(c["reviewer_id"], []).append(c)
        if not by_reviewer:
            return "decide"
        return [
            Send(
                "verify",
                {
                    "reviewer_id": rid,
                    "manuscript": state["manuscript"],
                    "previous_manuscript": state["previous_manuscript"],
                    "concerns": concerns,
                    "response_letter": letter,
                    "round": state["round"],
                },
            )
            for rid, concerns in by_reviewer.items()
        ]

    # ------------------------------------------------------------------ graph
    g = StateGraph(JournalState)
    g.add_node("intake", intake)
    g.add_node("screening", screening)
    g.add_node("desk_reject", desk_reject)
    g.add_node("assign", assign)
    g.add_node("review", review)
    g.add_node("aggregate", aggregate)
    g.add_node("decide", decide)
    g.add_node("editor_gate", editor_gate)
    g.add_node("author_revision", author_revision)
    g.add_node("verify", verify)
    g.add_node("rejected", rejected)
    g.add_node("publication", publication)
    g.add_node("production", production)

    g.add_edge(START, "intake")
    g.add_edge("intake", "screening")
    g.add_conditional_edges("screening", after_screening, ["assign", "desk_reject"])
    g.add_edge("desk_reject", END)
    g.add_conditional_edges("assign", fan_out_reviews, ["review"])
    g.add_edge("review", "aggregate")
    g.add_edge("aggregate", "decide")
    g.add_edge("decide", "editor_gate")
    g.add_conditional_edges("editor_gate", after_editor, ["publication", "rejected", "author_revision"])
    g.add_conditional_edges("author_revision", fan_out_verify, ["verify", "decide", END])
    g.add_edge("verify", "decide")
    g.add_edge("rejected", END)
    g.add_edge("publication", "production")
    g.add_edge("production", END)

    return g.compile(checkpointer=checkpointer)


def build_checkpointer(settings):
    kind = settings.checkpointer.lower()
    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    if kind == "postgres":  # pragma: no cover - needs a running Postgres
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(settings.postgres_url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.ensure_dirs()
    return SqliteSaver(sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False))
