"""End-to-end LangGraph workflow tests: every lifecycle path, interrupts,
persistence across processes, and the MCP tool transport."""

import pytest

from journal_ai import samples
from journal_ai.evaluation import run_lifecycle
from journal_ai.schemas import HumanDecision, Recommendation, RevisionSubmission
from journal_ai.service import JournalService


def approve(svc, ms_id, expect=None):
    pending = svc.pending_action(ms_id)
    assert pending["type"] == "editor_approval"
    if expect:
        assert pending["recommendation"] == expect
    return svc.editor_decision(ms_id, HumanDecision(decision=pending["recommendation"], editor="Dr E"))


def test_full_lifecycle_three_rounds_to_publication(svc):
    ms_id = svc.submit(samples.load_submission("ml_optimizer"))
    s = svc.summary(ms_id)
    assert s["status"] == "decision_pending" and s["domain"] == "machine_learning"
    # all assigned reviewers ran in parallel and each produced a structured report
    assert {r["reviewer_id"] for r in s["reviews"]} == {a["reviewer_id"] for a in s["assignments"]}

    s = approve(svc, ms_id, "major_revision")
    assert s["pending_action"]["type"] == "revision_request"
    assert "SCI-001" in {c["concern_id"] for c in s["pending_action"]["open_concerns"]}

    svc.submit_revision(ms_id, samples.load_revision("ml_optimizer", 1))
    s = approve(svc, ms_id, "major_revision")
    assert {c["concern_id"] for c in s["pending_action"]["open_concerns"]} == {"SCI-003", "SCI-004", "SCI-005"}

    svc.submit_revision(ms_id, samples.load_revision("ml_optimizer", 2))
    s = approve(svc, ms_id, "accept")
    assert s["status"] == "published" and s["version"] == 3 and s["round"] == 3
    assert s["publication"]["doi"].startswith("10.5555/")
    assert s["production"]["print_ready"]

    matrix = {r["concern_id"]: r["rounds"] for r in svc.concern_matrix(ms_id)}
    assert matrix["SCI-004"] == {"2": "partially_addressed", "3": "addressed"}
    assert matrix["SCI-003"] == {"2": "not_addressed", "3": "addressed"}
    assert matrix["METH-002"] == {"2": "addressed"}

    events = [e["event"] for e in svc.audit_trail(ms_id)]
    for e in ("manuscript_received", "screening", "reviewers_assigned", "review_submitted", "reviews_aggregated",
              "recommendation", "final_decision", "revision_submitted", "concerns_verified", "doi_registered",
              "production_complete"):
        assert e in events
    assert events.count("final_decision") == 3
    assert svc.store.get(ms_id, 1).version == 1 and svc.store.get(ms_id).version == 3
    statuses = [n["status"] for n in svc.notifier.sent]
    assert statuses[0] == "submitted" and statuses[-1] == "published" and "decision_pending" in statuses


def test_desk_reject_skips_peer_review(svc):
    ms_id = svc.submit(samples.load_submission("incomplete"))
    s = svc.summary(ms_id)
    assert s["status"] == "desk_rejected" and s["pending_action"] is None
    assert s["reviews"] == [] and s["decision"]["recommendation"] == "reject"


def test_duplicate_submission_is_desk_rejected(svc):
    first = svc.submit(samples.load_submission("cv_segmentation"))
    second = svc.submit(samples.load_submission("cv_segmentation"))
    s = svc.summary(second)
    assert s["status"] == "desk_rejected"
    dup = next(c for c in s["screening"]["checks"] if c["name"] == "duplicate_submission")
    assert first in dup["detail"]


def test_unchanged_resubmissions_end_in_rejection(svc):
    ms_id = svc.submit(samples.load_submission("clinical_trial"))
    approve(svc, ms_id, "major_revision")
    same = RevisionSubmission(body=samples.load_submission("clinical_trial").body, response_to_reviewers="We disagree.")
    svc.submit_revision(ms_id, same)
    approve(svc, ms_id, "major_revision")
    svc.submit_revision(ms_id, same)
    s = approve(svc, ms_id, "reject")  # round 3 > max_revision_rounds with critical issues open
    assert s["status"] == "rejected"
    assert {r["status"] for r in s["resolutions"]} == {"not_addressed"}


def test_editor_can_override_the_agents(svc):
    ms_id = svc.submit(samples.load_submission("cv_segmentation"))
    assert svc.pending_action(ms_id)["recommendation"] == "minor_revision"
    s = svc.editor_decision(ms_id, HumanDecision(decision=Recommendation.ACCEPT, editor="Chief Editor", notes="trivial fix"))
    assert s["status"] == "published"
    final = [e for e in svc.audit_trail(ms_id) if e["event"] == "final_decision"][0]["payload"]
    assert final["overridden"] is True and final["agent_recommendation"] == "minor_revision"


def test_author_can_withdraw(svc):
    ms_id = svc.submit(samples.load_submission("clinical_trial"))
    approve(svc, ms_id)
    assert svc.withdraw(ms_id)["status"] == "withdrawn"
    assert svc.pending_action(ms_id) is None


def test_wrong_action_is_refused(svc):
    ms_id = svc.submit(samples.load_submission("cv_segmentation"))
    with pytest.raises(ValueError, match="not awaiting 'revision_request'"):
        svc.submit_revision(ms_id, samples.load_revision("cv_segmentation", 1))


def test_without_human_gate_workflow_runs_autonomously(settings):
    settings.require_human_approval = False
    svc = JournalService(settings, llm=None)
    ms_id = svc.submit(samples.load_submission("cv_segmentation"))
    assert svc.pending_action(ms_id)["type"] == "revision_request"  # no editor interrupt
    s = svc.submit_revision(ms_id, samples.load_revision("cv_segmentation", 1))
    assert s["status"] == "published"


def test_state_survives_process_restart_with_sqlite_checkpointer(settings):
    settings.checkpointer = "sqlite"
    a = JournalService(settings, llm=None)
    ms_id = a.submit(samples.load_submission("cv_segmentation"))
    approve(a, ms_id)
    del a
    b = JournalService(settings, llm=None)  # new graph, new connections: simulates a pod restart
    assert b.pending_action(ms_id)["type"] == "revision_request"
    b.submit_revision(ms_id, samples.load_revision("cv_segmentation", 1))
    assert approve(b, ms_id, "accept")["status"] == "published"


def test_tools_over_mcp_transport(settings):
    from journal_ai.mcp_server import build_server
    from journal_ai.tools import MCPTools

    tools = MCPTools(build_server(settings))  # in-process MCP server over the real protocol
    svc = JournalService(settings, llm=None, tools=tools)
    s = run_lifecycle(svc, "cv_segmentation")
    assert s["status"] == "published"
    assert {"get_policy", "journal_scope", "check_duplicate", "search_literature", "register_doi",
            "validate_pdf", "log_event"} <= set(tools.calls)
    assert svc.audit_trail(s["manuscript_id"])  # audit events were written through MCP


def test_many_manuscripts_are_isolated(svc):
    ids = [svc.submit(samples.load_submission(n)) for n in ("cv_segmentation", "ml_optimizer", "clinical_trial")]
    assert len(set(ids)) == 3
    assert [svc.summary(i)["domain"] for i in ids] == ["computer_vision", "machine_learning", "clinical"]
    approve(svc, ids[0])
    assert svc.pending_action(ids[1])["type"] == "editor_approval"  # others untouched
