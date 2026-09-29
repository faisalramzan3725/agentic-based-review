"""Claude backend tests. A fake Anthropic client stands in for the API so the
request shape and every LLM code path (review, verification, decision letter)
are exercised offline; a live test runs only when explicitly enabled."""

import os
import re
from types import SimpleNamespace

import pytest

from journal_ai import samples
from journal_ai.agents import REVIEWERS
from journal_ai.agents.editorial import LLMLetter
from journal_ai.agents.reviewers import LLMResolutionBatch, LLMResolutionItem
from journal_ai.config import Settings
from journal_ai.llm import ClaudeLLM, LLMRefusalError, build_llm
from journal_ai.schemas import Concern, Evidence, HumanDecision, LLMReviewOutput, Recommendation, Severity
from journal_ai.service import JournalService


class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    def parse(self, **kw):
        self.calls.append(kw)
        schema, instruction = kw["output_format"], kw["messages"][0]["content"][1]["text"]
        if schema is LLMReviewOutput:
            prefix = re.search(r"of the form (\w+)-001", instruction).group(1)
            parsed = LLMReviewOutput(
                recommendation=Recommendation.MINOR_REVISION,
                confidence=0.8,
                summary=f"{prefix} review",
                strengths=["clear"],
                concerns=[Concern(concern_id="tmp", category="clarity", severity=Severity.MAJOR,
                                  description=f"{prefix} issue", required_change="fix it",
                                  evidence=[Evidence(section="Methods", issue="vague")])],
            )
        elif schema is LLMResolutionBatch:
            ids = re.findall(r"- (\w+-\d{3}) \[", instruction)
            parsed = LLMResolutionBatch(resolutions=[
                LLMResolutionItem(concern_id=i, status="addressed", author_response="done",
                                  revision_evidence="Section 3", remaining_issue="") for i in ids])
        elif schema is LLMLetter:
            parsed = LLMLetter(decision_letter="Dear author, ...", additional_rationale=["letter drafted by LLM"])
        else:
            raise AssertionError(schema)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=800, cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=parsed, usage=usage, stop_reason=self.stop_reason, stop_details=None)


def fake_client(stop_reason="end_turn"):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(stop_reason)))


def test_request_shape_follows_current_api(settings):
    client = fake_client()
    llm = ClaudeLLM(settings, client=client)
    out = llm.structured(context="MANUSCRIPT", instruction="Review. ids of the form STR-001", schema=LLMReviewOutput)
    kw = client.beta.messages.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_format"] is LLMReviewOutput
    assert kw["output_config"] == {"effort": "high"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw and "budget_tokens" not in str(kw) and "temperature" not in kw
    first = kw["messages"][0]["content"][0]
    assert first == {"type": "text", "text": "MANUSCRIPT", "cache_control": {"type": "ephemeral"}}  # shared, cached prefix
    assert out.summary == "STR review"
    assert llm.usage.snapshot() == {"llm_calls": 1, "input_tokens": 1000, "output_tokens": 200,
                                    "cache_read_tokens": 800, "cache_write_tokens": 0}


def test_refusal_is_raised(settings):
    llm = ClaudeLLM(settings, client=fake_client("refusal"))
    with pytest.raises(LLMRefusalError):
        llm.structured(context="x", instruction="ids of the form X-001", schema=LLMReviewOutput)


def test_backend_selection(settings, monkeypatch):
    settings.llm_backend = "heuristic"
    assert build_llm(settings) is None
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    settings.llm_backend = "auto"
    assert build_llm(settings) is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(build_llm(settings), ClaudeLLM)


def test_full_workflow_on_llm_backend(settings):
    client = fake_client()
    llm = ClaudeLLM(settings, client=client)
    svc = JournalService(settings, llm=llm)
    ms_id = svc.submit(samples.load_submission("cv_segmentation"))
    s = svc.summary(ms_id)
    n_reviewers = len(s["assignments"])
    # LLM concern ids are re-stamped with the reviewer's prefix so they stay unique and trackable
    for r in s["reviews"]:
        assert [c["concern_id"] for c in r["concerns"]] == [f"{REVIEWERS[r['reviewer_id']].prefix}-001"]
        assert r["concerns"][0]["reviewer_id"] == r["reviewer_id"]
    # the decision is rule-based even on the LLM backend: >=2 major concerns -> major revision
    assert s["decision"]["recommendation"] == "major_revision"
    svc.editor_decision(ms_id, HumanDecision(decision=s["pending_action"]["recommendation"]))
    svc.submit_revision(ms_id, samples.load_revision("cv_segmentation", 1))
    s = svc.summary(ms_id)
    assert {r["status"] for r in s["resolutions"]} == {"addressed"}
    assert "letter drafted by LLM" in s["decision"]["rationale"]
    svc.editor_decision(ms_id, HumanDecision(decision=Recommendation.ACCEPT))
    s = svc.summary(ms_id)
    assert s["status"] == "published"
    # reviews + verifications + one letter per decision
    assert s["metrics"]["llm_calls"] == len(client.beta.messages.calls) == n_reviewers * 2 + 2
    novelty_calls = [c for c in client.beta.messages.calls if "RETRIEVED RELATED LITERATURE" in c["messages"][0]["content"][1]["text"]]
    assert len(novelty_calls) == (1 if "reviewer_novelty" in [a["reviewer_id"] for a in s["assignments"]] else 0)


@pytest.mark.live
@pytest.mark.skipif(not (os.getenv("JOURNAL_LIVE_TESTS") and os.getenv("ANTHROPIC_API_KEY")), reason="live Claude test disabled")
def test_live_claude_review():  # pragma: no cover - costs money, run manually
    s = Settings()
    s.llm_backend = "claude"
    llm = ClaudeLLM(s)
    from journal_ai.agents import AgentContext, AgentTeam
    from journal_ai.tools import LocalTools, ManuscriptStore

    team = AgentTeam.build(AgentContext(settings=s, tools=LocalTools(ManuscriptStore()), llm=llm))
    ms = team.editorial_manager.intake(samples.load_submission("ml_optimizer"), "MS-LIVE")
    rep = team.reviewers["reviewer_methodology"].review(ms)
    assert rep.concerns and all(c.concern_id.startswith("METH-") for c in rep.concerns)
