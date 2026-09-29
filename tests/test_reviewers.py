import pytest

from journal_ai import samples
from journal_ai.agents.reviewers import REVIEWERS, _response_for, recommend
from journal_ai.documents import attach_sections
from journal_ai.schemas import Concern, Recommendation, ResolutionStatus, Severity


def ids(report):
    return [c.concern_id for c in report.concerns]


def test_pool_has_six_distinct_specialists():
    assert len(REVIEWERS) == 6
    prefixes = {s.prefix for s in REVIEWERS.values()}
    assert prefixes == {"STR", "METH", "NOV", "SCI", "LANG", "COMP"}
    rule_ids = [r.rule_id for s in REVIEWERS.values() for r in s.rules]
    assert len(rule_ids) == len(set(rule_ids))


def test_weak_ml_paper_methodology_and_quality_findings(team, intake):
    ms = intake("ml_optimizer")
    meth = team.reviewers["reviewer_methodology"].review(ms)
    sci = team.reviewers["reviewer_scientific_quality"].review(ms)
    assert ids(meth) == ["METH-002", "METH-003", "METH-004", "METH-005"]
    assert ids(sci) == ["SCI-001", "SCI-003", "SCI-004", "SCI-005"]
    sci001 = sci.concerns[0]
    assert sci001.severity == Severity.CRITICAL and sci001.evidence
    assert meth.recommendation == Recommendation.MAJOR_REVISION
    sci004 = next(c for c in sci.concerns if c.concern_id == "SCI-004")
    assert sorted(sci004.items) == ["12%", "30%"]


def test_strong_cv_paper_is_mostly_clean(team, intake):
    ms = intake("cv_segmentation")
    reports = {rid: a.review(ms) for rid, a in team.reviewers.items()}
    assert ids(reports["reviewer_methodology"]) == []
    assert ids(reports["reviewer_scientific_quality"]) == []
    assert ids(reports["reviewer_novelty"]) == []  # closest prior work (L006) is cited
    assert ids(reports["reviewer_language"]) == ["LANG-003"]
    assert reports["reviewer_language"].concerns[0].items == ["ASPP"]
    assert reports["reviewer_methodology"].recommendation == Recommendation.ACCEPT


def test_clinical_paper_compliance_findings(team, intake):
    rep = team.reviewers["reviewer_compliance"].review(intake("clinical_trial"))
    assert ids(rep) == ["COMP-002", "COMP-003", "COMP-004"]
    assert rep.recommendation == Recommendation.REJECT  # two critical ethics issues


def test_novelty_flags_uncited_close_prior_work(team, intake):
    ms = intake("cv_segmentation")
    ms.body = ms.body.replace("[6] Chen L. et al. Lightweight Attention Modules for Efficient Medical Image Segmentation. 2020.", "")
    attach_sections(ms)
    rep = team.reviewers["reviewer_novelty"].review(ms)
    assert "NOV-003" in ids(rep)
    assert "Lightweight Attention Modules" in rep.concerns[0].description


@pytest.mark.parametrize(
    "sevs,expected",
    [
        ([], Recommendation.ACCEPT),
        (["minor"], Recommendation.MINOR_REVISION),
        (["major"], Recommendation.MINOR_REVISION),
        (["major", "major"], Recommendation.MAJOR_REVISION),
        (["critical"], Recommendation.MAJOR_REVISION),
        (["critical", "critical"], Recommendation.REJECT),
    ],
)
def test_recommendation_policy(sevs, expected):
    cs = [Concern(concern_id=f"X-{i}", category="c", severity=Severity(s), description="d") for i, s in enumerate(sevs)]
    assert recommend(cs) == expected


def _revise(team, name, n):
    orig = team.editorial_manager.intake(samples.load_submission(name), "MS-R")
    rev = samples.load_revision(name, n)
    new = orig.model_copy(update={"version": 2, "body": rev.body, "abstract": rev.abstract, "sections": []})
    attach_sections(new)
    return orig, new, rev.response_to_reviewers


def test_concern_resolution_addressed_partial_and_not(team):
    orig, new, letter = _revise(team, "ml_optimizer", 1)
    agent = team.reviewers["reviewer_scientific_quality"]
    concerns = agent.review(orig).concerns
    res = {r.concern_id: r for r in agent.verify(orig, new, concerns, letter)}
    assert res["SCI-001"].status == ResolutionStatus.ADDRESSED
    assert res["SCI-004"].status == ResolutionStatus.PARTIALLY_ADDRESSED
    assert "30%" in res["SCI-004"].remaining_issue
    assert res["SCI-003"].status == ResolutionStatus.NOT_ADDRESSED
    assert res["SCI-003"].revision_evidence == ""
    assert "softened" in res["SCI-001"].author_response  # picked from the response letter by concern id
    assert res["SCI-001"].revision_evidence.startswith("Revised section:")


def test_itemised_concern_resolved_even_if_new_item_appears(team):
    orig, new, letter = _revise(team, "cv_segmentation", 1)
    agent = team.reviewers["reviewer_language"]
    concerns = agent.review(orig).concerns
    assert agent.review(new).concerns[0].items == ["ERC"]  # revision introduced a new acronym
    (res,) = agent.verify(orig, new, concerns, letter)
    assert res.status == ResolutionStatus.ADDRESSED  # ...but the original one (ASPP) is fixed


def test_response_letter_matching():
    c = Concern(concern_id="METH-003", category="statistics", severity=Severity.MAJOR,
                description="The statistical analysis is insufficient: no variance reported.")
    letter = "METH-002: added baselines.\n\nMETH-003: we now report standard deviations."
    assert _response_for(c, letter) == "METH-003: we now report standard deviations."
    assert _response_for(c, "") == ""
    assert "variance" in _response_for(c.model_copy(update={"concern_id": "Z-1"}), "Intro.\n\nWe added statistical variance analysis.")
