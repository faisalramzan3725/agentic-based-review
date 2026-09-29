from journal_ai.agents.editorial import EditorialDecisionAgent
from journal_ai.agents.reviewers import REVIEWERS
from journal_ai.schemas import (
    Concern,
    ConcernResolution,
    Recommendation,
    ResolutionStatus,
    ReviewReport,
    Severity,
)


def test_domain_classification(intake):
    assert intake("cv_segmentation").domain == "computer_vision"
    assert intake("ml_optimizer").domain == "machine_learning"
    assert intake("clinical_trial").domain == "clinical"


def test_screening_passes_good_and_fails_incomplete(team, intake):
    ok = team.editorial_manager.screen(intake("cv_segmentation"))
    assert ok.passed and all(c.passed for c in ok.checks)
    bad = team.editorial_manager.screen(intake("incomplete"))
    failed = {c.name for c in bad.checks if not c.passed}
    assert not bad.passed
    assert {"file_validity", "required_sections", "reference_structure", "metadata"} <= failed


def test_capability_based_assignment_covers_domain_needs(team, intake, ctx):
    for name in ("cv_segmentation", "ml_optimizer", "clinical_trial"):
        ms = intake(name)
        required = set(ctx.tools.get_policy(ms.article_type, ms.domain)["reviewer_capabilities"])
        chosen = [a.reviewer_id for a in team.editorial_manager.assign_reviewers(ms)]
        covered = {c for r in chosen for c in REVIEWERS[r].capabilities}
        assert required <= covered, name
        assert 3 <= len(chosen) <= 6 and len(set(chosen)) == len(chosen)
    clinical = [a.reviewer_id for a in team.editorial_manager.assign_reviewers(intake("clinical_trial"))]
    assert "reviewer_compliance" in clinical


def test_assignment_is_reproducible_with_seed_and_respects_bounds(team, intake, settings):
    ms = intake("cv_segmentation")
    a1 = [a.reviewer_id for a in team.editorial_manager.assign_reviewers(ms)]
    a2 = [a.reviewer_id for a in team.editorial_manager.assign_reviewers(ms)]
    assert a1 == a2
    settings.max_reviewers = 2
    assert len(team.editorial_manager.assign_reviewers(ms)) == 2
    settings.max_reviewers, settings.min_reviewers = 6, 6
    assert len(team.editorial_manager.assign_reviewers(ms)) == 6


def _report(rid, rec, concerns):
    return ReviewReport(manuscript_id="M", reviewer_id=rid, recommendation=rec, confidence=0.8, summary="s", concerns=concerns)


def _c(cid, rid, sev, desc, cat="stats"):
    return Concern(concern_id=cid, reviewer_id=rid, category=cat, severity=Severity(sev), description=desc)


def test_aggregation_common_unique_and_conflicts(team, intake):
    ms = intake("ml_optimizer")
    reps = [
        _report("r1", Recommendation.ACCEPT, [_c("A-1", "r1", "minor", "no statistical significance tests or variance reported")]),
        _report("r2", Recommendation.REJECT, [
            _c("B-1", "r2", "critical", "statistical significance tests and variance are not reported"),
            _c("B-2", "r2", "major", "ethics approval missing for human participants", "ethics"),
        ]),
    ]
    agg = team.editorial_manager.aggregate(ms, reps, 1)
    assert len(agg.clusters) == 2
    common = next(c for c in agg.clusters if c.cluster_id in agg.common_concerns)
    assert set(common.concern_ids) == {"A-1", "B-1"} and common.severity == Severity.CRITICAL
    assert len(agg.unique_concerns) == 1
    kinds = {c.kind for c in agg.conflicts}
    assert kinds == {"recommendation_spread", "severity_disagreement"}
    assert agg.worst_recommendation == Recommendation.REJECT
    assert agg.agreement == 0.0
    assert (agg.critical_count, agg.major_count, agg.minor_count) == (1, 1, 1)


def _decide(ctx, reports, resolutions=None, round_=1):
    agent = EditorialDecisionAgent(ctx)
    from journal_ai.agents.editorial import EditorialManager

    agg = EditorialManager(ctx).aggregate(_ms(), reports, 1)
    return agent.decide(_ms(), round_, agg, reports, resolutions)


def _ms():
    from journal_ai.schemas import Author, Manuscript

    return Manuscript(manuscript_id="M", title="T", body="b", authors=[Author(name="A B", email="a@b", corresponding=True)])


def test_first_round_decisions(ctx):
    clean = [_report("r1", Recommendation.ACCEPT, []), _report("r2", Recommendation.ACCEPT, [])]
    assert _decide(ctx, clean).recommendation == Recommendation.ACCEPT
    minor = [_report("r1", Recommendation.MINOR_REVISION, [_c("L-1", "r1", "minor", "typo")])]
    assert _decide(ctx, minor).recommendation == Recommendation.MINOR_REVISION
    crit = [_report("r1", Recommendation.MAJOR_REVISION, [_c("S-1", "r1", "critical", "unsupported")])]
    d = _decide(ctx, crit)
    assert d.recommendation == Recommendation.MAJOR_REVISION and d.unresolved_critical == ["S-1"]
    assert "S-1" in d.decision_letter
    rej = [_report(f"r{i}", Recommendation.REJECT, []) for i in range(2)] + [_report("r9", Recommendation.ACCEPT, [])]
    assert _decide(ctx, rej).recommendation == Recommendation.REJECT
    three = [_report("r1", Recommendation.MAJOR_REVISION, [_c(f"S-{i}", "r1", "critical", "x") for i in range(3)])]
    assert _decide(ctx, three).recommendation == Recommendation.REJECT


def _res(cid, sev, status):
    return ConcernResolution(concern_id=cid, reviewer_id="r1", original_concern="c", severity=Severity(sev), status=ResolutionStatus(status))


def test_post_revision_decisions_and_exhaustion(ctx):
    reps = [_report("r1", Recommendation.MAJOR_REVISION, [])]
    ok = [_res("A", "critical", "addressed"), _res("B", "minor", "partially_addressed")]
    assert _decide(ctx, reps, ok, 2).recommendation == Recommendation.ACCEPT
    minor_open = [_res("A", "major", "addressed"), _res("B", "minor", "not_addressed")]
    assert _decide(ctx, reps, minor_open, 2).recommendation == Recommendation.MINOR_REVISION
    major_partial = [_res("A", "major", "partially_addressed")]
    assert _decide(ctx, reps, major_partial, 2).recommendation == Recommendation.MAJOR_REVISION
    crit_open = [_res("A", "critical", "not_addressed")]
    d = _decide(ctx, reps, crit_open, 3)  # max_revision_rounds=2 -> round 3 is the last chance
    assert d.recommendation == Recommendation.REJECT and "exhausted" in " ".join(d.rationale)
    # empty resolution list after a revision is a post-revision decision, not a first-round one
    assert _decide(ctx, reps, [], 2).recommendation == Recommendation.ACCEPT
