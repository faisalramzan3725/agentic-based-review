"""Agent 1 (Editorial Manager) and Agent 2 (Editorial Decision)."""

from __future__ import annotations

import random
import re
from collections import Counter
from itertools import combinations

from pydantic import BaseModel

from ..documents import attach_sections, render_for_prompt
from ..schemas import (
    RECOMMENDATION_ORDER,
    AggregatedReview,
    ConcernCluster,
    ConcernResolution,
    EditorialDecision,
    Manuscript,
    ManuscriptSubmission,
    Recommendation,
    ResolutionStatus,
    ReviewConflict,
    ReviewerAssignment,
    ReviewReport,
    ScreeningCheck,
    ScreeningResult,
    Severity,
)
from ..tools.text import jaccard, tokens
from .base import AgentContext, count_references, has_any
from .reviewers import REVIEWERS

_SEVERITY_ORDER = {Severity.MINOR: 0, Severity.MAJOR: 1, Severity.CRITICAL: 2}


# =========================================================================== #
# Agent 1 - Editorial Manager
# =========================================================================== #


class EditorialManager:
    agent_id = "editorial_manager"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    # ------------------------------------------------------------- intake
    def intake(self, submission: ManuscriptSubmission, manuscript_id: str) -> Manuscript:
        ms = Manuscript(**submission.model_dump(), manuscript_id=manuscript_id, version=1)
        attach_sections(ms)
        if not ms.abstract:
            sec = ms.section("abstract")
            ms.abstract = sec.text if sec else ""
        ms.domain = self.classify_domain(ms)[0]
        return ms

    def classify_domain(self, ms: Manuscript) -> tuple[str, dict[str, float]]:
        scope = self.ctx.tools.journal_scope()
        toks = Counter(tokens(f"{ms.title} {ms.title} {' '.join(ms.keywords)} {' '.join(ms.keywords)} {ms.abstract} {ms.body[:6000]}"))
        total = sum(toks.values()) or 1
        scores = {d: round(sum(toks[w] for w in words) / total, 4) for d, words in scope.items()}
        # clinical / bio vocabularies are more specific than generic ML words; weight them up
        for d in ("clinical", "bioinformatics"):
            if d in scores:
                scores[d] = round(scores[d] * 1.5, 4)
        best = max(scores, key=scores.get) if scores else "general"
        return (best if scores.get(best, 0) >= 0.01 else "general"), scores

    # ---------------------------------------------------------- screening
    def screen(self, ms: Manuscript) -> ScreeningResult:
        policy = self.ctx.tools.get_policy(ms.article_type, ms.domain)
        checks: list[ScreeningCheck] = []
        words = len(ms.body.split())

        checks.append(ScreeningCheck(name="file_validity", passed=words >= 300, detail=f"{words} words in body"))

        headings = " | ".join(s.heading.lower() for s in ms.sections)
        missing = [s for s in policy["required_sections"] if s not in headings]
        checks.append(
            ScreeningCheck(
                name="required_sections",
                passed=len(missing) <= 2,
                detail=f"missing: {', '.join(missing) or 'none'}",
            )
        )

        ref_sec = ms.section("reference")
        n_refs = count_references(ref_sec.text) if ref_sec else 0
        checks.append(ScreeningCheck(name="reference_structure", passed=n_refs > 0, detail=f"{n_refs} references"))

        meta_issues = []
        if not ms.title.strip():
            meta_issues.append("title")
        if not ms.authors:
            meta_issues.append("authors")
        elif not any(a.corresponding and a.email for a in ms.authors):
            meta_issues.append("corresponding author e-mail")
        if not ms.abstract.strip():
            meta_issues.append("abstract")
        checks.append(
            ScreeningCheck(name="metadata", passed=not meta_issues, detail=f"missing: {', '.join(meta_issues) or 'none'}")
        )

        eng = _english_ratio(ms.body)
        checks.append(ScreeningCheck(name="language", passed=eng >= 0.08, detail=f"function-word ratio {eng:.2f}"))

        dups = self.ctx.tools.check_duplicate(ms.manuscript_id, ms.full_text, threshold=0.9)
        checks.append(
            ScreeningCheck(
                name="duplicate_submission",
                passed=not dups,
                detail="; ".join(f"{d['manuscript_id']} ({d['similarity']})" for d in dups) or "no duplicates",
            )
        )

        domain, scores = self.classify_domain(ms)
        in_scope = max(scores.values(), default=0) >= 0.01
        checks.append(ScreeningCheck(name="scope", passed=in_scope, detail=f"domain={domain} scores={scores}"))

        has_decl = has_any(ms.body, r"conflicts? of interest|competing interest|data availability|ethic")
        checks.append(
            ScreeningCheck(
                name="ethical_declarations",
                passed=has_decl,
                blocking=False,
                detail="declarations present" if has_decl else "no declarations section (flagged for compliance review)",
            )
        )

        passed = all(c.passed for c in checks if c.blocking)
        return ScreeningResult(passed=passed, checks=checks, domain=domain, domain_scores=scores)

    # --------------------------------------------------------- assignment
    def assign_reviewers(self, ms: Manuscript) -> list[ReviewerAssignment]:
        """Capability-based assignment with seeded stochastic tie-breaking.

        1. required capabilities come from the domain policy;
        2. greedy set cover picks reviewers that cover them;
        3. the pool is topped up to ``min_reviewers`` by score, ties broken at random.
        """
        s = self.ctx.settings
        policy = self.ctx.tools.get_policy(ms.article_type, ms.domain)
        required = set(policy.get("reviewer_capabilities", ["structure", "scientific_quality", "language"]))
        rng = random.Random(f"{s.assignment_seed}:{ms.manuscript_id}" if s.assignment_seed is not None else None)

        def score(rid: str) -> float:
            return float(len(required & set(REVIEWERS[rid].capabilities)))

        chosen: list[str] = []
        uncovered = set(required)
        while uncovered and len(chosen) < s.max_reviewers:
            cands = [r for r in REVIEWERS if r not in chosen and set(REVIEWERS[r].capabilities) & uncovered]
            if not cands:
                break
            best = max(len(set(REVIEWERS[r].capabilities) & uncovered) for r in cands)
            pick = rng.choice(sorted(r for r in cands if len(set(REVIEWERS[r].capabilities) & uncovered) == best))
            chosen.append(pick)
            uncovered -= set(REVIEWERS[pick].capabilities)

        rest = sorted((r for r in REVIEWERS if r not in chosen), key=lambda r: (-score(r), rng.random()))
        while len(chosen) < min(s.min_reviewers, s.max_reviewers) and rest:
            chosen.append(rest.pop(0))

        return [
            ReviewerAssignment(
                reviewer_id=r,
                score=score(r),
                reason=(
                    f"covers {sorted(required & set(REVIEWERS[r].capabilities))}"
                    if required & set(REVIEWERS[r].capabilities)
                    else "added to reach the minimum reviewer count"
                ),
            )
            for r in chosen
        ]

    # --------------------------------------------------------- aggregation
    def aggregate(self, ms: Manuscript, reports: list[ReviewReport], round_: int) -> AggregatedReview:
        recs = {r.reviewer_id: r.recommendation for r in reports}
        ords = [RECOMMENDATION_ORDER[v] for v in recs.values()]
        pairs = list(combinations(ords, 2))
        agreement = 1.0 if not pairs else sum(1 - abs(a - b) / 3 for a, b in pairs) / len(pairs)
        worst = max(recs.values(), key=lambda r: RECOMMENDATION_ORDER[r]) if recs else Recommendation.ACCEPT
        counts = Counter(recs.values())
        top = max(counts.values()) if counts else 0
        majority = max(
            (r for r, c in counts.items() if c == top), key=lambda r: RECOMMENDATION_ORDER[r], default=Recommendation.ACCEPT
        )

        clusters: list[ConcernCluster] = []
        texts: list[str] = []
        for rep in reports:
            for c in rep.concerns:
                text = f"{c.category} {c.description} {c.required_change}"
                match = next(
                    (i for i, t in enumerate(texts) if jaccard(text, t) >= 0.25 and rep.reviewer_id not in clusters[i].reviewers),
                    None,
                )
                if match is None:
                    clusters.append(
                        ConcernCluster(
                            cluster_id=f"C{len(clusters) + 1:02d}",
                            representative=c.description,
                            category=c.category,
                            severity=c.severity,
                            concern_ids=[c.concern_id],
                            reviewers=[rep.reviewer_id],
                        )
                    )
                    texts.append(text)
                else:
                    cl = clusters[match]
                    cl.concern_ids.append(c.concern_id)
                    cl.reviewers.append(rep.reviewer_id)
                    if _SEVERITY_ORDER[c.severity] > _SEVERITY_ORDER[cl.severity]:
                        cl.severity = c.severity

        conflicts: list[ReviewConflict] = []
        if ords and max(ords) - min(ords) >= 2:
            conflicts.append(
                ReviewConflict(
                    kind="recommendation_spread",
                    detail=f"recommendations range from {min(recs.values(), key=lambda r: RECOMMENDATION_ORDER[r]).value} "
                    f"to {worst.value}",
                    reviewers=list(recs),
                )
            )
        by_id = {c.concern_id: c for rep in reports for c in rep.concerns}
        for cl in clusters:
            sevs = {by_id[cid].severity for cid in cl.concern_ids}
            if len(sevs) > 1:
                conflicts.append(
                    ReviewConflict(
                        kind="severity_disagreement",
                        detail=f"{cl.cluster_id} rated {sorted(s.value for s in sevs)}",
                        reviewers=cl.reviewers,
                    )
                )

        all_c = [c for rep in reports for c in rep.concerns]
        return AggregatedReview(
            manuscript_id=ms.manuscript_id,
            round=round_,
            recommendations=recs,
            worst_recommendation=worst,
            majority_recommendation=majority,
            agreement=round(agreement, 3),
            clusters=clusters,
            common_concerns=[cl.cluster_id for cl in clusters if cl.is_common],
            unique_concerns=[cl.cluster_id for cl in clusters if not cl.is_common],
            conflicts=conflicts,
            critical_count=sum(c.severity == Severity.CRITICAL for c in all_c),
            major_count=sum(c.severity == Severity.MAJOR for c in all_c),
            minor_count=sum(c.severity == Severity.MINOR for c in all_c),
        )


def _english_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    func = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "we", "with", "on", "are", "as", "by", "this"}
    return sum(w in func for w in words) / len(words)


# =========================================================================== #
# Agent 2 - Editorial Decision
# =========================================================================== #


class LLMLetter(BaseModel):
    decision_letter: str
    additional_rationale: list[str]


class EditorialDecisionAgent:
    """Rules decide; the LLM (if present) only drafts the letter.

    Keeping the recommendation rule-based makes every decision auditable and
    stops an LLM from accepting a paper with an unresolved critical concern.
    """

    agent_id = "editorial_decision"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def decide(
        self,
        ms: Manuscript,
        round_: int,
        aggregated: AggregatedReview,
        reports: list[ReviewReport],
        resolutions: list[ConcernResolution] | None = None,
    ) -> EditorialDecision:
        max_rounds = self.ctx.settings.max_revision_rounds
        rationale: list[str] = []
        unresolved_crit: list[str] = []
        unresolved_major: list[str] = []

        if resolutions is None:  # first-round decision from fresh reviews
            crit = [c.concern_id for r in reports for c in r.concerns if c.severity == Severity.CRITICAL]
            major = [c.concern_id for r in reports for c in r.concerns if c.severity == Severity.MAJOR]
            n_reject = sum(r.recommendation == Recommendation.REJECT for r in reports)
            unresolved_crit, unresolved_major = crit, major
            rationale.append(
                f"Round {round_}: {len(reports)} reviewers, agreement {aggregated.agreement:.2f}; "
                f"{len(crit)} critical, {len(major)} major, {aggregated.minor_count} minor concerns."
            )
            if n_reject > len(reports) / 2:
                rec = Recommendation.REJECT
                rationale.append(f"A majority of reviewers ({n_reject}/{len(reports)}) recommend rejection.")
            elif len(set(crit)) >= 3:
                rec = Recommendation.REJECT
                rationale.append("Three or more distinct critical concerns: the work is not publishable in its current scope.")
            elif crit or len(major) >= 2:
                rec = Recommendation.MAJOR_REVISION
                rationale.append("Critical or multiple major concerns require substantial revision and re-review.")
            elif major or aggregated.minor_count:
                rec = Recommendation.MINOR_REVISION
                rationale.append("Only minor issues (at most one major) remain.")
            else:
                rec = Recommendation.ACCEPT
                rationale.append("No concerns were raised by any reviewer.")
        else:  # decision after a revision, from the concern-resolution matrix
            st = Counter(r.status for r in resolutions)
            unresolved_crit = [
                r.concern_id for r in resolutions if r.severity == Severity.CRITICAL and r.status != ResolutionStatus.ADDRESSED
            ]
            unresolved_major = [
                r.concern_id for r in resolutions if r.severity == Severity.MAJOR and r.status != ResolutionStatus.ADDRESSED
            ]
            unresolved_minor = [
                r.concern_id for r in resolutions if r.severity == Severity.MINOR and r.status == ResolutionStatus.NOT_ADDRESSED
            ]
            rationale.append(
                f"Round {round_}: {st[ResolutionStatus.ADDRESSED]} addressed, "
                f"{st[ResolutionStatus.PARTIALLY_ADDRESSED]} partially addressed, "
                f"{st[ResolutionStatus.NOT_ADDRESSED]} not addressed of {len(resolutions)} tracked concerns."
            )
            exhausted = round_ > max_rounds
            if unresolved_crit:
                rec = Recommendation.REJECT if exhausted else Recommendation.MAJOR_REVISION
                rationale.append(
                    f"Unresolved critical concern(s) {unresolved_crit}"
                    + ("; revision rounds exhausted." if exhausted else "; further revision required.")
                )
            elif unresolved_major:
                rec = Recommendation.REJECT if exhausted else Recommendation.MAJOR_REVISION
                rationale.append(
                    f"Unresolved major concern(s) {unresolved_major}"
                    + ("; revision rounds exhausted." if exhausted else "; further revision required.")
                )
            elif unresolved_minor and not exhausted:
                rec = Recommendation.MINOR_REVISION
                rationale.append(f"All mandatory concerns resolved; minor items {unresolved_minor} remain.")
            else:
                rec = Recommendation.ACCEPT
                rationale.append("All mandatory (critical/major) concerns resolved; required reviewers satisfied.")

        decision = EditorialDecision(
            manuscript_id=ms.manuscript_id,
            round=round_,
            recommendation=rec,
            rationale=rationale,
            unresolved_critical=unresolved_crit,
            unresolved_major=unresolved_major,
            requires_human_approval=self.ctx.settings.require_human_approval,
        )
        decision.decision_letter = self._letter(ms, decision, reports, resolutions)
        return decision

    def _letter(self, ms, decision, reports, resolutions) -> str:
        if self.ctx.llm:
            items = "\n".join(
                f"- {c.concern_id} ({c.severity.value}, {r.reviewer_id}): {c.description}" for r in reports for c in r.concerns
            )
            res = "\n".join(f"- {x.concern_id}: {x.status.value}. {x.remaining_issue}" for x in resolutions or [])
            out = self.ctx.llm.structured(
                context=render_for_prompt(ms),
                instruction=(
                    f"Draft the editor's decision letter to the authors. The decision is FIXED: "
                    f"{decision.recommendation.value}. Rationale: {decision.rationale}.\n"
                    f"Reviewer concerns:\n{items}\nConcern resolution status:\n{res or '(first round)'}\n"
                    "Be courteous and specific; list what must change. Do not alter the decision."
                ),
                schema=LLMLetter,
                effort="medium",
            )
            decision.rationale.extend(out.additional_rationale)
            return out.decision_letter
        lines = [
            f"Dear {next((a.name for a in ms.authors if a.corresponding), 'Author')},",
            "",
            f"Manuscript {ms.manuscript_id}, \"{ms.title}\" (version {ms.version}).",
            f"Editorial recommendation: {decision.recommendation.value.replace('_', ' ').upper()}.",
            "",
            *decision.rationale,
        ]
        if resolutions:
            open_items = [r for r in resolutions if r.status != ResolutionStatus.ADDRESSED]
            if open_items:
                lines += ["", "Outstanding items:"] + [
                    f"- {r.concern_id} [{r.status.value}]: {r.remaining_issue or r.original_concern}" for r in open_items
                ]
        elif decision.recommendation != Recommendation.ACCEPT:
            lines += ["", "Please address the following reviewer concerns and reply point by point, citing each concern id:"]
            for r in reports:
                for c in r.concerns:
                    lines.append(f"- {c.concern_id} [{c.severity.value}] {c.description} Required: {c.required_change}")
        lines += ["", "Sincerely,", "The Editorial Office"]
        return "\n".join(lines)
