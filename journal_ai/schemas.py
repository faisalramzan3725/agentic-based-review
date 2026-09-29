"""Typed data contracts shared by every agent.

Agents never exchange free text with each other: everything that crosses an
agent boundary is one of these Pydantic models, so later stages (aggregation,
concern tracking, editorial decision) reason over structure, not prose.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"


# Ordinal scale used for agreement statistics and "worst-of" aggregation.
RECOMMENDATION_ORDER = {
    Recommendation.ACCEPT: 0,
    Recommendation.MINOR_REVISION: 1,
    Recommendation.MAJOR_REVISION: 2,
    Recommendation.REJECT: 3,
}


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ResolutionStatus(str, Enum):
    ADDRESSED = "addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    NOT_ADDRESSED = "not_addressed"


# --------------------------------------------------------------------------- #
# Manuscript
# --------------------------------------------------------------------------- #


class Author(BaseModel):
    name: str
    email: str | None = None
    affiliation: str | None = None
    orcid: str | None = None
    corresponding: bool = False


class Section(BaseModel):
    heading: str
    level: int = 1
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class ManuscriptSubmission(BaseModel):
    """What an author (or the n8n webhook) sends in."""

    title: str
    abstract: str = ""
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    body: str = Field(description="Full manuscript text (Markdown headings preferred).")
    cover_letter: str = ""
    article_type: str = "research_article"
    figures: list[str] = Field(default_factory=list, description="Figure file names or captions.")
    supplementary: list[str] = Field(default_factory=list)


class Manuscript(ManuscriptSubmission):
    manuscript_id: str
    version: int = 1
    sections: list[Section] = Field(default_factory=list)
    domain: str | None = None

    def section(self, *names: str) -> Section | None:
        """First section whose heading contains any of ``names`` (case-insensitive)."""
        for sec in self.sections:
            h = sec.heading.lower()
            if any(n.lower() in h for n in names):
                return sec
        return None

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}\n\n{self.body}"


# --------------------------------------------------------------------------- #
# Screening / assignment
# --------------------------------------------------------------------------- #


class ScreeningCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    blocking: bool = True


class ScreeningResult(BaseModel):
    passed: bool
    checks: list[ScreeningCheck]
    domain: str
    domain_scores: dict[str, float] = Field(default_factory=dict)


class ReviewerAssignment(BaseModel):
    reviewer_id: str
    reason: str
    score: float


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    section: str
    location: str = ""
    issue: str


class Concern(BaseModel):
    concern_id: str = Field(description="Stable id, e.g. METH-003. Tracked across revisions.")
    reviewer_id: str = ""
    category: str
    severity: Severity
    description: str
    required_change: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    rule_id: str | None = Field(
        default=None,
        description="Id of the deterministic check that raised it (heuristic backend only).",
    )
    magnitude: float | None = Field(
        default=None, description="Size of the problem (e.g. number of missing items); heuristic only."
    )
    items: list[str] = Field(
        default_factory=list, description="Concrete offending items (acronyms, missing sections...); heuristic only."
    )


class ReviewReport(BaseModel):
    manuscript_id: str
    reviewer_id: str
    round: int = 1
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    strengths: list[str] = Field(default_factory=list)
    concerns: list[Concern] = Field(default_factory=list)

    @property
    def major_concerns(self) -> list[Concern]:
        return [c for c in self.concerns if c.severity != Severity.MINOR]

    @property
    def minor_concerns(self) -> list[Concern]:
        return [c for c in self.concerns if c.severity == Severity.MINOR]


class LLMReviewOutput(BaseModel):
    """Schema the LLM fills; the agent stamps ids/round onto it afterwards."""

    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    strengths: list[str]
    concerns: list[Concern]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


class ConcernCluster(BaseModel):
    cluster_id: str
    representative: str
    category: str
    severity: Severity
    concern_ids: list[str]
    reviewers: list[str]

    @property
    def is_common(self) -> bool:
        return len(set(self.reviewers)) > 1


class ReviewConflict(BaseModel):
    kind: Literal["recommendation_spread", "severity_disagreement"]
    detail: str
    reviewers: list[str]


class AggregatedReview(BaseModel):
    manuscript_id: str
    round: int
    recommendations: dict[str, Recommendation]
    worst_recommendation: Recommendation
    majority_recommendation: Recommendation
    agreement: float = Field(description="Mean pairwise ordinal agreement in [0, 1].")
    clusters: list[ConcernCluster]
    common_concerns: list[str]
    unique_concerns: list[str]
    conflicts: list[ReviewConflict]
    critical_count: int
    major_count: int
    minor_count: int


# --------------------------------------------------------------------------- #
# Revision / concern tracking
# --------------------------------------------------------------------------- #


class RevisionSubmission(BaseModel):
    body: str
    abstract: str | None = None
    title: str | None = None
    response_to_reviewers: str = ""


class ConcernResolution(BaseModel):
    concern_id: str
    reviewer_id: str
    round: int = 0
    original_concern: str
    severity: Severity
    author_response: str = ""
    revision_evidence: str = ""
    status: ResolutionStatus
    remaining_issue: str = ""


class LLMResolutionOutput(BaseModel):
    status: ResolutionStatus
    author_response: str
    revision_evidence: str
    remaining_issue: str


# --------------------------------------------------------------------------- #
# Editorial decision
# --------------------------------------------------------------------------- #


class EditorialDecision(BaseModel):
    manuscript_id: str
    round: int
    recommendation: Recommendation
    rationale: list[str]
    unresolved_critical: list[str] = Field(default_factory=list)
    unresolved_major: list[str] = Field(default_factory=list)
    decision_letter: str = ""
    requires_human_approval: bool = True


class HumanDecision(BaseModel):
    decision: Recommendation
    editor: str = "handling-editor"
    notes: str = ""


# --------------------------------------------------------------------------- #
# Publication / production
# --------------------------------------------------------------------------- #


class PublicationRecord(BaseModel):
    manuscript_id: str
    doi: str
    volume: int
    issue: int
    published_date: str
    metadata_valid: bool
    metadata_issues: list[str]
    crossref_xml: str
    jats_xml: str


class ProductionCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ProductionResult(BaseModel):
    manuscript_id: str
    pdf_path: str
    page_count: int
    checks: list[ProductionCheck]
    print_ready: bool
