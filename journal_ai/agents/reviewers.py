"""Agents 5-10: the specialised reviewer pool.

Every reviewer has
  * a capability profile (used by capability-based assignment),
  * an LLM brief (what to focus on when Claude is the backend),
  * a set of deterministic rules (the offline backend, and the mechanism that
    makes concern tracking reproducible: a concern raised by rule R is
    re-checked by re-running R on the revised manuscript).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel

from ..documents import render_for_prompt
from ..schemas import (
    Concern,
    ConcernResolution,
    Evidence,
    LLMReviewOutput,
    Manuscript,
    Recommendation,
    ResolutionStatus,
    ReviewReport,
    Severity,
)
from ..tools.text import jaccard, sentences, tokens
from .base import AgentContext, count_references, has_any, numbers_in

# --------------------------------------------------------------------------- #
# Rule framework
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    description: str
    required_change: str
    evidence: list[Evidence] = field(default_factory=list)
    magnitude: float = 1.0
    # For set-valued findings (missing sections, undefined acronyms...): the concrete
    # items, so a revision that fixes the original items but introduces new ones is
    # judged on the original concern.
    items: list[str] = field(default_factory=list)


@dataclass
class RuleContext:
    policy: dict[str, Any]
    ctx: AgentContext
    domain: str


@dataclass
class Rule:
    rule_id: str
    category: str
    severity: Severity
    check: Callable[[Manuscript, RuleContext], Finding | None]


def _sec_text(ms: Manuscript, *names: str) -> str:
    return "\n".join(s.text for s in ms.sections if any(n in s.heading.lower() for n in names))


def _declarations_text(ms: Manuscript) -> str:
    return ms.body + "\n" + ms.cover_letter


_STATS = (
    r"p\s?[<=>]\s?0?\.\d|p-value|confidence interval|\bci\b|standard deviation|\bstd\b|±|\+/-|"
    r"t-test|anova|wilcoxon|mann-whitney|bootstrap|chi-square|statistically significant|standard error"
)
_EMPIRICAL = r"experiment|evaluat|dataset|accuracy|results show|we measured|participants|patients|trial"
_HUMAN_SUBJECTS = r"\bpatients?\b|participants?\b|volunteers?\b|human subjects?|clinical data|medical records"


# ---------------------------- structural ----------------------------------- #


def _missing_sections(ms: Manuscript, rc: RuleContext) -> Finding | None:
    headings = " | ".join(s.heading.lower() for s in ms.sections)
    missing = [s for s in rc.policy["required_sections"] if s not in headings]
    if not missing:
        return None
    return Finding(
        f"Required sections are missing: {', '.join(missing)}.",
        f"Add the missing sections ({', '.join(missing)}) following the journal template.",
        [Evidence(section="Manuscript structure", issue=f"No heading matching '{m}'") for m in missing],
        magnitude=len(missing),
        items=missing,
    )


def _abstract_length(ms: Manuscript, rc: RuleContext) -> Finding | None:
    sec = ms.section("abstract")
    words = len((sec.text if sec else ms.abstract).split())
    lo, hi = rc.policy["abstract_words"]
    if lo <= words <= hi:
        return None
    return Finding(
        f"Abstract has {words} words; the journal requires {lo}-{hi}.",
        f"Revise the abstract to {lo}-{hi} words, stating problem, method, key quantitative result and conclusion.",
        [Evidence(section="Abstract", issue=f"{words} words")],
        magnitude=abs(words - (lo if words < lo else hi)),
    )


def _few_references(ms: Manuscript, rc: RuleContext) -> Finding | None:
    sec = ms.section("reference")
    n = count_references(sec.text) if sec else 0
    need = rc.policy["min_references"]
    if n >= need:
        return None
    return Finding(
        f"Only {n} references are listed; at least {need} are expected for this article type.",
        "Expand the reference list to situate the work within the relevant literature.",
        [Evidence(section="References", issue=f"{n} entries found")],
        magnitude=need - n,
    )


def _thin_core_sections(ms: Manuscript, rc: RuleContext) -> Finding | None:
    thin = [
        s.heading
        for s in ms.sections
        if any(k in s.heading.lower() for k in ("method", "result", "discussion")) and s.word_count < 80
    ]
    if not thin:
        return None
    return Finding(
        f"Core sections are too brief to evaluate: {', '.join(thin)}.",
        "Expand these sections with sufficient detail for independent assessment.",
        [Evidence(section=h, issue="fewer than 80 words") for h in thin],
        magnitude=len(thin),
    )


def _no_objective(ms: Manuscript, rc: RuleContext) -> Finding | None:
    intro = _sec_text(ms, "introduction")
    if not intro or has_any(intro, r"we propose|this (paper|study|work)|our contributions?|we aim|objective|we present|we introduce"):
        return None
    return Finding(
        "The introduction does not state the objective or contributions of the work.",
        "End the introduction with an explicit statement of objectives and contributions.",
        [Evidence(section="Introduction", issue="no objective/contribution statement")],
    )


def _few_keywords(ms: Manuscript, rc: RuleContext) -> Finding | None:
    need = rc.policy.get("min_keywords", 3)
    if len(ms.keywords) >= need:
        return None
    return Finding(
        f"{len(ms.keywords)} keywords supplied; at least {need} are required.",
        f"Provide at least {need} keywords.",
        [Evidence(section="Metadata", issue="keywords")],
        magnitude=need - len(ms.keywords),
    )


# ---------------------------- methodology ---------------------------------- #


def _is_empirical(ms: Manuscript) -> bool:
    return has_any(ms.body, _EMPIRICAL)


def _no_dataset(ms: Manuscript, rc: RuleContext) -> Finding | None:
    methods = _sec_text(ms, "method", "experiment", "material", "setup")
    if not _is_empirical(ms) or has_any(methods, r"dataset|data set|participants|samples|cohort|corpus|benchmark|recruited|collected"):
        return None
    return Finding(
        "The data used in the study (source, size, inclusion criteria, splits) is not described.",
        "Describe the dataset(s): provenance, size, preprocessing and train/validation/test splits.",
        [Evidence(section="Methods", issue="no description of data")],
    )


def _no_baselines(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if not _is_empirical(ms) or rc.domain == "clinical":
        return None
    # "outperforms" alone is a claim, not a comparison, so it does not count here
    if has_any(ms.body, r"baseline|compared? (with|to|against)|comparison with|state[- ]of[- ]the[- ]art|\bsota\b"):
        return None
    return Finding(
        "No comparison against baselines or prior methods is reported.",
        "Compare the proposed approach against established baselines and recent state-of-the-art methods.",
        [Evidence(section="Results", issue="no baseline comparison")],
    )


def _no_statistics(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if not _is_empirical(ms) or has_any(ms.body, _STATS):
        return None
    return Finding(
        "The statistical analysis is insufficient: no variance, confidence intervals or significance tests are reported.",
        "Report mean and standard deviation over repeated runs (or confidence intervals) and apply appropriate significance tests.",
        [Evidence(section="Results", issue="no statistical measures found")],
    )


def _no_reproducibility(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if not _is_empirical(ms) or rc.domain == "clinical":
        return None
    # require concrete settings: merely mentioning "learning rate" (e.g. in an optimizer paper) is not enough
    if has_any(ms.body, r"learning rate (of |was |is |= ?|set to )?\d|batch size|\d+ epochs|hyper-?parameters|implementation details"):
        return None
    return Finding(
        "Implementation details needed for reproduction (hyperparameters, training schedule, hardware) are missing.",
        "Add an implementation-details subsection listing all hyperparameters and the training protocol.",
        [Evidence(section="Methods", issue="no hyperparameters / implementation details")],
    )


def _no_repeated_runs(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if not _is_empirical(ms) or has_any(ms.body, r"seeds?|independent runs|repeated|folds?|cross-validation|runs\b"):
        return None
    return Finding(
        "Results appear to come from a single run; variability across runs/folds is unknown.",
        "Repeat experiments over multiple random seeds or folds and report the spread.",
        [Evidence(section="Results", issue="single-run results")],
    )


# ------------------------------- novelty ----------------------------------- #


def _no_related_work(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if ms.section("related work", "literature review", "background", "prior work"):
        return None
    intro = _sec_text(ms, "introduction")
    if len(re.findall(r"\[\d+\]|\(\w+[^()]*\d{4}\)", intro)) >= 5:
        return None
    return Finding(
        "The manuscript does not review related work, so the contribution cannot be positioned.",
        "Add a related-work section contrasting this work with the closest prior approaches.",
        [Evidence(section="Introduction", issue="no related-work discussion")],
    )


def _no_contribution_claim(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if has_any(ms.body + ms.abstract, r"we propose|our contributions?|novel|we introduce|to our knowledge|first to|this paper presents|we present"):
        return None
    return Finding(
        "The claimed contribution relative to existing work is not stated.",
        "State the specific novel contribution(s) and how they differ from prior work.",
        [Evidence(section="Abstract/Introduction", issue="no contribution claim")],
    )


def _overlap_uncited(ms: Manuscript, rc: RuleContext) -> Finding | None:
    hits = rc.ctx.tools.search_literature(f"{ms.title} {ms.abstract}", k=3)
    refs = (_sec_text(ms, "reference") or "").lower()
    close = [h for h in hits if (h.get("similarity") or 0) >= 0.4 and h["title"].lower()[:40] not in refs]
    if not close:
        return None
    return Finding(
        "Closely related prior work is not cited or discussed: " + "; ".join(f"\"{h['title']}\" ({h.get('year')})" for h in close) + ".",
        "Cite and explicitly differentiate the work from these publications.",
        [Evidence(section="References", location=h.get("doi", ""), issue=f"similarity {h.get('similarity')}") for h in close],
        magnitude=len(close),
        items=[h["title"] for h in close],
    )


def _no_recent_refs(ms: Manuscript, rc: RuleContext) -> Finding | None:
    refs = _sec_text(ms, "reference")
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", refs)]
    if not refs or not years:
        return None
    cutoff = datetime.now(timezone.utc).year - 5
    if any(y >= cutoff for y in years):
        return None
    return Finding(
        f"No references from the last five years (newest: {max(years)}); recent developments may be missing.",
        "Update the literature review with recent work.",
        [Evidence(section="References", issue=f"newest reference {max(years)}")],
    )


# --------------------------- scientific quality ---------------------------- #


def _unsupported_claims(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = _sec_text(ms, "result", "discussion", "conclusion") + "\n" + ms.abstract
    strong = re.findall(r"[^.]*\b(prove[sn]?|always|never fails|guarantees?|significantly (?:outperform|better|improve)s?)\b[^.]*\.", text, re.I)
    if not strong or has_any(ms.body, _STATS):
        return None
    sents = [s.strip() for s in re.findall(r"[^.]*\b(?:prove[sn]?|always|never fails|guarantees?|significantly (?:outperform|better|improve)s?)\b[^.]*\.", text, re.I)]
    return Finding(
        "Strong claims (e.g. 'significantly', 'proves') are made without statistical support.",
        "Either support these claims with statistical evidence or soften the wording.",
        [Evidence(section="Results/Discussion", location=s[:120], issue="unsupported strong claim") for s in sents[:3]],
        magnitude=len(sents),
    )


def _results_not_quantitative(ms: Manuscript, rc: RuleContext) -> Finding | None:
    res = _sec_text(ms, "result")
    if not _is_empirical(ms):
        return None
    n = len(re.findall(r"\d+(?:\.\d+)?", res))
    if n >= 3:
        return None
    return Finding(
        "The results section contains little or no quantitative evidence for the conclusions.",
        "Report the quantitative results (tables/metrics) that support each conclusion.",
        [Evidence(section="Results", issue=f"only {n} numeric values")],
    )


def _no_limitations(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if has_any(ms.body, r"limitation|threats to validity|shortcoming|future work should address|caveat"):
        return None
    return Finding(
        "Limitations of the study are not discussed.",
        "Add a limitations paragraph covering threats to internal and external validity.",
        [Evidence(section="Discussion", issue="no limitations")],
    )


def _abstract_results_mismatch(ms: Manuscript, rc: RuleContext) -> Finding | None:
    abstract = ms.abstract or _sec_text(ms, "abstract")
    body = _sec_text(ms, "result", "discussion", "table", "experiment")
    claimed = {n.replace(" ", "") for n in numbers_in(abstract + " " + _sec_text(ms, "conclusion"))}
    if not claimed or not body:
        return None
    body_nums = {n.replace(" ", "") for n in numbers_in(body)}
    orphan = sorted(n for n in claimed if n not in body_nums)
    if not orphan:
        return None
    return Finding(
        f"Figures quoted in the abstract/conclusion ({', '.join(orphan)}) do not appear in the results.",
        "Make the abstract and conclusions consistent with the reported results.",
        [Evidence(section="Abstract/Conclusion", issue=f"value {n} not found in Results") for n in orphan],
        magnitude=len(orphan),
        items=orphan,
    )


def _no_ablation(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if rc.domain not in {"computer_vision", "nlp", "machine_learning"} or not _is_empirical(ms):
        return None
    if has_any(ms.body, r"ablation|sensitivity analysis|error analysis|component analysis"):
        return None
    return Finding(
        "No ablation or error analysis shows which components drive the improvement.",
        "Add an ablation study isolating each proposed component.",
        [Evidence(section="Results", issue="no ablation")],
    )


# ------------------------------- language ---------------------------------- #

_COMMON_ACRONYMS = {"DNA", "RNA", "USA", "UK", "EU", "AI", "PDF", "GPU", "CPU", "RAM", "ID", "HTTP", "API", "URL", "DOI", "ISBN", "II", "III", "IV", "NCT", "IRB", "COVID", "MRI", "CT", "ORCID", "JSON", "XML", "SQL", "RGB", "OK"}


def _long_sentences(ms: Manuscript, rc: RuleContext) -> Finding | None:
    sents = sentences(ms.body)
    long_ = [s for s in sents if len(s.split()) > 40]
    if not sents or len(long_) / len(sents) < 0.15:
        return None
    return Finding(
        f"{len(long_)} of {len(sents)} sentences exceed 40 words, which hurts readability.",
        "Split long sentences and tighten the prose.",
        [Evidence(section="Throughout", location=s[:100], issue="sentence > 40 words") for s in long_[:3]],
        magnitude=len(long_),
    )


def _repeated_words(ms: Manuscript, rc: RuleContext) -> Finding | None:
    reps = re.findall(r"\b(\w+)\s+\1\b", ms.body, re.I)
    reps = [r for r in reps if not r.isdigit()]
    if not reps:
        return None
    return Finding(
        f"Duplicated words found ({', '.join(sorted(set(r.lower() for r in reps)))}).",
        "Proof-read the manuscript for typographical errors.",
        [Evidence(section="Throughout", issue=f"'{r} {r}'") for r in reps[:3]],
        magnitude=len(reps),
    )


def _undefined_acronyms(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = ms.abstract + "\n" + ms.body
    used = set(re.findall(r"\b([A-Z]{2,6})s?\b", text))
    defined = set(re.findall(r"\(([A-Z]{2,6})s?\)", text))
    undefined = sorted(a for a in used - defined - _COMMON_ACRONYMS if not re.fullmatch(r"[IVX]+", a))
    if not undefined:
        return None
    return Finding(
        f"Acronyms used without definition: {', '.join(undefined)}.",
        "Define every acronym at first use.",
        [Evidence(section="Throughout", issue=a) for a in undefined[:5]],
        magnitude=len(undefined),
        items=undefined,
    )


def _figure_numbering(ms: Manuscript, rc: RuleContext) -> Finding | None:
    nums = sorted({int(n) for n in re.findall(r"\b(?:Figure|Fig\.)\s?(\d+)", ms.body)})
    issues = []
    if nums and nums != list(range(1, max(nums) + 1)):
        issues.append(f"figure numbering is not sequential ({nums})")
    if ms.figures and nums and max(nums) > len(ms.figures):
        issues.append(f"text refers to Figure {max(nums)} but only {len(ms.figures)} figure files were supplied")
    if not issues:
        return None
    return Finding(
        "Figure references are inconsistent: " + "; ".join(issues) + ".",
        "Number figures sequentially and supply every referenced figure.",
        [Evidence(section="Figures", issue=i) for i in issues],
        magnitude=len(issues),
    )


def _informal(ms: Manuscript, rc: RuleContext) -> Finding | None:
    hits = re.findall(r"\b(a lot of|lots of|huge|really|pretty good|gonna|stuff|super\s+\w+|awesome|basically)\b", ms.body, re.I)
    if not hits:
        return None
    return Finding(
        f"Informal language detected ({', '.join(sorted(set(h.lower() for h in hits)))}).",
        "Replace colloquial expressions with precise academic wording.",
        [Evidence(section="Throughout", issue=h) for h in hits[:3]],
        magnitude=len(hits),
    )


def _mixed_citation_style(ms: Manuscript, rc: RuleContext) -> Finding | None:
    numeric = len(re.findall(r"\[\d+(?:[,-]\s?\d+)*\]", ms.body))
    author_year = len(re.findall(r"\([A-Z][a-z]+(?: et al\.)?,? \d{4}\)", ms.body))
    if numeric and author_year:
        return Finding(
            "Numeric and author-year citation styles are mixed.",
            "Use a single citation style consistent with the journal guidelines.",
            [Evidence(section="Throughout", issue=f"{numeric} numeric vs {author_year} author-year citations")],
        )
    return None


# ------------------------------ compliance --------------------------------- #


_DECL_PATTERNS = {
    "data availability": r"data availability|data are available|data is available|available (up)?on (reasonable )?request",
    "conflict of interest": r"conflicts? of interest|competing interests?|declare no",
    "code availability": r"code availability|code is (publicly )?available|github\.com|source code",
    "ethics": r"ethic|institutional review board|\birb\b",
    "informed consent": r"informed consent|consent was obtained",
    "trial registration": r"trial registration|registered (at|with|in)|nct\d{6,}",
}


def _missing_declarations(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = _declarations_text(ms)
    # ethics / consent / registration are handled by dedicated rules
    wanted = [d for d in rc.policy["required_declarations"] if d not in {"ethics", "informed consent", "trial registration"}]
    missing = [d for d in wanted if not has_any(text, _DECL_PATTERNS.get(d, re.escape(d)))]
    if not missing:
        return None
    return Finding(
        f"Mandatory declarations are missing: {', '.join(missing)}.",
        f"Add statements for: {', '.join(missing)}.",
        [Evidence(section="Declarations", issue=m) for m in missing],
        magnitude=len(missing),
        items=missing,
    )


def _no_ethics(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = _declarations_text(ms)
    if not has_any(text, _HUMAN_SUBJECTS) or has_any(text, _DECL_PATTERNS["ethics"]):
        return None
    return Finding(
        "The study involves human participants/patient data but reports no ethics committee approval.",
        "Provide the name of the approving ethics committee/IRB and the approval number.",
        [Evidence(section="Methods", issue="human subjects without ethics approval")],
    )


def _no_consent(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = _declarations_text(ms)
    if not has_any(text, _HUMAN_SUBJECTS) or has_any(text, _DECL_PATTERNS["informed consent"]):
        return None
    return Finding(
        "No statement on informed consent from participants is provided.",
        "State how informed consent was obtained (or why it was waived, and by whom).",
        [Evidence(section="Methods", issue="no consent statement")],
    )


def _no_trial_registration(ms: Manuscript, rc: RuleContext) -> Finding | None:
    text = _declarations_text(ms)
    if not has_any(text, r"randomi[sz]ed (controlled )?trial|clinical trial") or has_any(text, _DECL_PATTERNS["trial registration"]):
        return None
    return Finding(
        "A clinical trial is reported without a trial registration number.",
        "Provide the trial registry and registration number (e.g. ClinicalTrials.gov NCT id).",
        [Evidence(section="Methods", issue="no registration")],
    )


def _word_limit(ms: Manuscript, rc: RuleContext) -> Finding | None:
    words = len(ms.body.split())
    limit = rc.policy["max_words"]
    if words <= limit:
        return None
    return Finding(
        f"The manuscript has {words} words, exceeding the {limit}-word limit.",
        f"Shorten the manuscript to at most {limit} words or move material to supplementary files.",
        [Evidence(section="Manuscript", issue=f"{words} words")],
        magnitude=words - limit,
    )


def _no_funding(ms: Manuscript, rc: RuleContext) -> Finding | None:
    if has_any(_declarations_text(ms), r"funding|funded by|grant|no external funding|financial support"):
        return None
    return Finding(
        "No funding statement is provided.",
        "Add a funding statement (or state that no funding was received).",
        [Evidence(section="Declarations", issue="funding")],
    )


# --------------------------------------------------------------------------- #
# Reviewer specifications
# --------------------------------------------------------------------------- #


@dataclass
class ReviewerSpec:
    reviewer_id: str
    name: str
    prefix: str
    capabilities: list[str]
    brief: str
    rules: list[Rule]


def _r(prefix: str, n: int, category: str, sev: Severity, fn) -> Rule:
    return Rule(f"{prefix}-{n:03d}", category, sev, fn)


REVIEWERS: dict[str, ReviewerSpec] = {
    spec.reviewer_id: spec
    for spec in [
        ReviewerSpec(
            "reviewer_structure", "Structural Reviewer", "STR", ["structure", "general"],
            "Assess manuscript organisation: presence and adequacy of abstract, introduction, methods, results, "
            "discussion, conclusion and references; logical flow between sections; whether the introduction states objectives.",
            [
                _r("STR", 1, "structure", Severity.MAJOR, _missing_sections),
                _r("STR", 2, "structure", Severity.MINOR, _abstract_length),
                _r("STR", 3, "references", Severity.MAJOR, _few_references),
                _r("STR", 4, "structure", Severity.MAJOR, _thin_core_sections),
                _r("STR", 5, "structure", Severity.MINOR, _no_objective),
                _r("STR", 6, "metadata", Severity.MINOR, _few_keywords),
            ],
        ),
        ReviewerSpec(
            "reviewer_methodology", "Technical/Methodology Reviewer", "METH", ["methodology", "reproducibility", "statistics"],
            "Assess experimental design, algorithms, data description, baselines, statistical methodology, "
            "reproducibility (hyperparameters, seeds, code) and technical correctness.",
            [
                _r("METH", 1, "data", Severity.MAJOR, _no_dataset),
                _r("METH", 2, "baselines", Severity.MAJOR, _no_baselines),
                _r("METH", 3, "statistics", Severity.MAJOR, _no_statistics),
                _r("METH", 4, "reproducibility", Severity.MAJOR, _no_reproducibility),
                _r("METH", 5, "reproducibility", Severity.MINOR, _no_repeated_runs),
            ],
        ),
        ReviewerSpec(
            "reviewer_novelty", "Novelty/Contribution Reviewer", "NOV", ["novelty"],
            "Assess the research contribution and novelty; compare against the retrieved related literature; "
            "flag potential duplication of existing approaches and uncited closely related work.",
            [
                _r("NOV", 1, "related_work", Severity.MAJOR, _no_related_work),
                _r("NOV", 2, "contribution", Severity.MAJOR, _no_contribution_claim),
                _r("NOV", 3, "related_work", Severity.MAJOR, _overlap_uncited),
                _r("NOV", 4, "references", Severity.MINOR, _no_recent_refs),
            ],
        ),
        ReviewerSpec(
            "reviewer_scientific_quality", "Scientific Quality Reviewer", "SCI", ["scientific_quality", "statistics"],
            "Assess scientific reasoning: whether the evidence supports the conclusions, interpretation of results, "
            "internal consistency (abstract vs results), limitations, and missing experiments or analyses.",
            [
                _r("SCI", 1, "claims", Severity.CRITICAL, _unsupported_claims),
                _r("SCI", 2, "evidence", Severity.CRITICAL, _results_not_quantitative),
                _r("SCI", 3, "limitations", Severity.MAJOR, _no_limitations),
                _r("SCI", 4, "consistency", Severity.MAJOR, _abstract_results_mismatch),
                _r("SCI", 5, "analysis", Severity.MINOR, _no_ablation),
            ],
        ),
        ReviewerSpec(
            "reviewer_language", "Language and Presentation Reviewer", "LANG", ["language", "structure"],
            "Assess academic writing: clarity, grammar, terminology and acronyms, figure/table presentation and "
            "numbering, and citation presentation.",
            [
                _r("LANG", 1, "clarity", Severity.MINOR, _long_sentences),
                _r("LANG", 2, "grammar", Severity.MINOR, _repeated_words),
                _r("LANG", 3, "terminology", Severity.MINOR, _undefined_acronyms),
                _r("LANG", 4, "figures", Severity.MINOR, _figure_numbering),
                _r("LANG", 5, "style", Severity.MINOR, _informal),
                _r("LANG", 6, "citations", Severity.MINOR, _mixed_citation_style),
            ],
        ),
        ReviewerSpec(
            "reviewer_compliance", "Compliance/Ethics Reviewer", "COMP", ["compliance", "ethics"],
            "Assess compliance with journal requirements: ethics approval, informed consent, trial registration, "
            "data/code availability, conflict-of-interest and funding statements, word limits and required supplementary information.",
            [
                _r("COMP", 1, "declarations", Severity.MAJOR, _missing_declarations),
                _r("COMP", 2, "ethics", Severity.CRITICAL, _no_ethics),
                _r("COMP", 3, "ethics", Severity.CRITICAL, _no_consent),
                _r("COMP", 4, "registration", Severity.MAJOR, _no_trial_registration),
                _r("COMP", 5, "policy", Severity.MAJOR, _word_limit),
                _r("COMP", 6, "declarations", Severity.MINOR, _no_funding),
            ],
        ),
    ]
}


# --------------------------------------------------------------------------- #
# Recommendation policy (shared by heuristic reviewers)
# --------------------------------------------------------------------------- #


def recommend(concerns: list[Concern]) -> Recommendation:
    crit = sum(c.severity == Severity.CRITICAL for c in concerns)
    major = sum(c.severity == Severity.MAJOR for c in concerns)
    if crit >= 2:
        return Recommendation.REJECT
    if crit == 1 or major >= 2:
        return Recommendation.MAJOR_REVISION
    if major == 1 or concerns:
        return Recommendation.MINOR_REVISION
    return Recommendation.ACCEPT


# --------------------------------------------------------------------------- #
# Reviewer agent
# --------------------------------------------------------------------------- #


class LLMResolutionItem(BaseModel):
    concern_id: str
    status: ResolutionStatus
    author_response: str
    revision_evidence: str
    remaining_issue: str


class LLMResolutionBatch(BaseModel):
    resolutions: list[LLMResolutionItem]


class ReviewerAgent:
    def __init__(self, spec: ReviewerSpec, ctx: AgentContext):
        self.spec = spec
        self.ctx = ctx

    def _rule_context(self, ms: Manuscript) -> RuleContext:
        policy = self.ctx.tools.get_policy(ms.article_type, ms.domain)
        return RuleContext(policy=policy, ctx=self.ctx, domain=ms.domain or "general")

    def _run_rules(self, ms: Manuscript) -> list[Concern]:
        rc = self._rule_context(ms)
        out = []
        for rule in self.spec.rules:
            f = rule.check(ms, rc)
            if f:
                out.append(
                    Concern(
                        concern_id=rule.rule_id,
                        reviewer_id=self.spec.reviewer_id,
                        category=rule.category,
                        severity=rule.severity,
                        description=f.description,
                        required_change=f.required_change,
                        evidence=f.evidence,
                        rule_id=rule.rule_id,
                        magnitude=f.magnitude,
                        items=f.items,
                    )
                )
        return out

    # ------------------------------------------------------------ review
    def review(self, ms: Manuscript, round_: int = 1) -> ReviewReport:
        if self.ctx.llm:
            return self._llm_review(ms, round_)
        concerns = self._run_rules(ms)
        rec = recommend(concerns)
        n_checks = len(self.spec.rules)
        strengths = [
            f"Meets the {r.category} criterion checked by {r.rule_id}"
            for r in self.spec.rules
            if r.rule_id not in {c.rule_id for c in concerns}
        ][:3]
        return ReviewReport(
            manuscript_id=ms.manuscript_id,
            reviewer_id=self.spec.reviewer_id,
            round=round_,
            recommendation=rec,
            confidence=round(0.55 + 0.35 * (n_checks - len(concerns)) / n_checks if concerns else 0.9, 2),
            summary=(
                f"{self.spec.name}: {len(concerns)} issue(s) found across {n_checks} checks "
                f"({sum(c.severity != Severity.MINOR for c in concerns)} major/critical)."
            ),
            strengths=strengths,
            concerns=concerns,
        )

    def _llm_review(self, ms: Manuscript, round_: int) -> ReviewReport:
        policy = self.ctx.tools.get_policy(ms.article_type, ms.domain)
        extra = ""
        if self.spec.reviewer_id == "reviewer_novelty":
            hits = self.ctx.tools.search_literature(f"{ms.title} {ms.abstract}", k=8)
            extra = "\n\nRETRIEVED RELATED LITERATURE (from the literature-search tool):\n" + "\n".join(
                f"- {h['title']} ({h.get('year')}), doi {h.get('doi')}: {h.get('abstract', '')[:400]}" for h in hits
            )
        instruction = (
            f"You are the {self.spec.name} (id {self.spec.reviewer_id}). Review ONLY your area: {self.spec.brief}\n"
            f"Journal policy for this article: {policy}{extra}\n\n"
            "Produce a structured review. Each concern needs a category, severity, a precise description, "
            "the required change, and evidence entries (section + location + issue). "
            f"Use concern ids of the form {self.spec.prefix}-001, {self.spec.prefix}-002, ... "
            "Recommendation must reflect only the issues in your area. Confidence is your calibrated confidence in the recommendation."
        )
        out = self.ctx.llm.structured(context=render_for_prompt(ms), instruction=instruction, schema=LLMReviewOutput)
        concerns = []
        for i, c in enumerate(out.concerns, 1):
            concerns.append(c.model_copy(update={"concern_id": f"{self.spec.prefix}-{i:03d}", "reviewer_id": self.spec.reviewer_id}))
        return ReviewReport(
            manuscript_id=ms.manuscript_id,
            reviewer_id=self.spec.reviewer_id,
            round=round_,
            recommendation=out.recommendation,
            confidence=out.confidence,
            summary=out.summary,
            strengths=out.strengths,
            concerns=concerns,
        )

    # ------------------------------------------------ concern resolution
    def verify(
        self,
        original: Manuscript,
        revised: Manuscript,
        concerns: list[Concern],
        response_letter: str,
    ) -> list[ConcernResolution]:
        if not concerns:
            return []
        if self.ctx.llm:
            return self._llm_verify(revised, concerns, response_letter)
        current = {c.rule_id: c for c in self._run_rules(revised)}
        changed = _changed_sections(original, revised)
        out = []
        for c in concerns:
            response = _response_for(c, response_letter)
            still = current.get(c.rule_id)
            evidence = _best_evidence(c, changed)
            left = set(c.items) & set(still.items) if still is not None and c.items else None
            if still is None or left == set():
                status, remaining = ResolutionStatus.ADDRESSED, ""
            elif left is not None and len(left) < len(c.items):
                status, remaining = ResolutionStatus.PARTIALLY_ADDRESSED, f"Still outstanding: {', '.join(sorted(left))}"
            elif left is None and (c.magnitude or 1) > (still.magnitude or 1):
                status, remaining = ResolutionStatus.PARTIALLY_ADDRESSED, still.description
            else:
                status, remaining = ResolutionStatus.NOT_ADDRESSED, still.description
            out.append(
                ConcernResolution(
                    concern_id=c.concern_id,
                    reviewer_id=self.spec.reviewer_id,
                    original_concern=c.description,
                    severity=c.severity,
                    author_response=response,
                    revision_evidence=evidence if status != ResolutionStatus.NOT_ADDRESSED else "",
                    status=status,
                    remaining_issue=remaining,
                )
            )
        return out

    def _llm_verify(self, revised: Manuscript, concerns: list[Concern], response_letter: str) -> list[ConcernResolution]:
        listing = "\n".join(
            f"- {c.concern_id} [{c.severity.value}] {c.description} | required change: {c.required_change}" for c in concerns
        )
        instruction = (
            f"You are the {self.spec.name}. The manuscript above is the REVISED version. In the previous round you raised:\n"
            f"{listing}\n\nAUTHOR RESPONSE TO REVIEWERS:\n{response_letter or '(none provided)'}\n\n"
            "For EVERY concern id, decide whether it is addressed, partially_addressed or not_addressed, judging by the "
            "revised manuscript text (not by the author's claims alone). Quote the section/location that provides the "
            "revision evidence, summarise the author's response, and state any remaining issue."
        )
        out = self.ctx.llm.structured(context=render_for_prompt(revised), instruction=instruction, schema=LLMResolutionBatch)
        by_id = {r.concern_id: r for r in out.resolutions}
        res = []
        for c in concerns:
            r = by_id.get(c.concern_id)
            res.append(
                ConcernResolution(
                    concern_id=c.concern_id,
                    reviewer_id=self.spec.reviewer_id,
                    original_concern=c.description,
                    severity=c.severity,
                    author_response=r.author_response if r else "",
                    revision_evidence=r.revision_evidence if r else "",
                    status=r.status if r else ResolutionStatus.NOT_ADDRESSED,
                    remaining_issue=r.remaining_issue if r else "Concern not assessed by the verifier.",
                )
            )
        return res


def _changed_sections(a: Manuscript, b: Manuscript) -> dict[str, str]:
    old = {s.heading.lower(): s.text for s in a.sections}
    return {s.heading: s.text for s in b.sections if old.get(s.heading.lower()) != s.text}


def _best_evidence(c: Concern, changed: dict[str, str]) -> str:
    if not changed:
        return ""
    target = f"{c.description} {c.required_change}"
    heading = max(changed, key=lambda h: jaccard(target, h + " " + changed[h]))
    return f"Revised section: {heading}"


def _response_for(c: Concern, letter: str) -> str:
    """Pick the paragraph of the response letter that answers this concern."""
    if not letter:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-*] )", letter) if p.strip()]
    for p in paras:
        if c.concern_id in p:
            return p
    kw = set(tokens(c.description))
    best = max(paras, key=lambda p: len(kw & set(tokens(p))), default="")
    return best if len(kw & set(tokens(best))) >= 2 else ""
