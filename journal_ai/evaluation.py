"""Research evaluation harness.

Metrics from the proposal (Section 13):
  * reviewer agreement - Fleiss' kappa over recommendations, mean pairwise ordinal agreement
  * issue detection    - precision / recall / F1 of detected concern ids vs gold annotations
  * concern tracking   - accuracy of addressed / partially / not-addressed labels vs gold
  * decision accuracy  - final editorial outcome vs gold
  * efficiency         - wall-clock, LLM calls, tokens per manuscript

The gold file (data/eval/gold.json) annotates the bundled samples. For real
studies, replace it with annotated OpenReview/journal data in the same format.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from . import samples
from .schemas import RECOMMENDATION_ORDER, HumanDecision, Recommendation
from .service import JournalService

GOLD_FILE = Path(__file__).resolve().parents[1] / "data" / "eval" / "gold.json"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss' kappa. ``ratings[i]`` holds the labels the raters gave item i
    (items may have different numbers of raters; items with <2 raters are skipped)."""
    items = [r for r in ratings if len(r) >= 2]
    if not items:
        return float("nan")
    cats = sorted({c for r in items for c in r})
    p_items, totals, n_total = [], Counter(), 0
    for r in items:
        n = len(r)
        counts = Counter(r)
        totals.update(counts)
        n_total += n
        p_items.append((sum(v * v for v in counts.values()) - n) / (n * (n - 1)))
    p_bar = sum(p_items) / len(p_items)
    p_e = sum((totals[c] / n_total) ** 2 for c in cats)
    if p_e == 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def pairwise_ordinal_agreement(labels: list[str]) -> float:
    vals = [RECOMMENDATION_ORDER[Recommendation(v)] for v in labels]
    pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1 :]]
    return 1.0 if not pairs else sum(1 - abs(a - b) / 3 for a, b in pairs) / len(pairs)


def prf(predicted: set[str], gold: set[str]) -> dict[str, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp,
            "fp": len(predicted - gold), "fn": len(gold - predicted)}


def accuracy(pred: dict[str, str], gold: dict[str, str]) -> float:
    keys = [k for k in gold if k in pred]
    if not gold:
        return 1.0
    return sum(pred[k] == gold[k] for k in keys) / len(gold)


# --------------------------------------------------------------------------- #
# Running a manuscript through the full lifecycle with a scripted author/editor
# --------------------------------------------------------------------------- #


def run_lifecycle(svc: JournalService, name: str, editor: Callable[[dict], HumanDecision] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    ms_id = svc.submit(samples.load_submission(name))
    n = 0
    while True:
        pending = svc.pending_action(ms_id)
        if not pending:
            break
        if pending["type"] == "editor_approval":
            decision = editor(pending) if editor else HumanDecision(decision=pending["recommendation"], editor="eval")
            svc.editor_decision(ms_id, decision)
        else:
            n += 1
            try:
                svc.submit_revision(ms_id, samples.load_revision(name, n))
            except FileNotFoundError:
                svc.withdraw(ms_id)
    summary = svc.summary(ms_id)
    summary["wall_clock_s"] = round(time.perf_counter() - t0, 3)
    summary["revisions_submitted"] = n
    return summary


def evaluate(service_factory: Callable[[], JournalService], gold_file: Path = GOLD_FILE) -> dict[str, Any]:
    gold = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    per_item: list[dict[str, Any]] = []
    rec_ratings: list[list[str]] = []
    det_pred: set[str] = set()
    det_gold: set[str] = set()
    res_pred: dict[str, str] = {}
    res_gold: dict[str, str] = {}
    decisions_ok = 0

    for g in gold["manuscripts"]:
        svc = service_factory()
        s = run_lifecycle(svc, g["sample"])
        final = s["status"]
        ok = final == g["expected_final_status"]
        decisions_ok += ok

        r1 = [r for r in s["reviews"] if r["round"] == 1]
        if r1:
            rec_ratings.append([r["recommendation"] for r in r1])
        found = {c["concern_id"] for r in r1 for c in r["concerns"]}
        det_pred |= {f"{g['sample']}:{c}" for c in found}
        det_gold |= {f"{g['sample']}:{c}" for c in g.get("gold_concerns", [])}

        for rnd, statuses in g.get("gold_resolutions", {}).items():
            for cid, st in statuses.items():
                res_gold[f"{g['sample']}:{rnd}:{cid}"] = st
        for r in s["resolutions"]:
            res_pred[f"{g['sample']}:{r['round']}:{r['concern_id']}"] = r["status"]

        m = s.get("metrics") or {}
        per_item.append(
            {
                "sample": g["sample"],
                "manuscript_id": s["manuscript_id"],
                "final_status": final,
                "expected_final_status": g["expected_final_status"],
                "correct": ok,
                "rounds": s.get("round"),
                "revisions": s["revisions_submitted"],
                "reviewers": [a["reviewer_id"] for a in (s.get("assignments") or [])],
                "agreement_round1": (s.get("aggregated") or {}).get("agreement"),
                "wall_clock_s": s["wall_clock_s"],
                "llm_calls": m.get("llm_calls", 0),
                "input_tokens": m.get("input_tokens", 0),
                "output_tokens": m.get("output_tokens", 0),
            }
        )

    n = len(per_item)
    return {
        "n_manuscripts": n,
        "decision_accuracy": round(decisions_ok / n, 3) if n else None,
        "issue_detection": prf(det_pred, det_gold),
        "concern_resolution_accuracy": round(accuracy(res_pred, res_gold), 3),
        "reviewer_agreement": {
            "fleiss_kappa": round(fleiss_kappa(rec_ratings), 3) if rec_ratings else None,
            "mean_pairwise_ordinal": round(sum(pairwise_ordinal_agreement(r) for r in rec_ratings) / len(rec_ratings), 3)
            if rec_ratings
            else None,
        },
        "efficiency": {
            "mean_wall_clock_s": round(sum(i["wall_clock_s"] for i in per_item) / n, 3) if n else None,
            "total_llm_calls": sum(i["llm_calls"] for i in per_item),
            "total_input_tokens": sum(i["input_tokens"] for i in per_item),
            "total_output_tokens": sum(i["output_tokens"] for i in per_item),
            "workflow_completion_rate": round(
                sum(i["final_status"] in {"published", "rejected", "desk_rejected", "withdrawn"} for i in per_item) / n, 3
            ) if n else None,
        },
        "manuscripts": per_item,
    }
