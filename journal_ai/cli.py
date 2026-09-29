"""Command line entry point.

  python -m journal_ai.cli demo ml_optimizer      # full lifecycle of a bundled sample
  python -m journal_ai.cli submit paper.pdf --meta meta.json
  python -m journal_ai.cli evaluate               # research metrics over data/eval/gold.json
  python -m journal_ai.cli status MS-2026-00001
"""

from __future__ import annotations

import argparse
import json
import sys

from . import samples
from .config import Settings
from .documents import extract_title_and_abstract, load_manuscript_text
from .evaluation import evaluate, run_lifecycle
from .schemas import ManuscriptSubmission
from .service import JournalService


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_demo(args) -> None:
    svc = JournalService()
    s = run_lifecycle(svc, args.sample)
    print(f"\n{s['manuscript_id']}  '{s['title']}'  domain={s['domain']}")
    print(f"final status: {s['status']}   rounds: {s['round']}   revisions: {s['revisions_submitted']}")
    for r in [r for r in s["reviews"] if r["round"] == 1]:
        print(f"  {r['reviewer_id']:<30} {r['recommendation']:<15} concerns={[c['concern_id'] for c in r['concerns']]}")
    print("\nconcern resolution matrix:")
    for row in svc.concern_matrix(s["manuscript_id"]):
        print(f"  {row['concern_id']:<9} {row['severity']:<9} {row['rounds']}")
    if s.get("publication"):
        print(f"\nDOI {s['publication']['doi']}  PDF {s['production']['pdf_path']}  print-ready={s['production']['print_ready']}")
    print(f"\naudit events: {len(svc.audit_trail(s['manuscript_id']))}   llm mode: {svc.ctx.mode}")


def cmd_submit(args) -> None:
    body = load_manuscript_text(args.file)
    title, abstract = extract_title_and_abstract(body)
    meta = json.load(open(args.meta, encoding="utf-8")) if args.meta else {}
    meta.setdefault("title", title or "Untitled")
    meta.setdefault("abstract", abstract or "")
    svc = JournalService()
    ms_id = svc.submit(ManuscriptSubmission(**meta, body=body))
    _print(svc.summary(ms_id))


def cmd_status(args) -> None:
    _print(JournalService().summary(args.manuscript_id))


def cmd_evaluate(args) -> None:
    base = Settings()

    def factory() -> JournalService:
        s = Settings()
        s.checkpointer = "memory"
        s.require_human_approval = base.require_human_approval
        return JournalService(s)

    _print(evaluate(factory))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="journal_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("sample", choices=samples.available())
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("submit")
    s.add_argument("file")
    s.add_argument("--meta")
    s.set_defaults(fn=cmd_submit)
    st = sub.add_parser("status")
    st.add_argument("manuscript_id")
    st.set_defaults(fn=cmd_status)
    e = sub.add_parser("evaluate")
    e.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
