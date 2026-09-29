"""REST API (FastAPI). Long-running agent work runs in background tasks; clients
(web UI, n8n) poll GET /manuscripts/{id} or receive n8n webhook notifications."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from .agents import REVIEWERS
from .documents import extract_title_and_abstract, load_manuscript_text
from .schemas import HumanDecision, ManuscriptSubmission, RevisionSubmission
from .service import JournalService

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_service() -> JournalService:
    return JournalService()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("JOURNAL_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(service: JournalService | None = None) -> FastAPI:
    app = FastAPI(title="Agentic Journal Management API", version="1.0.0")
    svc_dep = (lambda: service) if service else get_service

    def run_safely(svc: JournalService, manuscript_id: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # surfaced through the status endpoint
            log.exception("workflow step failed for %s", manuscript_id)
            svc.store.set_status(manuscript_id, "error")
            svc.store.log_event(manuscript_id, "system", "error", {"error": repr(exc)})

    def summary_or_404(svc: JournalService, manuscript_id: str) -> dict[str, Any]:
        try:
            return svc.summary(manuscript_id)
        except KeyError:
            status = svc.store.get_status(manuscript_id)
            if status is None:
                raise HTTPException(404, f"unknown manuscript {manuscript_id}")
            return {"manuscript_id": manuscript_id, "status": status, "pending_action": None}

    def expect_pending(svc: JournalService, manuscript_id: str, kind: str) -> None:
        summary_or_404(svc, manuscript_id)
        pending = svc.pending_action(manuscript_id)
        if not pending or pending.get("type") != kind:
            raise HTTPException(409, f"manuscript is not awaiting {kind} (pending: {pending and pending.get('type')})")

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health(svc: JournalService = Depends(svc_dep)) -> dict[str, Any]:
        return {"status": "ok", "llm_mode": svc.ctx.mode, "tools": svc.tools.name, "model": svc.settings.model}

    @app.get("/agents")
    def agents(svc: JournalService = Depends(svc_dep)) -> dict[str, Any]:
        return {
            "agents": svc.team.agent_ids,
            "reviewer_pool": [
                {"reviewer_id": s.reviewer_id, "name": s.name, "capabilities": s.capabilities, "checks": [r.rule_id for r in s.rules]}
                for s in REVIEWERS.values()
            ],
        }

    @app.get("/stats")
    def stats(svc: JournalService = Depends(svc_dep)) -> dict[str, Any]:
        rows = svc.store.list_manuscripts()
        return {"manuscripts": len(rows), "by_status": dict(Counter(r["status"] for r in rows))}

    # ------------------------------------------------------------ manuscripts
    @app.post("/manuscripts", status_code=202, dependencies=[Depends(require_api_key)])
    def submit(sub: ManuscriptSubmission, bg: BackgroundTasks, svc: JournalService = Depends(svc_dep)) -> dict[str, str]:
        ms_id = svc.new_manuscript_id()
        svc.store.set_status(ms_id, "queued")
        bg.add_task(run_safely, svc, ms_id, svc.submit, sub, ms_id)
        return {"manuscript_id": ms_id, "status": "queued"}

    @app.post("/manuscripts/upload", status_code=202, dependencies=[Depends(require_api_key)])
    async def upload(
        bg: BackgroundTasks,
        file: UploadFile = File(...),
        metadata: str = Form("{}"),
        svc: JournalService = Depends(svc_dep),
    ) -> dict[str, str]:
        suffix = Path(file.filename or "manuscript.md").suffix.lower()
        if suffix not in {".pdf", ".md", ".txt"}:
            raise HTTPException(415, "upload a .pdf, .md or .txt manuscript")
        dest = svc.settings.data_dir / "uploads" / f"{svc.store.next_sequence('upload'):06d}{suffix}"
        dest.write_bytes(await file.read())
        body = load_manuscript_text(dest)
        title, abstract = extract_title_and_abstract(body)
        meta = json.loads(metadata or "{}")
        meta.setdefault("title", title or Path(file.filename or "Untitled").stem)
        meta.setdefault("abstract", abstract or "")
        sub = ManuscriptSubmission(**meta, body=body)
        ms_id = svc.new_manuscript_id()
        svc.store.set_status(ms_id, "queued")
        bg.add_task(run_safely, svc, ms_id, svc.submit, sub, ms_id)
        return {"manuscript_id": ms_id, "status": "queued"}

    @app.get("/manuscripts")
    def list_manuscripts(svc: JournalService = Depends(svc_dep)) -> list[dict[str, Any]]:
        return svc.store.list_manuscripts()

    @app.get("/manuscripts/{manuscript_id}")
    def get_manuscript(manuscript_id: str, svc: JournalService = Depends(svc_dep)) -> dict[str, Any]:
        return summary_or_404(svc, manuscript_id)

    @app.get("/manuscripts/{manuscript_id}/audit")
    def audit(manuscript_id: str, svc: JournalService = Depends(svc_dep)) -> list[dict[str, Any]]:
        summary_or_404(svc, manuscript_id)
        return svc.audit_trail(manuscript_id)

    @app.get("/manuscripts/{manuscript_id}/concerns")
    def concerns(manuscript_id: str, svc: JournalService = Depends(svc_dep)) -> list[dict[str, Any]]:
        summary_or_404(svc, manuscript_id)
        return svc.concern_matrix(manuscript_id)

    @app.post("/manuscripts/{manuscript_id}/revision", status_code=202, dependencies=[Depends(require_api_key)])
    def revision(manuscript_id: str, rev: RevisionSubmission, bg: BackgroundTasks, svc: JournalService = Depends(svc_dep)):
        expect_pending(svc, manuscript_id, "revision_request")
        bg.add_task(run_safely, svc, manuscript_id, svc.submit_revision, manuscript_id, rev)
        return {"manuscript_id": manuscript_id, "status": "revision_queued"}

    @app.post("/manuscripts/{manuscript_id}/withdraw", dependencies=[Depends(require_api_key)])
    def withdraw(manuscript_id: str, svc: JournalService = Depends(svc_dep)):
        expect_pending(svc, manuscript_id, "revision_request")
        return svc.withdraw(manuscript_id)

    @app.post("/manuscripts/{manuscript_id}/editor-decision", status_code=202, dependencies=[Depends(require_api_key)])
    def editor_decision(manuscript_id: str, decision: HumanDecision, bg: BackgroundTasks, svc: JournalService = Depends(svc_dep)):
        expect_pending(svc, manuscript_id, "editor_approval")
        bg.add_task(run_safely, svc, manuscript_id, svc.editor_decision, manuscript_id, decision)
        return {"manuscript_id": manuscript_id, "status": "decision_queued"}

    @app.get("/manuscripts/{manuscript_id}/pdf")
    def pdf(manuscript_id: str, svc: JournalService = Depends(svc_dep)):
        prod = summary_or_404(svc, manuscript_id).get("production")
        if not prod:
            raise HTTPException(404, "not in production yet")
        return FileResponse(prod["pdf_path"], media_type="application/pdf", filename=Path(prod["pdf_path"]).name)

    @app.get("/manuscripts/{manuscript_id}/metadata/{fmt}")
    def metadata_xml(manuscript_id: str, fmt: str, svc: JournalService = Depends(svc_dep)):
        summary_or_404(svc, manuscript_id)
        pub = svc.state(manuscript_id).get("publication")
        if not pub or fmt not in {"jats", "crossref"}:
            raise HTTPException(404, "publication metadata not available")
        return Response(pub[f"{fmt}_xml"], media_type="application/xml")

    return app


app = create_app()
