import json

import pytest
from fastapi.testclient import TestClient

from journal_ai import samples
from journal_ai.api import create_app


@pytest.fixture
def client(svc):
    return TestClient(create_app(svc))


def test_health_and_agents(client):
    assert client.get("/health").json()["llm_mode"] == "heuristic"
    agents = client.get("/agents").json()
    assert len(agents["agents"]) == 10 and len(agents["reviewer_pool"]) == 6


def test_full_http_lifecycle(client):
    sub = samples.load_submission("cv_segmentation").model_dump(mode="json")
    r = client.post("/manuscripts", json=sub)
    assert r.status_code == 202
    ms_id = r.json()["manuscript_id"]

    s = client.get(f"/manuscripts/{ms_id}").json()  # background task already ran under TestClient
    assert s["status"] == "decision_pending"
    assert s["pending_action"]["recommendation"] == "minor_revision"
    assert s["corresponding_author"] == "amina.rahman@example.edu"

    # wrong action -> 409
    rev = samples.load_revision("cv_segmentation", 1).model_dump(mode="json")
    assert client.post(f"/manuscripts/{ms_id}/revision", json=rev).status_code == 409

    assert client.post(f"/manuscripts/{ms_id}/editor-decision", json={"decision": "minor_revision", "editor": "E"}).status_code == 202
    assert client.post(f"/manuscripts/{ms_id}/revision", json=rev).status_code == 202
    assert client.post(f"/manuscripts/{ms_id}/editor-decision", json={"decision": "accept", "editor": "E"}).status_code == 202

    s = client.get(f"/manuscripts/{ms_id}").json()
    assert s["status"] == "published"
    pdf = client.get(f"/manuscripts/{ms_id}/pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content[:4] == b"%PDF"
    assert client.get(f"/manuscripts/{ms_id}/metadata/jats").text.startswith("<article")
    assert client.get(f"/manuscripts/{ms_id}/concerns").json()[0]["rounds"] == {"2": "addressed"}
    assert len(client.get(f"/manuscripts/{ms_id}/audit").json()) > 10
    assert client.get("/stats").json()["by_status"] == {"published": 1}
    assert client.get("/manuscripts").json()[0]["manuscript_id"] == ms_id


def test_unknown_manuscript_404(client):
    assert client.get("/manuscripts/MS-NOPE").status_code == 404
    assert client.post("/manuscripts/MS-NOPE/withdraw").status_code == 404


def test_upload_markdown_and_pdf(client, tmp_path):
    body = (samples.SAMPLES_DIR / "ml_optimizer" / "manuscript.md").read_bytes()
    meta = json.loads((samples.SAMPLES_DIR / "ml_optimizer" / "meta.json").read_text(encoding="utf-8"))
    r = client.post("/manuscripts/upload", files={"file": ("paper.md", body, "text/markdown")}, data={"metadata": json.dumps(meta)})
    assert r.status_code == 202
    s = client.get(f"/manuscripts/{r.json()['manuscript_id']}").json()
    assert s["status"] == "decision_pending" and s["title"].startswith("MomentumBoost")

    assert client.post("/manuscripts/upload", files={"file": ("x.docx", b"...", "application/octet-stream")}).status_code == 415


def test_api_key_enforced_on_writes(client, monkeypatch):
    monkeypatch.setenv("JOURNAL_API_KEY", "secret")
    sub = samples.load_submission("cv_segmentation").model_dump(mode="json")
    assert client.post("/manuscripts", json=sub).status_code == 401
    assert client.post("/manuscripts", json=sub, headers={"X-API-Key": "secret"}).status_code == 202
    assert client.get("/health").status_code == 200  # reads stay open


def test_background_failure_is_reported(client, svc, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM outage")

    monkeypatch.setattr(svc, "submit", boom)
    ms_id = client.post("/manuscripts", json=samples.load_submission("cv_segmentation").model_dump(mode="json")).json()["manuscript_id"]
    s = client.get(f"/manuscripts/{ms_id}").json()
    assert s["status"] == "error"
    assert "LLM outage" in svc.audit_trail(ms_id)[-1]["payload"]["error"]
