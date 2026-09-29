"""Static checks of the deployment artefacts (Kubernetes, docker-compose, n8n)."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _k8s_docs():
    docs = []
    for f in sorted((ROOT / "k8s").glob("*.yaml")):
        docs += [d for d in yaml.safe_load_all(f.read_text(encoding="utf-8")) if d]
    return docs


def test_kubernetes_manifests():
    docs = _k8s_docs()
    kinds = {(d["kind"], d["metadata"].get("name")) for d in docs if "metadata" in d}
    for expected in [("Namespace", "journal-ai"), ("ConfigMap", "journal-config"), ("Deployment", "journal-api"),
                     ("Service", "journal-api"), ("StatefulSet", "postgres"), ("Deployment", "n8n"), ("Ingress", "journal-api")]:
        assert expected in kinds
    kustom = next(d for d in docs if d["kind"] == "Kustomization")
    assert {r for r in kustom["resources"]} == {p.name for p in (ROOT / "k8s").glob("*.yaml") if p.name != "kustomization.yaml"}

    api = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "journal-api")
    containers = {c["name"]: c for c in api["spec"]["template"]["spec"]["containers"]}
    assert set(containers) == {"api", "mcp-tools"}
    assert "journal_ai.mcp_server" in containers["mcp-tools"]["command"]
    cfg = next(d for d in docs if d["kind"] == "ConfigMap")["data"]
    assert cfg["JOURNAL_MCP_URL"].endswith(":8765/mcp") and cfg["JOURNAL_CHECKPOINTER"] == "postgres"
    # every config key is a real setting
    from journal_ai.config import Settings
    import inspect
    src = inspect.getsource(Settings)
    for key in cfg:
        assert key in src or key == "N8N_WEBHOOK_URL", key
    # no secrets committed
    assert not any(d["kind"] == "Secret" for d in docs)


def test_compose():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(c["services"]) == {"api", "mcp", "postgres", "n8n"}
    assert c["services"]["api"]["environment"]["JOURNAL_MCP_URL"] == "http://mcp:8765/mcp"


def test_n8n_workflows_are_wired():
    for f in (ROOT / "n8n").glob("*.json"):
        wf = json.loads(f.read_text(encoding="utf-8"))
        names = {n["name"] for n in wf["nodes"]}
        for src, conn in wf["connections"].items():
            assert src in names
            for branch in conn["main"]:
                for edge in branch:
                    assert edge["node"] in names, (f.name, edge)
    intake = json.loads((ROOT / "n8n" / "journal-intake.json").read_text(encoding="utf-8"))
    http = next(n for n in intake["nodes"] if n["type"] == "n8n-nodes-base.httpRequest")
    assert http["parameters"]["url"].endswith("/manuscripts")
    events = json.loads((ROOT / "n8n" / "journal-events.json").read_text(encoding="utf-8"))
    hook = next(n for n in events["nodes"] if n["type"] == "n8n-nodes-base.webhook")
    assert hook["parameters"]["path"] == "journal-events"


def test_dockerfile_runs_api():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "journal_ai.api:app" in df and "USER journal" in df
