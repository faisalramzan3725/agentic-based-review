import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp import Client

from journal_ai.mcp_server import build_server
from journal_ai.tools import MCPTools

EXPECTED = {
    # workflow tools (for MCP clients such as Claude Code / Claude Desktop)
    "review_manuscript", "review_manuscript_file", "submit_manuscript", "submit_manuscript_file",
    "list_sample_manuscripts", "submit_sample_manuscript", "get_manuscript_status", "editor_decision",
    "submit_revision", "withdraw_manuscript", "get_concern_matrix",
    # service tools (used by the agents themselves)
    "search_literature", "check_duplicate", "get_policy", "journal_scope", "get_manuscript",
    "list_manuscripts", "register_doi", "validate_pdf", "log_event", "get_audit_trail",
}
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_server_lists_all_tools(settings):
    async def go():
        async with Client(build_server(settings)) as c:
            return {t.name for t in (await c.list_tools()).tools}

    assert asyncio.run(go()) == EXPECTED


def test_tool_calls_roundtrip(settings, svc):
    from journal_ai import samples

    ms_id = svc.submit(samples.load_submission("cv_segmentation"))
    tools = MCPTools(build_server(settings))
    ms = tools._call("get_manuscript", manuscript_id=ms_id)
    assert ms["title"].startswith("LiteSeg")
    listed = tools._call("list_manuscripts")
    assert listed[0]["manuscript_id"] == ms_id and listed[0]["status"] == "decision_pending"
    trail = tools._call("get_audit_trail", manuscript_id=ms_id)
    assert trail[0]["event"] == "manuscript_received"
    assert tools.search_literature("attention transformer translation", 1)[0]["id"] == "L004"


def test_one_shot_review_tools(settings, svc):
    tools = MCPTools(build_server(settings, service=svc))
    r = tools._call("review_manuscript_file", path=str(SAMPLES / "clinical_trial" / "manuscript.md"))
    assert r["domain"] == "clinical" and r["screening_passed"]
    comp = next(x for x in r["reviews"] if x["reviewer"] == "reviewer_compliance")
    assert [c["id"] for c in comp["concerns"]] == ["COMP-002", "COMP-003", "COMP-004"]
    assert r["recommendation"] == "major_revision" and "COMP-002" in r["decision_letter"]
    assert svc.store.list_manuscripts() == []  # quick review stores nothing

    body = (SAMPLES / "incomplete" / "manuscript.md").read_text(encoding="utf-8")
    r = tools._call("review_manuscript", title="Notes", body=body)
    assert r["recommendation"] == "desk_reject" and r["screening_failures"]


def test_full_workflow_through_mcp_tools(settings, svc):
    tools = MCPTools(build_server(settings, service=svc))
    s = tools._call("submit_manuscript_file", path=str(SAMPLES / "cv_segmentation" / "manuscript.md"))
    ms_id = s["manuscript_id"]
    assert s["title"].startswith("LiteSeg") and s["pending_action"]["type"] == "editor_approval"
    assert s["recommendation"] == "minor_revision"
    s = tools._call("editor_decision", manuscript_id=ms_id, decision="minor_revision", editor="Dr E")
    assert s["pending_action"]["type"] == "revision_request"
    d = SAMPLES / "cv_segmentation"
    s = tools._call("submit_revision", manuscript_id=ms_id, body_path=str(d / "revision_1.md"),
                    response_path=str(d / "response_1.md"))
    assert s["resolutions_this_round"] == ["LANG-003: addressed"] and s["recommendation"] == "accept"
    s = tools._call("editor_decision", manuscript_id=ms_id, decision="accept")
    assert s["status"] == "published" and s["doi"].startswith("10.5555/") and s["print_ready"]
    assert tools._call("get_concern_matrix", manuscript_id=ms_id)[0]["rounds"] == {"2": "addressed"}
    assert tools._call("get_manuscript_status", manuscript_id=ms_id)["status"] == "published"
    with pytest.raises(RuntimeError):  # wrong action surfaces as an MCP tool error
        tools._call("submit_revision", manuscript_id=ms_id, body="x")


@pytest.mark.slow
def test_stdio_server_as_launched_by_mcp_clients(settings):
    """Launch the server from .mcp.json over stdio, the way Claude Code / Claude Desktop do."""
    import json

    from mcp.client.stdio import StdioServerParameters

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["journal-ai"]
    env = {**os.environ, **cfg["env"], "JOURNAL_DATA_DIR": str(settings.data_dir), "JOURNAL_LLM_BACKEND": "heuristic"}
    params = StdioServerParameters(command=sys.executable, args=cfg["args"], env=env)

    async def go():
        async with Client(params) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool("submit_sample_manuscript", {"name": "cv_segmentation"})
            return names, res.structured_content

    names, s = asyncio.run(go())
    assert names == EXPECTED
    assert s["status"] == "decision_pending" and s["domain"] == "computer_vision"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_streamable_http_server_process(settings):
    """Starts the real MCP server process (as deployed in the k8s sidecar) and calls it over HTTP."""
    port = _free_port()
    env = {**os.environ, "JOURNAL_DATA_DIR": str(settings.data_dir), "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.Popen(
        [sys.executable, "-m", "journal_ai.mcp_server", "--transport", "streamable-http", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.3)
        tools = MCPTools(f"http://127.0.0.1:{port}/mcp")
        assert tools.get_policy("research_article", "clinical")["domain"] == "clinical"
        assert tools.register_doi("MS-HTTP", "10.5555")["doi"].startswith("10.5555/jas.")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
