"""Tool gateway used by agents.

Agents depend on the ``JournalTools`` interface only. ``LocalTools`` calls the
implementations in-process; ``MCPTools`` sends the same calls through the MCP
server (``journal_ai.mcp_server``), so tools can run as a separate service in
Kubernetes without any agent code changing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import literature, policy
from .store import ManuscriptStore
from .text import TfIdfIndex


class JournalTools:
    """Interface. Every method returns JSON-serialisable data."""

    name = "abstract"

    def search_literature(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...
    def check_duplicate(self, manuscript_id: str, text: str, threshold: float = 0.8) -> list[dict[str, Any]]: ...
    def get_policy(self, article_type: str = "research_article", domain: str | None = None) -> dict[str, Any]: ...
    def journal_scope(self) -> dict[str, list[str]]: ...
    def register_doi(self, manuscript_id: str, prefix: str) -> dict[str, Any]: ...
    def validate_pdf(self, path: str) -> dict[str, Any]: ...
    def log_event(self, manuscript_id: str, agent: str, event: str, payload: dict[str, Any] | None = None) -> None: ...


def validate_pdf_file(path: str) -> dict[str, Any]:
    from pypdf import PdfReader

    p = Path(path)
    if not p.exists():
        return {"exists": False, "page_count": 0, "text_chars": 0, "has_metadata_title": False}
    reader = PdfReader(str(p))
    text = "".join((pg.extract_text() or "") for pg in reader.pages)
    meta = reader.metadata or {}
    boxes = {(round(float(pg.mediabox.width)), round(float(pg.mediabox.height))) for pg in reader.pages}
    return {
        "exists": True,
        "page_count": len(reader.pages),
        "text_chars": len(text),
        "has_metadata_title": bool(meta.get("/Title")),
        "uniform_page_size": len(boxes) == 1,
        "page_sizes": sorted(boxes),
        "size_bytes": p.stat().st_size,
    }


class LocalTools(JournalTools):
    name = "local"

    def __init__(self, store: ManuscriptStore):
        self.store = store

    def search_literature(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return literature.search_literature(query, k)

    def check_duplicate(self, manuscript_id: str, text: str, threshold: float = 0.8) -> list[dict[str, Any]]:
        others = self.store.all_latest(exclude=manuscript_id)
        if not others:
            return []
        index = TfIdfIndex({m.manuscript_id: m.full_text for m in others})
        titles = {m.manuscript_id: m.title for m in others}
        return [
            {"manuscript_id": mid, "title": titles[mid], "similarity": round(s, 4)}
            for mid, s in index.query(text, k=3)
            if s >= threshold
        ]

    def get_policy(self, article_type: str = "research_article", domain: str | None = None) -> dict[str, Any]:
        return policy.get_policy(article_type, domain)

    def journal_scope(self) -> dict[str, list[str]]:
        return policy.journal_scope()

    def register_doi(self, manuscript_id: str, prefix: str) -> dict[str, Any]:
        year = datetime.now(timezone.utc).year
        seq = self.store.next_sequence(f"doi-{year}")
        return {"doi": f"{prefix}/jas.{year}.{seq:05d}", "sequence": seq, "year": year}

    def validate_pdf(self, path: str) -> dict[str, Any]:
        return validate_pdf_file(path)

    def log_event(self, manuscript_id: str, agent: str, event: str, payload: dict[str, Any] | None = None) -> None:
        self.store.log_event(manuscript_id, agent, event, payload)


class MCPTools(JournalTools):
    """Same interface, executed on an MCP server (URL, or an in-process server for tests)."""

    name = "mcp"

    def __init__(self, server: Any):
        self.server = server  # URL string or mcp.server.mcpserver.MCPServer instance
        self.calls: list[str] = []

    async def _acall(self, tool: str, args: dict[str, Any]) -> Any:
        from mcp import Client

        async with Client(self.server) as client:
            result = await client.call_tool(tool, args)
        if result.is_error:
            raise RuntimeError(f"MCP tool {tool} failed: {result.content}")
        structured = result.structured_content
        if structured is not None:
            # MCP wraps non-object return values as {"result": ...}
            if isinstance(structured, dict) and set(structured) == {"result"}:
                return structured["result"]
            return structured
        text = "".join(getattr(c, "text", "") for c in result.content)
        return json.loads(text) if text else None

    def _call(self, tool: str, **args: Any) -> Any:
        self.calls.append(tool)
        return asyncio.run(self._acall(tool, {k: v for k, v in args.items() if v is not None}))

    def search_literature(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self._call("search_literature", query=query, k=k)

    def check_duplicate(self, manuscript_id: str, text: str, threshold: float = 0.8) -> list[dict[str, Any]]:
        return self._call("check_duplicate", manuscript_id=manuscript_id, text=text, threshold=threshold)

    def get_policy(self, article_type: str = "research_article", domain: str | None = None) -> dict[str, Any]:
        return self._call("get_policy", article_type=article_type, domain=domain)

    def journal_scope(self) -> dict[str, list[str]]:
        return self._call("journal_scope")

    def register_doi(self, manuscript_id: str, prefix: str) -> dict[str, Any]:
        return self._call("register_doi", manuscript_id=manuscript_id, prefix=prefix)

    def validate_pdf(self, path: str) -> dict[str, Any]:
        return self._call("validate_pdf", path=path)

    def log_event(self, manuscript_id: str, agent: str, event: str, payload: dict[str, Any] | None = None) -> None:
        self._call("log_event", manuscript_id=manuscript_id, agent=agent, event=event, payload=payload or {})


__all__ = ["JournalTools", "LocalTools", "MCPTools", "ManuscriptStore", "validate_pdf_file"]
