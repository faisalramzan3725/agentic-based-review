# Agentic AI Framework for End-to-End Scientific Journal Management

A multi-agent system that runs the full manuscript lifecycle: submission, screening, capability-based reviewer
assignment, parallel specialised review, aggregation, revision with **per-concern resolution tracking**,
editorial decision (human-in-the-loop), publication (DOI, Crossref/JATS) and production (print-ready PDF).

| Proposal component | Implementation |
|---|---|
| LangGraph orchestrator | `journal_ai/graph.py`: stateful graph with `Send` fan-out for parallel reviewers, the cyclic revision loop, `interrupt()` for author revision and editor approval, and SQLite/Postgres checkpointing |
| LangChain | `journal_ai/documents.py`: Markdown/PDF parsing with LangChain text splitters, section-aware `Document` chunks |
| LLM | `journal_ai/llm.py`: Claude (`claude-opus-5-5`) via the Anthropic SDK with structured outputs (Pydantic), prompt caching of the manuscript across reviewers, and refusal fallbacks |
| MCP tool layer | `journal_ai/mcp_server.py`: 10 tools (literature search, policy DB, duplicate check, manuscript DB, DOI registry, PDF validation, audit log). Agents switch to it with `JOURNAL_TOOL_TRANSPORT=mcp` |
| n8n | `n8n/*.json`: intake webhook, then the API; lifecycle notifications (editor/author e-mails); daily deadline monitor |
| Kubernetes | `k8s/`: API with an MCP sidecar, PostgreSQL checkpoints, n8n, Ingress (`kubectl apply -k k8s`) |
| REST API | `journal_ai/api.py` (FastAPI) |
| Evaluation | `journal_ai/evaluation.py`: Fleiss' kappa, detection P/R/F1, concern-resolution accuracy, decision accuracy, efficiency (time, LLM calls, tokens) |

## The 10 agents

| # | Agent | Role |
|---|---|---|
| 1 | Editorial Manager | intake, screening (validity, sections, references, metadata, language, duplicates, scope, declarations), domain classification, capability-based assignment, review aggregation |
| 2 | Editorial Decision | rule-based, auditable recommendation; the LLM only drafts the letter, so it can never accept a paper with an unresolved critical concern |
| 3 | Publication | metadata validation, DOI registration, Crossref 5.3.1 and JATS XML |
| 4 | Production | typesets the PDF (reportlab), then validates pages, text, metadata, DOI, sections, figure placement and table numbering |
| 5–10 | Structural, Methodology, Novelty, Scientific Quality, Language, Compliance reviewers | each has a capability profile, an LLM brief, and deterministic checks (`STR-`, `METH-`, `NOV-`, `SCI-`, `LANG-`, `COMP-` concern ids) |

## Lifecycle

```
intake → screening ─fail→ desk_reject
            │
          assign ──Send×N──> review (parallel) → aggregate → decide → editor_gate (interrupt)
                                                     ^                   │ accept → publication → production
   verify (parallel, per reviewer) <──Send×N── author_revision <─────────┤ revision
                                                (interrupt)              │ reject → rejected
```

After a revision, reviewers do not re-review from scratch. Each open concern gets `addressed`,
`partially_addressed` or `not_addressed`, with the author's response and evidence from the revised text. The
decision after a revision is based on this concern-resolution matrix (`GET /manuscripts/{id}/concerns`).

## Two backends

* **Claude** (`JOURNAL_LLM_BACKEND=claude`, or `auto` when `ANTHROPIC_API_KEY` or an `ant auth login` profile is present):
  reviewers, verifiers and decision letters are LLM calls with schema-validated structured output.
* **Heuristic** (`heuristic`, or `auto` without credentials): each reviewer runs its deterministic checks. The whole
  system works offline, the tests are reproducible, and it serves as a transparent baseline.

## Run

```bash
pip install -r requirements-dev.txt
python -m pytest                                  # 70 tests (+1 live test, skipped by default)
python -m journal_ai.cli demo ml_optimizer        # 3-round lifecycle to publication
python -m journal_ai.cli evaluate                 # research metrics over data/eval/gold.json
uvicorn journal_ai.api:app --port 8000            # REST API, docs at /docs
python -m journal_ai.mcp_server --port 8765       # MCP tool server (streamable HTTP at /mcp)
docker compose up --build                         # API + MCP + Postgres + n8n
kubectl apply -k k8s                              # after creating the journal-secrets secret (see k8s/base.yaml)
```

Main API calls: `POST /manuscripts` (JSON) or `POST /manuscripts/upload` (PDF/MD), `GET /manuscripts/{id}`,
`POST /manuscripts/{id}/editor-decision`, `POST /manuscripts/{id}/revision`, `GET /manuscripts/{id}/pdf`,
`/audit`, `/concerns`, `/metadata/jats|crossref`. Set `JOURNAL_API_KEY` to require `X-API-Key` on write calls.

### Using it through MCP (Claude Code / Claude Desktop)

`.mcp.json` registers the server as a stdio MCP server named `journal-ai`. Claude Code picks it up when you open a
session in this folder. For Claude Desktop, copy the same block into `claude_desktop_config.json`. The server offers
21 tools:

* **Workflow:** `review_manuscript` / `review_manuscript_file` (quick multi-agent review, nothing stored),
  `submit_manuscript` / `submit_manuscript_file` / `submit_sample_manuscript`, `get_manuscript_status`,
  `editor_decision`, `submit_revision`, `withdraw_manuscript`, `get_concern_matrix`.
* **Service:** `search_literature`, `check_duplicate`, `get_policy`, `journal_scope`, `get_manuscript`,
  `list_manuscripts`, `register_doi`, `validate_pdf`, `log_event`, `get_audit_trail`.

Live Claude test: `JOURNAL_LIVE_TESTS=1 ANTHROPIC_API_KEY=... python -m pytest -m live`.

## Important limitations (read before reporting results)

* **The bundled benchmark is a pipeline check, not research evidence.** The same person wrote the sample
  manuscripts, their gold annotations and the heuristic rules, so the 1.0 scores only show that the machinery is
  correct. Real experiments need the Claude backend and independently annotated data (for example OpenReview
  papers with reviews, revisions and author responses) in the `data/eval/gold.json` format.
* Low reviewer agreement (negative kappa on the samples) is expected: specialists only judge their own area. That
  is research question D in the proposal, not a bug.
* The Claude code path was tested against a fake SDK client. It has not run against the live API here (no key
  was available).
* The Docker image, docker-compose stack, Kubernetes manifests and n8n workflows were validated statically (YAML
  and JSON structure, wiring, config keys), but not deployed: Docker, kubectl and n8n are not installed on this
  machine. The n8n e-mail nodes need SMTP credentials after import.
* The manuscript store is SQLite on a volume, so the API runs as one replica (the MCP sidecar shares the volume).
  LangGraph checkpoints already go to Postgres. Moving the store to Postgres is the step needed to scale out.
* Automated peer review is decision support. Many publishers restrict AI-generated reviews, so the human editor
  gate is on by default.
