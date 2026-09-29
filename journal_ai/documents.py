"""Document processing: Markdown/PDF manuscripts -> sections -> LangChain Documents.

LangChain's text splitters do the heavy lifting; this module adds the
journal-specific bits (heading normalisation for PDFs, section objects,
chunking for retrieval-style prompts).
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .schemas import Manuscript, Section

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Plain-text headings typical of PDFs: "1. Introduction", "2 Methods", "Abstract".
_KNOWN_HEADINGS = (
    "abstract|introduction|background|related work|literature review|method|methods|methodology|"
    "materials and methods|experimental setup|experiments|results|discussion|limitations|"
    "conclusion|conclusions|references|acknowledg(e)?ments|ethics statement|ethical approval|"
    "data availability|conflict of interest|conflicts of interest|competing interests|funding|"
    "author contributions|appendix"
)
_PLAIN_HEADING = re.compile(
    rf"^\s*(?:(\d+(?:\.\d+)*)\.?\s+)?({_KNOWN_HEADINGS})(?:\s+statement)?\s*:?\s*$",
    re.IGNORECASE,
)


def split_sections(markdown: str) -> list[Section]:
    """Split a Markdown manuscript into sections, keeping the deepest heading as the name."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    docs = splitter.split_text(markdown)
    sections: list[Section] = []
    for doc in docs:
        heading, level = "Preamble", 0
        for lvl, key in ((3, "h3"), (2, "h2"), (1, "h1")):
            if key in doc.metadata:
                heading, level = doc.metadata[key], lvl
                break
        text = doc.page_content.strip()
        if sections and sections[-1].heading == heading:
            sections[-1] = Section(heading=heading, level=level, text=f"{sections[-1].text}\n{text}")
        else:
            sections.append(Section(heading=heading.strip(), level=level, text=text))
    return sections


def plain_text_to_markdown(text: str) -> str:
    """Promote recognisable plain-text headings (as extracted from PDFs) to Markdown."""
    out: list[str] = []
    lines = text.splitlines()
    # the first non-empty line of an extracted paper is its title
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is not None and not _PLAIN_HEADING.match(lines[first]):
        lines[first] = f"# {lines[first].strip()}"
    for line in lines:
        m = _PLAIN_HEADING.match(line)
        if m and len(line.strip()) < 60:
            depth = "###" if m.group(1) and "." in m.group(1) else "##"
            out.append(f"{depth} {line.strip().rstrip(':')}")
        else:
            out.append(line)
    return "\n".join(out)


def read_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_manuscript_text(path: str | Path) -> str:
    """Read a .md/.txt/.pdf file and return Markdown-ish text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return plain_text_to_markdown(read_pdf_text(path))
    text = path.read_text(encoding="utf-8")
    return text if re.search(r"^#{1,3} ", text, re.M) else plain_text_to_markdown(text)


def extract_title_and_abstract(markdown: str) -> tuple[str | None, str | None]:
    """Best-effort title/abstract extraction from a full manuscript file."""
    title = None
    m = re.search(r"^#\s+(.+)$", markdown, re.M)
    if m:
        title = m.group(1).strip()
    abstract = None
    for sec in split_sections(markdown):
        if sec.heading.lower().startswith("abstract"):
            abstract = sec.text
            break
    return title, abstract


def attach_sections(manuscript: Manuscript) -> Manuscript:
    """Populate ``manuscript.sections`` from its body (and abstract field)."""
    body = manuscript.body
    if not re.search(r"^#{1,3} ", body, re.M):
        body = plain_text_to_markdown(body)
    sections = split_sections(body)
    if manuscript.abstract and not any(s.heading.lower().startswith("abstract") for s in sections):
        sections.insert(0, Section(heading="Abstract", level=2, text=manuscript.abstract))
    manuscript.sections = [s for s in sections if s.text or s.heading != "Preamble"]
    return manuscript


def to_documents(manuscript: Manuscript, chunk_size: int = 1500) -> list[Document]:
    """Section-aware chunks with metadata, for retrieval-augmented prompting."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)
    docs: list[Document] = []
    for sec in manuscript.sections:
        for i, chunk in enumerate(splitter.split_text(sec.text) or [""]):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "manuscript_id": manuscript.manuscript_id,
                        "version": manuscript.version,
                        "section": sec.heading,
                        "chunk": i,
                    },
                )
            )
    return docs


def render_for_prompt(manuscript: Manuscript, max_chars: int = 180_000) -> str:
    """Whole manuscript as a sectioned string for an LLM prompt.

    Opus-class models have a 1M-token window, so real manuscripts fit; the cap
    exists only as a guard against pathological uploads and is reported, not silent.
    """
    parts = [f"TITLE: {manuscript.title}", f"KEYWORDS: {', '.join(manuscript.keywords)}"]
    for sec in manuscript.sections:
        parts.append(f"\n## {sec.heading}\n{sec.text}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        raise ValueError(
            f"Manuscript {manuscript.manuscript_id} is {len(text)} chars, above the {max_chars} prompt cap"
        )
    return text
