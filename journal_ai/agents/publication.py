"""Agent 3 (Publication) and Agent 4 (Production/Printing)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import Manuscript, ProductionCheck, ProductionResult, PublicationRecord
from .base import AgentContext


# =========================================================================== #
# Agent 3 - Publication
# =========================================================================== #


class PublicationAgent:
    agent_id = "publication"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def validate_metadata(self, ms: Manuscript) -> list[str]:
        issues = []
        if not ms.title.strip():
            issues.append("missing title")
        if not ms.abstract.strip():
            issues.append("missing abstract")
        if not ms.authors:
            issues.append("no authors")
        for a in ms.authors:
            if not a.affiliation:
                issues.append(f"author '{a.name}' has no affiliation")
            if a.orcid and not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", a.orcid):
                issues.append(f"author '{a.name}' has a malformed ORCID")
        if not any(a.corresponding and a.email for a in ms.authors):
            issues.append("no corresponding author with e-mail")
        if len(ms.keywords) < 3:
            issues.append("fewer than 3 keywords")
        return issues

    def publish(self, ms: Manuscript) -> PublicationRecord:
        s = self.ctx.settings
        issues = self.validate_metadata(ms)
        doi = self.ctx.tools.register_doi(ms.manuscript_id, s.doi_prefix)["doi"]
        now = datetime.now(timezone.utc)
        volume, issue = now.year - 2025 + 1, (now.month - 1) // 3 + 1
        return PublicationRecord(
            manuscript_id=ms.manuscript_id,
            doi=doi,
            volume=volume,
            issue=issue,
            published_date=now.date().isoformat(),
            metadata_valid=not issues,
            metadata_issues=issues,
            crossref_xml=self._crossref(ms, doi, now, volume, issue),
            jats_xml=self._jats(ms, doi, now, volume, issue),
        )

    def _crossref(self, ms: Manuscript, doi: str, now: datetime, volume: int, issue: int) -> str:
        s = self.ctx.settings
        root = ET.Element("doi_batch", {"version": "5.3.1", "xmlns": "http://www.crossref.org/schema/5.3.1"})
        head = ET.SubElement(root, "head")
        ET.SubElement(head, "doi_batch_id").text = f"{ms.manuscript_id}-{int(now.timestamp())}"
        ET.SubElement(head, "timestamp").text = now.strftime("%Y%m%d%H%M%S")
        body = ET.SubElement(root, "body")
        journal = ET.SubElement(body, "journal")
        jm = ET.SubElement(journal, "journal_metadata")
        ET.SubElement(jm, "full_title").text = s.journal_name
        ET.SubElement(jm, "issn", {"media_type": "electronic"}).text = s.journal_issn
        ji = ET.SubElement(journal, "journal_issue")
        ET.SubElement(ET.SubElement(ji, "publication_date"), "year").text = str(now.year)
        ET.SubElement(ET.SubElement(ji, "journal_volume"), "volume").text = str(volume)
        ET.SubElement(ji, "issue").text = str(issue)
        art = ET.SubElement(journal, "journal_article", {"publication_type": "full_text"})
        ET.SubElement(ET.SubElement(art, "titles"), "title").text = ms.title
        contribs = ET.SubElement(art, "contributors")
        for i, a in enumerate(ms.authors):
            pn = ET.SubElement(
                contribs, "person_name", {"sequence": "first" if i == 0 else "additional", "contributor_role": "author"}
            )
            parts = a.name.split()
            ET.SubElement(pn, "given_name").text = " ".join(parts[:-1]) or a.name
            ET.SubElement(pn, "surname").text = parts[-1]
            if a.orcid:
                ET.SubElement(pn, "ORCID").text = f"https://orcid.org/{a.orcid}"
        pd = ET.SubElement(art, "publication_date", {"media_type": "online"})
        ET.SubElement(pd, "month").text = f"{now.month:02d}"
        ET.SubElement(pd, "day").text = f"{now.day:02d}"
        ET.SubElement(pd, "year").text = str(now.year)
        dd = ET.SubElement(art, "doi_data")
        ET.SubElement(dd, "doi").text = doi
        ET.SubElement(dd, "resource").text = f"https://journal.example.org/articles/{doi}"
        return ET.tostring(root, encoding="unicode")

    def _jats(self, ms: Manuscript, doi: str, now: datetime, volume: int, issue: int) -> str:
        root = ET.Element("article", {"article-type": ms.article_type.replace("_", "-"), "dtd-version": "1.3"})
        front = ET.SubElement(root, "front")
        jm = ET.SubElement(front, "journal-meta")
        ET.SubElement(ET.SubElement(jm, "journal-title-group"), "journal-title").text = self.ctx.settings.journal_name
        am = ET.SubElement(front, "article-meta")
        ET.SubElement(am, "article-id", {"pub-id-type": "doi"}).text = doi
        ET.SubElement(am, "article-id", {"pub-id-type": "manuscript"}).text = ms.manuscript_id
        ET.SubElement(ET.SubElement(am, "title-group"), "article-title").text = ms.title
        cg = ET.SubElement(am, "contrib-group")
        for a in ms.authors:
            c = ET.SubElement(cg, "contrib", {"contrib-type": "author", **({"corresp": "yes"} if a.corresponding else {})})
            ET.SubElement(ET.SubElement(c, "name"), "string-name").text = a.name
            if a.affiliation:
                ET.SubElement(c, "aff").text = a.affiliation
        ET.SubElement(am, "volume").text = str(volume)
        ET.SubElement(am, "issue").text = str(issue)
        ET.SubElement(ET.SubElement(am, "abstract"), "p").text = ms.abstract
        kg = ET.SubElement(am, "kwd-group")
        for k in ms.keywords:
            ET.SubElement(kg, "kwd").text = k
        body = ET.SubElement(root, "body")
        for sec in ms.sections:
            if sec.heading.lower().startswith(("abstract", "reference")):
                continue
            el = ET.SubElement(body, "sec")
            ET.SubElement(el, "title").text = sec.heading
            for para in [p for p in sec.text.split("\n\n") if p.strip()]:
                ET.SubElement(el, "p").text = para.strip()
        refs = ms.section("reference")
        if refs:
            rl = ET.SubElement(ET.SubElement(root, "back"), "ref-list")
            for i, line in enumerate([ln for ln in refs.text.splitlines() if ln.strip()], 1):
                ET.SubElement(ET.SubElement(rl, "ref", {"id": f"R{i}"}), "mixed-citation").text = line.strip()
        return ET.tostring(root, encoding="unicode")


# =========================================================================== #
# Agent 4 - Production / Printing
# =========================================================================== #


class ProductionAgent:
    agent_id = "production"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def produce(self, ms: Manuscript, pub: PublicationRecord) -> ProductionResult:
        out_dir = self.ctx.settings.data_dir / "pdf"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{ms.manuscript_id}-v{ms.version}.pdf"
        self._typeset(ms, pub, pdf_path)
        info = self.ctx.tools.validate_pdf(str(pdf_path))
        checks = self._checks(ms, pub, pdf_path, info)
        return ProductionResult(
            manuscript_id=ms.manuscript_id,
            pdf_path=str(pdf_path),
            page_count=info.get("page_count", 0),
            checks=checks,
            print_ready=all(c.passed for c in checks),
        )

    def _typeset(self, ms: Manuscript, pub: PublicationRecord, path: Path) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table
        from xml.sax.saxutils import escape

        styles = getSampleStyleSheet()
        s = self.ctx.settings
        doc = SimpleDocTemplate(
            str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
            title=ms.title, author=", ".join(a.name for a in ms.authors), subject=pub.doi,
        )

        def footer(canvas, _doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.drawString(20 * mm, 10 * mm, f"{s.journal_name} {pub.volume}({pub.issue}) - https://doi.org/{pub.doi}")
            canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(canvas.getPageNumber()))
            canvas.restoreState()

        story = [
            Paragraph(escape(s.journal_name), styles["Normal"]),
            Paragraph(escape(ms.title), styles["Title"]),
            Paragraph(escape(", ".join(a.name for a in ms.authors)), styles["Normal"]),
            Paragraph(f"DOI: {escape(pub.doi)} &nbsp; Published: {pub.published_date}", styles["Normal"]),
            Spacer(1, 6 * mm),
            Paragraph("<b>Abstract</b>", styles["Heading2"]),
            Paragraph(escape(ms.abstract), styles["BodyText"]),
            Paragraph("<b>Keywords:</b> " + escape(", ".join(ms.keywords)), styles["BodyText"]),
        ]
        figures_placed = set()
        for sec in ms.sections:
            if sec.heading.lower().startswith("abstract"):
                continue
            story.append(Paragraph(escape(sec.heading), styles["Heading2" if sec.level <= 2 else "Heading3"]))
            for para in [p for p in re.split(r"\n\s*\n", sec.text) if p.strip()]:
                if para.lstrip().startswith("|"):  # markdown table
                    rows = [
                        [c.strip() for c in ln.strip().strip("|").split("|")]
                        for ln in para.strip().splitlines()
                        if ln.strip() and not re.fullmatch(r"\|?[\s:\-|]+\|?", ln.strip())
                    ]
                    story.append(Table(rows, hAlign="LEFT"))
                else:
                    story.append(Paragraph(escape(para.strip()).replace("\n", "<br/>"), styles["BodyText"]))
                for n in re.findall(r"\b(?:Figure|Fig\.)\s?(\d+)", para):
                    n = int(n)
                    if n not in figures_placed and n <= len(ms.figures):
                        figures_placed.add(n)
                        story.append(Paragraph(f"<i>[Figure {n}: {escape(ms.figures[n - 1])}]</i>", styles["Italic"]))
        doc.build(story, onFirstPage=footer, onLaterPages=footer)

    def _checks(self, ms: Manuscript, pub: PublicationRecord, path: Path, info: dict) -> list[ProductionCheck]:
        from ..documents import read_pdf_text

        text = re.sub(r"\s+", " ", read_pdf_text(path)) if info.get("exists") else ""
        checks = [
            ProductionCheck(name="pdf_generated", passed=bool(info.get("exists")) and info.get("page_count", 0) > 0,
                            detail=f"{info.get('page_count', 0)} pages, {info.get('size_bytes', 0)} bytes"),
            ProductionCheck(name="text_extractable", passed=info.get("text_chars", 0) > 200,
                            detail=f"{info.get('text_chars', 0)} characters extractable"),
            ProductionCheck(name="pdf_metadata", passed=bool(info.get("has_metadata_title")), detail="document title set"),
            ProductionCheck(name="layout_uniform_pages", passed=bool(info.get("uniform_page_size")),
                            detail=f"page sizes {info.get('page_sizes')}"),
            ProductionCheck(name="doi_printed", passed=pub.doi in text, detail=pub.doi),
        ]
        missing = [s.heading for s in ms.sections if not s.heading.lower().startswith("abstract")
                   and re.sub(r"\s+", " ", s.heading) not in text]
        checks.append(ProductionCheck(name="all_sections_rendered", passed=not missing,
                                      detail=f"missing: {missing}" if missing else "all section headings present"))
        refs = sorted({int(n) for n in re.findall(r"\b(?:Figure|Fig\.)\s?(\d+)", ms.body)})
        placed_ok = all(n <= len(ms.figures) for n in refs)
        sequential = refs == list(range(1, len(refs) + 1))
        checks.append(ProductionCheck(name="figure_placement", passed=placed_ok and sequential,
                                      detail=f"referenced {refs}, supplied {len(ms.figures)}"))
        tables = sorted({int(n) for n in re.findall(r"\bTable\s?(\d+)", ms.body)})
        checks.append(ProductionCheck(name="table_numbering", passed=tables == list(range(1, len(tables) + 1)),
                                      detail=f"tables referenced {tables}"))
        return checks
