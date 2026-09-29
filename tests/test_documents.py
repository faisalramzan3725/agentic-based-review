from journal_ai import samples
from journal_ai.documents import (
    attach_sections,
    extract_title_and_abstract,
    load_manuscript_text,
    plain_text_to_markdown,
    render_for_prompt,
    split_sections,
    to_documents,
)
from journal_ai.schemas import Manuscript


def test_split_sections_keeps_deepest_heading():
    md = "# Title\n\n## Abstract\nShort.\n\n## 1. Introduction\nIntro text.\n\n### 1.1 Scope\nScope text.\n"
    secs = split_sections(md)
    assert [s.heading for s in secs] == ["Abstract", "1. Introduction", "1.1 Scope"]
    assert secs[2].level == 3 and secs[2].text == "Scope text."


def test_plain_text_headings_are_promoted():
    text = "Abstract\nWe study x.\n1. Introduction\nBody.\n2.1 Methods\nMore.\nReferences\n[1] A."
    md = plain_text_to_markdown(text)
    assert "## Abstract" in md and "## 1. Introduction" in md and "### 2.1 Methods" in md and "## References" in md


def test_sample_sections_and_abstract():
    sub = samples.load_submission("cv_segmentation")
    assert sub.abstract.startswith("Accurate segmentation")
    ms = attach_sections(Manuscript(**sub.model_dump(), manuscript_id="MS-X"))
    heads = [s.heading for s in ms.sections]
    assert "3. Methods" in heads and "References" in heads
    title, _ = extract_title_and_abstract(sub.body)
    assert title.startswith("LiteSeg")


def test_langchain_documents_carry_section_metadata():
    sub = samples.load_submission("ml_optimizer")
    ms = attach_sections(Manuscript(**sub.model_dump(), manuscript_id="MS-Y"))
    docs = to_documents(ms, chunk_size=300)
    assert len(docs) > len(ms.sections)  # long sections were chunked
    assert {d.metadata["section"] for d in docs} >= {"3. Methods", "4. Results"}
    assert all(d.metadata["manuscript_id"] == "MS-Y" for d in docs)
    assert "## 4. Results" in render_for_prompt(ms)


def test_pdf_manuscript_roundtrip(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "paper.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    y = 800
    for line in ["A PDF Paper", "Abstract", "We study PDFs.", "1. Introduction", "Intro text here.",
                 "2. Methods", "We did things.", "References", "[1] Someone. 2024."]:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    md = load_manuscript_text(pdf)
    heads = [s.heading for s in split_sections(md) if s.text]
    assert heads == ["Abstract", "1. Introduction", "2. Methods", "References"]
    title, abstract = extract_title_and_abstract(md)
    assert (title, abstract) == ("A PDF Paper", "We study PDFs.")
