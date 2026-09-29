import xml.etree.ElementTree as ET

from journal_ai.documents import read_pdf_text


def test_metadata_validation(team, intake):
    ms = intake("cv_segmentation")
    assert team.publication.validate_metadata(ms) == []
    ms.authors[1].affiliation = None
    ms.authors[0].orcid = "123"
    ms.keywords = ["a"]
    issues = team.publication.validate_metadata(ms)
    assert any("affiliation" in i for i in issues)
    assert any("ORCID" in i for i in issues)
    assert "fewer than 3 keywords" in issues


def test_publication_record_and_xml(team, intake):
    ms = intake("cv_segmentation")
    rec = team.publication.publish(ms)
    assert rec.metadata_valid and rec.doi.startswith("10.5555/jas.")
    crossref = ET.fromstring(rec.crossref_xml)
    ns = {"c": "http://www.crossref.org/schema/5.3.1"}
    assert crossref.find(".//c:doi_data/c:doi", ns).text == rec.doi
    assert len(crossref.findall(".//c:person_name", ns)) == 2
    jats = ET.fromstring(rec.jats_xml)
    assert jats.find(".//article-id[@pub-id-type='doi']").text == rec.doi
    assert [t.text for t in jats.findall("./body/sec/title")][:2] == ["1. Introduction", "2. Related Work"]
    assert len(jats.findall(".//ref")) == 13


def test_production_generates_print_ready_pdf(team, intake):
    ms = intake("cv_segmentation")
    rec = team.publication.publish(ms)
    prod = team.production.produce(ms, rec)
    assert prod.print_ready, [c for c in prod.checks if not c.passed]
    assert prod.page_count >= 2
    text = read_pdf_text(prod.pdf_path)
    assert rec.doi in text and "LiteSeg" in text and "Figure 1: LiteSeg architecture overview" in text


def test_production_flags_figure_problems(team, intake):
    ms = intake("cv_segmentation")
    ms.figures = ms.figures[:1]  # Figure 2 referenced but not supplied
    prod = team.production.produce(ms, team.publication.publish(ms))
    failed = {c.name for c in prod.checks if not c.passed}
    assert failed == {"figure_placement"} and not prod.print_ready
