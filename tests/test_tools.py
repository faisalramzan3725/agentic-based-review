from journal_ai.schemas import Manuscript
from journal_ai.tools import LocalTools, ManuscriptStore
from journal_ai.tools.literature import search_local
from journal_ai.tools.policy import get_policy
from journal_ai.tools.text import TfIdfIndex, jaccard


def _ms(mid, text, version=1):
    return Manuscript(manuscript_id=mid, version=version, title=f"T {mid}", body=text)


def test_store_ids_versions_status_events(tmp_path):
    s = ManuscriptStore(tmp_path / "db.sqlite")
    a, b = s.next_manuscript_id(2026), s.next_manuscript_id(2026)
    assert (a, b) == ("MS-2026-00001", "MS-2026-00002")
    s.save_version(_ms(a, "v1 text"))
    s.save_version(_ms(a, "v2 text", version=2))
    assert s.get(a).version == 2 and s.get(a, 1).body == "v1 text"
    s.set_status(a, "under_review")
    assert s.get_status(a) == "under_review"
    s.log_event(a, "agent", "evt", {"k": 1})
    assert s.events(a)[0]["payload"] == {"k": 1}
    assert s.list_manuscripts()[0]["status"] == "under_review"
    # persistence across connections
    assert ManuscriptStore(tmp_path / "db.sqlite").get(a).version == 2


def test_literature_search_ranks_relevant_paper_first():
    hits = search_local("lightweight attention for medical image segmentation", k=3)
    assert hits[0]["id"] == "L006"
    assert hits[0]["similarity"] > hits[1]["similarity"]


def test_policy_layers_domain_requirements():
    base = get_policy("research_article", None)
    clinical = get_policy("research_article", "clinical")
    assert "ethics" not in base["required_declarations"]
    assert {"ethics", "informed consent", "trial registration", "data availability"} <= set(clinical["required_declarations"])
    assert get_policy("short_communication")["max_words"] == 4000


def test_duplicate_detection_and_doi(tmp_path):
    store = ManuscriptStore(tmp_path / "db.sqlite")
    tools = LocalTools(store)
    text = "graph neural networks for protein folding with attention and residual blocks " * 5
    store.save_version(_ms("MS-A", text))
    assert tools.check_duplicate("MS-B", "T MS-A " + text, threshold=0.9)[0]["manuscript_id"] == "MS-A"
    assert tools.check_duplicate("MS-B", "completely unrelated clinical hypertension trial", 0.5) == []
    d1, d2 = tools.register_doi("MS-A", "10.5555"), tools.register_doi("MS-B", "10.5555")
    assert d1["doi"].startswith("10.5555/jas.") and d2["sequence"] == d1["sequence"] + 1


def test_text_utils():
    idx = TfIdfIndex({"a": "cats purr softly", "b": "dogs bark loudly"})
    assert idx.query("purring cats", 1)[0][0] == "a"
    assert jaccard("alpha beta gamma", "alpha beta delta") == 0.5
