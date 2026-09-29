import math

from journal_ai.cli import main as cli_main
from journal_ai.config import Settings
from journal_ai.evaluation import accuracy, evaluate, fleiss_kappa, pairwise_ordinal_agreement, prf
from journal_ai.service import JournalService


def test_fleiss_kappa():
    assert fleiss_kappa([["a", "a"], ["b", "b"]]) == 1.0
    assert fleiss_kappa([["a", "b"], ["a", "b"]]) == -1.0
    # Worked example (Fleiss 1971 style): 3 items, 3 raters
    k = fleiss_kappa([["a", "a", "a"], ["a", "a", "b"], ["b", "b", "b"]])
    assert math.isclose(k, 0.55, abs_tol=0.01)
    assert math.isnan(fleiss_kappa([["a"]]))


def test_other_metrics():
    assert pairwise_ordinal_agreement(["accept", "reject"]) == 0.0
    assert math.isclose(pairwise_ordinal_agreement(["accept", "accept", "minor_revision"]), (1 + 2 / 3 + 2 / 3) / 3)
    m = prf({"a", "b", "c"}, {"a", "b", "d"})
    assert (m["precision"], m["recall"], m["tp"], m["fp"], m["fn"]) == (0.667, 0.667, 2, 1, 1)
    assert accuracy({"x": "addressed", "y": "not_addressed"}, {"x": "addressed", "y": "addressed"}) == 0.5


def test_benchmark_on_bundled_gold(settings):
    def factory():
        s = Settings()
        s.__dict__.update(settings.__dict__)
        return JournalService(s, llm=None)

    report = evaluate(factory)
    assert report["n_manuscripts"] == 4
    assert report["decision_accuracy"] == 1.0, report["manuscripts"]
    assert report["issue_detection"]["f1"] == 1.0, report["issue_detection"]
    assert report["concern_resolution_accuracy"] == 1.0
    assert report["efficiency"]["workflow_completion_rate"] == 1.0
    assert report["efficiency"]["total_llm_calls"] == 0  # heuristic backend
    assert report["reviewer_agreement"]["fleiss_kappa"] is not None


def test_cli_demo(settings, monkeypatch, capsys):
    monkeypatch.setenv("JOURNAL_LLM_BACKEND", "heuristic")
    monkeypatch.setenv("JOURNAL_CHECKPOINTER", "memory")
    monkeypatch.setenv("JOURNAL_DATA_DIR", str(settings.data_dir))
    cli_main(["demo", "ml_optimizer"])
    out = capsys.readouterr().out
    assert "final status: published" in out and "SCI-004" in out and "partially_addressed" in out
