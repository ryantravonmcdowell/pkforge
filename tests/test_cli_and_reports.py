import pandas as pd
import pytest

from pkforge.bench import generate
from pkforge.cli import demo_study, main
from pkforge.estimation.initial import from_nca
from pkforge.report import bench_report


def test_initial_estimates_from_nca_are_close_to_the_truth():
    est = from_nca(demo_study(30, seed=2), n_cmt=1)
    assert est["CL"] == pytest.approx(4.0, rel=0.3)
    assert est["V"] == pytest.approx(40.0, rel=0.3)
    assert est["KA"] == pytest.approx(1.2, rel=0.6)
    two = from_nca(demo_study(10, seed=2), n_cmt=2)
    assert set(two) == {"CL", "V1", "Q", "V2", "KA"}


def test_simulate_then_nca(tmp_path, capsys):
    study = tmp_path / "study.csv"
    assert main(["simulate", "-o", str(study), "--n", "6"]) == 0
    df = pd.read_csv(study)
    assert set(df.columns) >= {"ID", "TIME", "EVID", "AMT", "RATE", "CMT", "DV", "WT"}
    assert main(["nca", str(study), "-o", str(tmp_path / "nca.csv")]) == 0
    assert "geometric_mean" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "nca.csv")) == 6


@pytest.mark.slow
def test_fit_command_writes_a_report(tmp_path, capsys):
    study = tmp_path / "study.csv"
    main(["simulate", "-o", str(study), "--n", "16", "--seed", "3"])
    report = tmp_path / "fit.html"
    assert main(["fit", str(study), "--covariate", "CL:WT:70:0.75", "--covariate",
                 "V:WT:70:1", "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "FOCE-I fit: 16 subjects" in out and "VPC:" in out
    page = report.read_text(encoding="utf-8")
    for section in ("Parameters", "Goodness of fit", "Visual predictive check",
                    "Non-compartmental analysis"):
        assert f"<h2>{section}</h2>" in page
    assert page.count("data:image/png;base64,") == 2


def test_bench_commands(tmp_path, capsys):
    problems, db = tmp_path / "p.jsonl", tmp_path / "b.db"
    assert main(["bench", "generate", "--n", "45", "--seed", "3", "-o", str(problems)]) == 0
    assert main(["bench", "show", str(problems), "--index", "2"]) == 0
    shown = capsys.readouterr().out
    assert "ANSWER: <number> <unit>" in shown and "engine check" in shown
    assert main(["bench", "run", "--problems", str(problems), "--db", str(db), "--model",
                 "student:expert", "--model", "student:novice"]) == 0
    html = tmp_path / "r.html"
    assert main(["bench", "report", "--db", str(db), "-o", str(html)]) == 0
    page = html.read_text(encoding="utf-8")
    assert "<h2>Which misconceptions</h2>" in page and "student:novice" in page


def test_bench_report_escapes_model_output():
    p = generate(1, seed=1, families=["half_life"])[0]
    df = pd.DataFrame([{
        "run_id": "r", "problem_id": p.id, "adapter": "evil", "label": "misconception",
        "misconception": "forgot_ln2", "text": "<script>alert(1)</script>", "family": p.family,
        "tier": p.tier, "answer": p.answer, "unit": p.unit, "source": "answer_line"}])
    page = bench_report(df)
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_errors_are_reported_cleanly(tmp_path, capsys):
    assert main(["nca", str(tmp_path / "missing.csv")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["bench", "generate", "--family", "astrology", "-o",
                 str(tmp_path / "x.jsonl")]) == 2
