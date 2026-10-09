"""Regenerate the figures in docs/images used by the README.

    python scripts/make_figures.py

Runs the simulated-study fit, the theophylline fit (downloads the data once)
and the simulated-student benchmark, and writes PNGs. Takes about a minute.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pkforge import plots
from pkforge.bench import analysis, generate, save
from pkforge.bench.adapters import make_adapter
from pkforge.bench.runner import load_responses, run
from pkforge.cli import build_model, demo_study
from pkforge.datasets import theoph
from pkforge.estimation import fit
from pkforge.estimation.vpc import vpc_from_fit

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    study = demo_study()
    model = build_model(study, 1, "proportional", [], ["CL:WT:70:0.75", "V:WT:70:1"], [], 0.09)
    result = fit(model, study)
    (OUT / "gof.png").write_bytes(plots.to_png(plots.goodness_of_fit(result.diagnostics)))
    (OUT / "vpc.png").write_bytes(plots.to_png(plots.vpc_figure(
        vpc_from_fit(result, study, n_sim=500), "Simulated study: visual predictive check")))

    subjects = theoph()
    model = build_model(subjects, 1, "combined", [], ["CL:WT:70:0.75", "V:WT:70:1"], [], 0.1)
    result = fit(model, subjects)
    (OUT / "theoph_vpc.png").write_bytes(plots.to_png(plots.vpc_figure(
        vpc_from_fit(result, subjects, n_sim=500, n_bins=8, prediction_corrected=True),
        "Theophylline: prediction-corrected VPC")))

    problems = generate(300, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        sha = save(problems, Path(tmp) / "p.jsonl")
        db = Path(tmp) / "b.db"
        for spec in ("student:expert", "student:sloppy", "student:novice"):
            run(db, problems, sha, make_adapter(spec), workers=1, progress=lambda m: None)
        board = analysis.leaderboard(load_responses(db))
    (OUT / "bench_tiers.png").write_bytes(plots.to_png(plots.accuracy_by_tier(board)))
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
