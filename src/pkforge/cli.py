"""Command-line interface: ``pkforge <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pkforge import plots
from pkforge.dosing import Dose, Subject, read_dataset, to_frame
from pkforge.estimation import ErrorModel, Param, PopModel, fit, simulate
from pkforge.estimation.initial import from_nca
from pkforge.estimation.vpc import vpc_from_fit
from pkforge.models import required_parameters
from pkforge.nca import nca_subjects, summarize
from pkforge.report import bench_report, fit_report

DEMO_TRUTH = PopModel(
    n_cmt=1,
    params=(Param("CL", 4.0, 0.09, "WT", 70, 0.75), Param("V", 40.0, 0.04, "WT", 70, 1.0),
            Param("KA", 1.2, 0.16)),
    error=ErrorModel("proportional"), sigma=(0.12,))
DEMO_SAMPLES = [0.5, 1, 2, 3, 4, 6, 8, 12, 24]


def _show(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}") if not df.empty
              else "(empty)")


def demo_study(n: int = 40, seed: int = 1) -> list[Subject]:
    """A simulated single-dose oral study with body-weight effects on CL and V."""
    rng = np.random.default_rng(seed)
    design = [Subject(str(i + 1), [Dose(0.0, 100.0, "oral")], DEMO_SAMPLES,
                      covariates={"WT": round(float(np.clip(rng.normal(72, 14), 40, 120)), 1)})
              for i in range(n)]
    return simulate(DEMO_TRUTH, design, seed=seed)


def _truth_table(model: PopModel) -> dict[str, float]:
    out = {p.name: p.value for p in model.params}
    out.update({f"omega2({p.name})": p.omega for p in model.iiv})
    out.update(dict(zip(model.error.names, model.sigma, strict=True)))
    return out


def build_model(subjects: list[Subject], n_cmt: int, error: str, params: list[str],
                covariates: list[str], no_iiv: list[str], omega: float) -> PopModel:
    oral = any(d.route == "oral" for s in subjects for d in s.doses)
    names = required_parameters(n_cmt, oral)
    given = {}
    for item in params:
        name, _, value = item.partition("=")
        given[name.strip().upper()] = float(value)
    initial = {**(from_nca(subjects, n_cmt) if set(names) - set(given) else {}), **given}
    cov = {}
    for item in covariates:  # PARAM:COVARIATE:REFERENCE:EXPONENT
        p, c, ref, exp = item.split(":")
        cov[p.upper()] = (c.upper(), float(ref), float(exp))
    spec = []
    for name in names:
        c, ref, exp = cov.get(name, (None, 1.0, 0.0))
        spec.append(Param(name, initial[name], None if name in no_iiv else omega, c, ref, exp))
    sigma = {"additive": (0.1 * float(np.nanmedian(np.concatenate([s.dv for s in subjects]))),),
             "proportional": (0.2,)}.get(error)
    if sigma is None:
        sigma = (0.1 * float(np.nanmedian(np.concatenate([s.dv for s in subjects]))), 0.2)
    return PopModel(n_cmt, tuple(spec), ErrorModel(error), sigma)


def run_fit(subjects, model, report: str | None, title: str, truth=None, n_sim: int = 200):
    result = fit(model, subjects)
    print(result.summary())
    vpc_res = vpc_from_fit(result, subjects, n_sim=n_sim, n_bins=8)
    print(f"\nVPC: {100 * vpc_res.coverage():.0f}% of observed percentiles inside the simulated "
          f"95% intervals")
    nca_table = nca_subjects(subjects)
    if report:
        Path(report).write_text(fit_report(result, vpc_res, title, truth=truth,
                                           nca_summary=summarize(nca_table)), encoding="utf-8")
        print(f"report written to {report}")
    return result, vpc_res


# ------------------------------------------------------------------------- commands
def cmd_simulate(args) -> int:
    df = to_frame(demo_study(args.n, args.seed))
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {df['ID'].nunique()} subjects, {int((df['EVID'] == 0).sum())} "
          f"observations (one-compartment oral, CL and V scaled by body weight)")
    return 0


def cmd_nca(args) -> int:
    table = nca_subjects(read_dataset(args.data))
    if args.out:
        table.to_csv(args.out, index=False)
    cols = [c for c in ["ID", "cmax", "tmax", "auc_last", "auc_inf", "half_life", "cl", "vz",
                        "note"] if c in table]
    _show(table[cols])
    print()
    _show(summarize(table))
    return 0


def cmd_fit(args) -> int:
    subjects = read_dataset(args.data)
    model = build_model(subjects, args.cmt, args.error, args.param or [], args.covariate or [],
                        [n.upper() for n in (args.no_iiv or [])], args.omega)
    run_fit(subjects, model, args.report, f"Population fit: {Path(args.data).name}")
    return 0


def cmd_theoph(args) -> int:
    from pkforge.datasets import theoph

    subjects = theoph()
    covariates = ["CL:WT:70:0.75", "V:WT:70:1"] if args.weight else []
    model = build_model(subjects, 1, "combined", [], covariates, [], 0.1)
    run_fit(subjects, model, args.report, "Theophylline: one-compartment oral, FOCE-I")
    return 0


def cmd_bench_generate(args) -> int:
    from pkforge.bench import generate, save

    problems = generate(args.n, args.seed, args.family, args.tier)
    sha = save(problems, args.out)
    print(f"wrote {len(problems)} verified problems to {args.out} (sha256 {sha[:12]})")
    return 0


def cmd_bench_show(args) -> int:
    from pkforge.bench import load

    problems, _ = load(args.problems)
    p = problems[args.index]
    print(f"{p.id}  (tier {p.tier}, {p.family})\n\n{p.full_prompt()}\n")
    print(f"answer: {p.answer:.6g} {p.unit}   (engine check: {p.verified:.6g})")
    for name, value in p.distractors.items():
        print(f"  if {name.replace('_', ' ')}: {value:.6g} {p.unit}")
    return 0


def cmd_bench_run(args) -> int:
    from pkforge.bench import load
    from pkforge.bench.adapters import AdapterConfigError, make_adapter
    from pkforge.bench.runner import run

    problems, sha = load(args.problems)
    status = 0
    for spec in args.model:
        try:
            summary = run(args.db, problems, sha, make_adapter(spec), workers=args.workers,
                          resume_run_id=args.resume)
        except AdapterConfigError as exc:
            print(f"[{spec}] stopped: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"{summary.adapter}: accuracy {summary.accuracy}  (run {summary.run_id}, "
              f"{summary.failed} failed calls)")
    return status


def cmd_bench_report(args) -> int:
    from pkforge.bench import analysis
    from pkforge.bench.runner import load_responses

    df = load_responses(args.db)
    if df.empty:
        print(f"no finished runs in {args.db}", file=sys.stderr)
        return 1
    _show(analysis.leaderboard(df))
    Path(args.out).write_text(bench_report(df), encoding="utf-8")
    print(f"\nreport written to {args.out}")
    return 0


def cmd_demo(args) -> int:
    from pkforge.bench import analysis, generate, save
    from pkforge.bench.adapters import make_adapter
    from pkforge.bench.runner import load_responses, run

    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("== 1. Simulated study: 40 subjects, one oral dose, weight effects on CL and V")
    subjects = demo_study()
    to_frame(subjects).to_csv(out / "study.csv", index=False)
    print("\n== 2. Non-compartmental analysis")
    _show(summarize(nca_subjects(subjects)))
    print("\n== 3. Population fit (FOCE-I), starting from NCA-based estimates")
    model = build_model(subjects, 1, "proportional", [], ["CL:WT:70:0.75", "V:WT:70:1"], [],
                        0.09)
    result, vpc_res = run_fit(subjects, model, str(out / "fit_report.html"),
                              "Simulated study: population PK fit",
                              truth=_truth_table(DEMO_TRUTH))
    (figures / "gof.png").write_bytes(plots.to_png(plots.goodness_of_fit(result.diagnostics)))
    (figures / "vpc.png").write_bytes(plots.to_png(plots.vpc_figure(vpc_res)))

    print("\n== 4. Benchmark: 300 generated problems, three simulated students")
    problems = generate(300, seed=args.seed)
    sha = save(problems, out / "problems.jsonl")
    db = out / "bench.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    for spec in ("student:expert", "student:sloppy", "student:novice"):
        run(db, problems, sha, make_adapter(spec), workers=1, progress=lambda m: None)
    df = load_responses(db)
    board = analysis.leaderboard(df)
    _show(board[["model", "accuracy", "ci_low", "ci_high", "misconception_rate",
                 "unit_error_rate", "tier1", "tier2", "tier3"]])
    (out / "bench_report.html").write_text(bench_report(df), encoding="utf-8")
    (figures / "bench_tiers.png").write_bytes(plots.to_png(plots.accuracy_by_tier(board)))
    print(f"\nreports: {out / 'fit_report.html'}, {out / 'bench_report.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pkforge", description="Pharmacokinetic modelling engine "
                                "and a self-verifying PK benchmark for language models.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="write a simulated study (NONMEM-style CSV)")
    s.add_argument("-o", "--out", default="study.csv")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)

    n = sub.add_parser("nca", help="non-compartmental analysis of every subject")
    n.add_argument("data")
    n.add_argument("-o", "--out")
    n.set_defaults(func=cmd_nca)

    f = sub.add_parser("fit", help="population fit by FOCE-I, with diagnostics and a VPC")
    f.add_argument("data")
    f.add_argument("--cmt", type=int, choices=[1, 2, 3], default=1)
    f.add_argument("--error", choices=["additive", "proportional", "combined"],
                   default="proportional")
    f.add_argument("--param", action="append", metavar="NAME=VALUE",
                   help="initial typical value (default: derived from NCA)")
    f.add_argument("--covariate", action="append", metavar="PARAM:COV:REF:EXP",
                   help="power covariate effect, e.g. CL:WT:70:0.75")
    f.add_argument("--no-iiv", action="append", metavar="NAME",
                   help="no between-subject variability on this parameter")
    f.add_argument("--omega", type=float, default=0.09, help="initial omega^2 (default 0.09)")
    f.add_argument("--report", help="write an HTML report here")
    f.set_defaults(func=cmd_fit)

    t = sub.add_parser("theoph", help="download and fit the classic theophylline data")
    t.add_argument("--weight", action="store_true", help="scale CL and V by body weight")
    t.add_argument("--report", help="write an HTML report here")
    t.set_defaults(func=cmd_theoph)

    b = sub.add_parser("bench", help="the LLM benchmark").add_subparsers(dest="bench_cmd",
                                                                         required=True)
    g = b.add_parser("generate", help="generate verified problems")
    g.add_argument("-o", "--out", default="problems.jsonl")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--family", action="append", help="restrict to these families")
    g.add_argument("--tier", type=int, action="append", choices=[1, 2, 3])
    g.set_defaults(func=cmd_bench_generate)
    sh = b.add_parser("show", help="print one problem with its answer and distractors")
    sh.add_argument("problems")
    sh.add_argument("--index", type=int, default=0)
    sh.set_defaults(func=cmd_bench_show)
    r = b.add_parser("run", help="answer and grade the problems with one or more models")
    r.add_argument("--problems", default="problems.jsonl")
    r.add_argument("--model", action="append", required=True,
                   help="student:<expert|novice|sloppy>, openai:<model>, "
                        "anthropic:<model>[@effort]")
    r.add_argument("--db", default="bench.db")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--resume", metavar="RUN_ID")
    r.set_defaults(func=cmd_bench_run)
    rep = b.add_parser("report", help="leaderboard and HTML report")
    rep.add_argument("--db", default="bench.db")
    rep.add_argument("-o", "--out", default="bench_report.html")
    rep.set_defaults(func=cmd_bench_report)

    d = sub.add_parser("demo", help="offline end-to-end demo of both halves")
    d.add_argument("--out-dir", default="demo_output")
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
