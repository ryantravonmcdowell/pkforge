"""Self-contained HTML reports: a population PK fit, and a benchmark run."""

from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pkforge import plots
from pkforge.bench import analysis

CSS = """
:root { --fg:#1d232b; --muted:#5b6673; --bg:#ffffff; --panel:#f5f7fa; --line:#dde3ea; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6ebf0; --muted:#9aa6b2; --bg:#14181d; --panel:#1c2229; --line:#2c343d; } }
* { box-sizing: border-box; }
body { margin:0; padding:32px 16px 64px; background:var(--bg); color:var(--fg);
       font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
.lead, .muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 8px 0; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 5px 10px; text-align: right; }
th { background: var(--panel); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); }
details { background: var(--panel); border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
pre { white-space: pre-wrap; font: 13px/1.45 ui-monospace, Consolas, monospace; }
"""


def _table(df: pd.DataFrame, index: bool = False, digits: int = 4) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'

    def number(x):
        if isinstance(x, (float, np.floating)):
            return "-" if not np.isfinite(x) else f"{x:.{digits}g}"
        return x

    shown = df.map(number) if hasattr(df, "map") else df.applymap(number)
    return f'<div class="scroll">{shown.to_html(index=index, border=0, escape=True)}</div>'


def _page(title: str, lead: str, sections: list[tuple[str, str]]) -> str:
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    body = "".join(f"<h2>{html.escape(t)}</h2>{c}" for t, c in sections)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p class=\"lead\">{lead} &middot; generated {stamp}"
            f"</p>{body}</main></body></html>")


def fit_report(fit, vpc_result=None, title: str = "Population PK fit",
               truth: dict | None = None, nca_summary: pd.DataFrame | None = None) -> str:
    table = fit.parameter_table()
    if truth:
        table.insert(3, "true_value", [truth.get(p, np.nan) for p in table["parameter"]])
    sections = [
        ("Estimation", f"<pre>{html.escape(fit.summary().split(chr(10) + chr(10))[0])}</pre>"),
        ("Parameters", _table(table) + '<p class="muted">RSE is the relative standard error '
         "from the covariance step. CV is the between-subject coefficient of variation implied "
         "by omega&sup2; (log-normal). Shrinkage above ~30% makes individual estimates "
         "unreliable for diagnostics.</p>"),
        ("Goodness of fit", f'<img alt="goodness-of-fit plots" '
         f'src="{plots.data_uri(plots.to_png(plots.goodness_of_fit(fit.diagnostics)))}">'
         '<p class="muted">Points should scatter evenly around the identity line, and CWRES '
         "should centre on zero with most values within &plusmn;2 and no trend.</p>"),
    ]
    if vpc_result is not None:
        sections.append(("Visual predictive check",
                         f'<img alt="visual predictive check" src="'
                         f'{plots.data_uri(plots.to_png(plots.vpc_figure(vpc_result)))}">'
                         f'<p class="muted">{100 * vpc_result.coverage():.0f}% of observed '
                         "percentiles fall inside their simulated confidence interval.</p>"))
    if nca_summary is not None and not nca_summary.empty:
        sections.append(("Non-compartmental analysis", _table(nca_summary)))
    lead = (f"FOCE-I &middot; {fit.n_subjects} subjects &middot; {fit.n_obs} observations "
            f"&middot; OFV {fit.ofv:.2f}")
    return _page(title, lead, sections)


def bench_report(df: pd.DataFrame, title: str = "PK reasoning benchmark") -> str:
    board = analysis.leaderboard(df)
    sections = [
        ("Leaderboard", _table(board, digits=3) + '<p class="muted">Accuracy with a 95% Wilson '
         "interval. <em>misconception_rate</em> is the share of answers that matched a known "
         "wrong method exactly; <em>format_ok</em> is the share that used the requested "
         "ANSWER line.</p>"),
        ("Accuracy by tier", f'<img alt="accuracy by tier" '
         f'src="{plots.data_uri(plots.to_png(plots.accuracy_by_tier(board)))}">'),
        ("Is the gap real?", _table(analysis.paired_tests(df), digits=3) +
         '<p class="muted">Exact McNemar tests on the questions both models answered; only '
         "questions one got right and the other wrong carry information.</p>"),
        ("How answers went wrong", _table(analysis.outcome_table(df), index=True, digits=3)),
        ("Which misconceptions", _table(analysis.misconceptions(df), index=True) +
         '<p class="muted">Each count is a wrong answer that equals, within tolerance, what '
         "that specific mistake produces for that question.</p>"),
        ("Accuracy by problem family", _table(analysis.by_family(df), index=True, digits=3)),
    ]
    wrong = df[df["label"].isin(["misconception", "unit_error"])].head(8)
    if not wrong.empty:
        cards = "".join(
            f"<details><summary>{html.escape(r.problem_id)} &middot; {html.escape(r.adapter)}"
            f" &middot; {html.escape(r.label)}"
            f"{': ' + html.escape(str(r.misconception)) if r.misconception else ''}</summary>"
            f"<pre>{html.escape(str(r.text))}</pre><p class=\"muted\">correct answer "
            f"{r.answer:.4g} {html.escape(r.unit)}</p></details>"
            for r in wrong.itertuples(index=False))
        sections.append(("Diagnosed answers", cards))
    lead = (f"{df['problem_id'].nunique()} problems &middot; {df['adapter'].nunique()} models "
            f"&middot; {len(df)} graded answers")
    return _page(title, lead, sections)
