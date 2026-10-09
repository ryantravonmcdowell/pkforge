"""Benchmark statistics: accuracy with intervals, paired tests, misconception profiles."""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import binomtest

LABELS = ("correct", "misconception", "unit_error", "wrong_dimension", "incorrect", "no_answer",
          "adapter_error")


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sound at small n and near 0 or 1)."""
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for adapter, g in df.groupby("adapter"):
        graded = g[g["label"] != "adapter_error"]
        n, k = len(graded), int((graded["label"] == "correct").sum())
        lo, hi = wilson_interval(k, n)
        row = {"model": adapter, "n": n, "accuracy": k / n if n else float("nan"),
               "ci_low": lo, "ci_high": hi}
        for label in ("misconception", "unit_error", "no_answer"):
            row[f"{label}_rate"] = float((graded["label"] == label).mean()) if n else float("nan")
        row["format_ok"] = float((graded["source"] == "answer_line").mean()) if n else float("nan")
        for tier in sorted(df["tier"].unique()):
            t = graded[graded["tier"] == tier]
            row[f"tier{tier}"] = float((t["label"] == "correct").mean()) if len(t) else None
        rows.append(row)
    board = pd.DataFrame(rows)
    return board.sort_values("accuracy", ascending=False, ignore_index=True) if rows else board


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each outcome label per model."""
    table = pd.crosstab(df["adapter"], df["label"], normalize="index")
    return table[[c for c in LABELS if c in table.columns]]


def by_family(df: pd.DataFrame) -> pd.DataFrame:
    graded = df[df["label"] != "adapter_error"].assign(correct=lambda d: d["label"] == "correct")
    return graded.pivot_table(index=["tier", "family"], columns="adapter", values="correct",
                              aggfunc="mean")


def misconceptions(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """How often each model's wrong answers matched each known mistake."""
    hits = df[df["label"] == "misconception"]
    if hits.empty:
        return pd.DataFrame()
    table = pd.crosstab(hits["family"] + ": " + hits["misconception"], hits["adapter"])
    table.index.name, table.columns.name = "misconception", None
    return table.loc[table.sum(axis=1).sort_values(ascending=False).index[:top]]


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Exact McNemar tests of the most accurate model against each other model.

    Only questions both models answered count, and only the discordant ones
    (one right, the other wrong) carry information.
    """
    graded = df[df["label"] != "adapter_error"].assign(correct=lambda d: d["label"] == "correct")
    wide = graded.pivot_table(index="problem_id", columns="adapter", values="correct",
                              aggfunc="first")
    if wide.shape[1] < 2:
        return pd.DataFrame()
    order = wide.mean().sort_values(ascending=False).index.tolist()
    best, rows = order[0], []
    for other in order[1:]:
        pair = wide[[best, other]].dropna().astype(bool)
        b = int((pair[best] & ~pair[other]).sum())
        c = int((~pair[best] & pair[other]).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append({"comparison": f"{best} vs {other}", "questions": len(pair),
                     "only_first_right": b, "only_second_right": c,
                     "accuracy_gap": round((b - c) / len(pair), 3) if len(pair) else None,
                     "p_value": p, "significant": "yes" if p < 0.05 else "no"})
    return pd.DataFrame(rows)
