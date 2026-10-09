"""Static figures for the reports (matplotlib, rendered to PNG).

Colours follow one documented palette: a light chart surface painted into the
image (so it stays readable on dark pages), text in ink tones rather than series
colours, one categorical order for model identity, and a single-hue blue ramp
for interval bands.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BAND_INNER, BAND_OUTER = "#86b6ef", "#cde2fb"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
}


def to_png(fig, dpi: int = 130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def goodness_of_fit(diag: pd.DataFrame):
    """DV vs PRED / IPRED, and CWRES vs time / PRED."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), layout="constrained")
        top = float(max(diag["DV"].max(), diag["PRED"].max(), diag["IPRED"].max())) * 1.05
        for ax, col, title in [(axes[0, 0], "PRED", "Observed vs population prediction"),
                               (axes[0, 1], "IPRED", "Observed vs individual prediction")]:
            ax.scatter(diag[col], diag["DV"], s=12, color=SERIES[0], alpha=0.6, linewidths=0)
            ax.plot([0, top], [0, top], color=INK2, linewidth=1.2)
            ax.set(xlim=(0, top), ylim=(0, top), xlabel=f"{col} (mg/L)", ylabel="DV (mg/L)",
                   title=title)
        for ax, col, label, title in [
                (axes[1, 0], "TIME", "time (h)", "CWRES vs time"),
                (axes[1, 1], "PRED", "PRED (mg/L)", "CWRES vs population prediction")]:
            ax.scatter(diag[col], diag["CWRES"], s=12, color=SERIES[0], alpha=0.6, linewidths=0)
            ax.axhline(0, color=INK2, linewidth=1.2)
            for y in (-2, 2):
                ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
            ax.set(xlabel=label, ylabel="CWRES", title=title)
        return fig


def vpc_figure(result, title: str = "Visual predictive check"):
    bins, obs = result.bins, result.observed
    labels = [f"p{p:g}".replace(".", "_") for p in result.percentiles]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
        ax.scatter(obs["TIME"], obs["DV"], s=9, color=MUTED, alpha=0.7, linewidths=0,
                   label="observations", zorder=3)  # data above the bands, below the lines
        t = bins["t_mid"]
        for i, lab in enumerate(labels):
            middle = i == len(labels) // 2
            ax.fill_between(t, bins[f"sim_{lab}_lo"], bins[f"sim_{lab}_hi"],
                            color=BAND_INNER if middle else BAND_OUTER, linewidth=0,
                            label=(f"{result.ci:g}% CI of simulated median" if middle else
                                   f"{result.ci:g}% CI of simulated outer percentiles"
                                   if i == 0 else None))
            ax.plot(t, bins[f"obs_{lab}"], color=INK if middle else INK2,
                    linewidth=2 if middle else 1.5, linestyle="-" if middle else "--",
                    marker="o", markersize=4, zorder=4,
                    label="observed median" if middle else
                    ("observed 5th / 95th percentiles" if i == 0 else None))
        kind = "prediction-corrected " if result.prediction_corrected else ""
        ax.set(xlabel="time (h)", ylabel=f"{kind}concentration (mg/L)",
               title=f"{title}  ·  {result.n_sim} simulated replicates")
        ax.legend(loc="upper right", fontsize=8)
        return fig


def accuracy_by_tier(board: pd.DataFrame):
    tiers = [c for c in board.columns if c.startswith("tier")]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        x = np.arange(1, len(tiers) + 1)
        for i, row in enumerate(board.itertuples(index=False)):
            color = SERIES[i % len(SERIES)]
            ys = [getattr(row, t) for t in tiers]
            ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=7, label=row.model)
            if len(board) <= 4:
                ax.annotate(row.model, (x[-1], ys[-1]), xytext=(8, 0),
                            textcoords="offset points", va="center", color=INK2, fontsize=8)
        ax.set(xticks=x, xticklabels=[f"tier {t[4:]}" for t in tiers], ylim=(0, 1.02),
               xlim=(0.8, len(tiers) + 0.9), ylabel="share correct",
               title="Accuracy by tier (tier 3 needs the most steps)")
        ax.legend(loc="lower left", fontsize=8)
        return fig
