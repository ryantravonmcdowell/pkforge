"""Initial estimates for a population fit, derived from non-compartmental analysis."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from pkforge.dosing import Subject
from pkforge.nca import nca_subjects


def _ka_from_tmax(tmax: float, k: float) -> float:
    """Solve tmax = ln(ka/k) / (ka - k) for ka > k (one-compartment oral)."""
    if not (tmax > 0 and k > 0) or tmax >= 1 / k:
        return 3 * k

    def gap(ka):
        return math.log(ka / k) / (ka - k) - tmax

    return float(brentq(gap, k * (1 + 1e-6), k * 1e4))


def from_nca(subjects: list[Subject], n_cmt: int) -> dict[str, float]:
    """Typical-value starting points: geometric means of per-subject NCA results."""
    table = nca_subjects(subjects)
    ok = table[table.get("note", "") == ""] if "note" in table else table
    if ok.empty or "cl" not in ok:
        raise ValueError("NCA initial estimates need subjects with a single dose at t = 0; "
                         "give initial values with --param instead")

    def gmean(col):
        x = ok[col].astype(float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.empty:
            raise ValueError(f"NCA produced no usable {col}")
        return float(np.exp(np.log(x).mean()))

    cl, vz = gmean("cl"), gmean("vz")
    est = {"CL": cl}
    if n_cmt == 1:
        est["V"] = vz
    elif n_cmt == 2:
        est.update(V1=vz / 2, Q=cl, V2=vz / 2)
    else:
        est.update(V1=vz / 3, Q2=cl, V2=vz / 3, Q3=cl / 2, V3=vz / 3)
    if any(d.route == "oral" for s in subjects for d in s.doses):
        est["KA"] = _ka_from_tmax(gmean("tmax"), cl / vz)
    return est
