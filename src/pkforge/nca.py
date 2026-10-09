"""Non-compartmental analysis (NCA).

Implements the standard single-dose parameters with the conventions used by
regulatory-grade tools:

* AUC by the linear-up / log-down trapezoidal rule (linear rule also available).
  The log rule is exact for exponential decline, so a monoexponential profile
  gives AUC exactly.
* The terminal slope lambda_z by log-linear regression on the last k >= 3
  positive concentrations, choosing k by adjusted R-squared: the best value,
  preferring more points among fits within ``r2_tolerance`` of the best.
  Cmax is excluded from the fit after extravascular dosing.
* IV bolus: an observation at t = 0 is dropped and C0 is back-extrapolated from
  the first two concentrations, since a sample at the dose time is pre-dose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pkforge.dosing import Subject

METHODS = ("linear-up/log-down", "linear")


@dataclass(frozen=True)
class NCAResult:
    route: str
    dose: float
    cmax: float
    tmax: float
    clast: float
    tlast: float
    auc_last: float
    auc_inf: float
    auc_pct_extrapolated: float
    aumc_inf: float
    lambda_z: float
    half_life: float
    r2_adjusted: float
    n_lambda_points: int
    mrt: float
    cl: float  # CL, or CL/F after oral dosing
    vz: float  # Vz, or Vz/F after oral dosing
    vss: float  # IV only

    @property
    def apparent(self) -> bool:
        """True when CL and Vz are divided by an unknown bioavailability."""
        return self.route == "oral"

    def as_dict(self) -> dict:
        return asdict(self)


def _segments(t: np.ndarray, c: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval AUC and AUMC."""
    t1, t2, c1, c2 = t[:-1], t[1:], c[:-1], c[1:]
    dt = t2 - t1
    auc = (c1 + c2) / 2 * dt
    aumc = (t1 * c1 + t2 * c2) / 2 * dt
    if method == "linear-up/log-down":
        log = (c2 < c1) & (c2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.log(c1 / c2) / dt
            auc_log = (c1 - c2) / k
            aumc_log = (t1 * c1 - t2 * c2) / k + (c1 - c2) / k**2
        auc = np.where(log, auc_log, auc)
        aumc = np.where(log, aumc_log, aumc)
    return auc, aumc


def auc(times, conc, method: str = "linear-up/log-down") -> float:
    """Area under the concentration curve between the first and last sample."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t, c = np.asarray(times, float), np.asarray(conc, float)
    return float(_segments(t, c, method)[0].sum())


def lambda_z(times, conc, first_index: int = 0, min_points: int = 3,
             r2_tolerance: float = 1e-4) -> tuple[float, float, int]:
    """Terminal rate constant, adjusted R-squared and number of points used.

    Candidate fits use the last k positive concentrations at or after
    ``first_index``. Returns NaNs (and 0 points) when no fit has a negative slope.
    """
    t, c = np.asarray(times, float), np.asarray(conc, float)
    keep = np.arange(len(t)) >= first_index
    t, c = t[keep & (c > 0)], c[keep & (c > 0)]
    best: list[tuple[float, int, float]] = []  # (adj_r2, n, lambda)
    for k in range(min_points, len(t) + 1):
        tt, yy = t[-k:], np.log(c[-k:])
        slope, intercept = np.polyfit(tt, yy, 1)
        if slope >= 0:
            continue
        resid = yy - (slope * tt + intercept)
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        r2 = 1.0 if ss_tot == 0 else 1 - (resid**2).sum() / ss_tot
        adj = 1 - (1 - r2) * (k - 1) / (k - 2)
        best.append((adj, k, -slope))
    if not best:
        return float("nan"), float("nan"), 0
    top = max(b[0] for b in best)
    adj, k, lam = max((b for b in best if b[0] >= top - r2_tolerance), key=lambda b: b[1])
    return float(lam), float(adj), int(k)


def nca(times, conc, dose: float, route: str = "oral", duration: float = 0.0,
        method: str = "linear-up/log-down") -> NCAResult:
    """Single-dose NCA of one concentration-time profile (dose given at t = 0)."""
    if route not in ("iv_bolus", "infusion", "oral"):
        raise ValueError("route must be iv_bolus, infusion or oral")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t, c = np.asarray(times, float), np.asarray(conc, float)
    ok = ~np.isnan(c)
    t, c = t[ok], c[ok]
    if route == "iv_bolus":
        t, c = t[t > 0], c[t > 0]
    if len(t) < 3:
        raise ValueError("NCA needs at least three concentrations")
    if np.any(np.diff(t) <= 0) or np.any(c < 0) or t[0] < 0:
        raise ValueError("times must be strictly increasing and concentrations non-negative")

    i_max = int(np.argmax(c))
    cmax, tmax = float(c[i_max]), float(t[i_max])
    positive = np.nonzero(c > 0)[0]
    if positive.size == 0:
        raise ValueError("all concentrations are zero")
    i_last = int(positive[-1])
    clast, tlast = float(c[i_last]), float(t[i_last])

    # profile from t = 0 for the area calculations
    if t[0] > 0:
        if route == "iv_bolus":
            c0 = c[0]
            if c[1] > 0 and c[1] < c[0]:
                c0 = c[0] * np.exp(np.log(c[0] / c[1]) / (t[1] - t[0]) * t[0])
            t_full, c_full = np.r_[0.0, t], np.r_[c0, c]
        else:
            t_full, c_full = np.r_[0.0, t], np.r_[0.0, c]
    else:
        t_full, c_full = t, c
    upto = t_full <= tlast
    seg_auc, seg_aumc = _segments(t_full[upto], c_full[upto], method)
    auc_last, aumc_last = float(seg_auc.sum()), float(seg_aumc.sum())

    first = i_max + 1 if route != "iv_bolus" else i_max
    lam, r2_adj, n_pts = lambda_z(t[: i_last + 1], c[: i_last + 1], first_index=first)
    nan = float("nan")
    if np.isfinite(lam):
        auc_inf = auc_last + clast / lam
        aumc_inf = aumc_last + clast * tlast / lam + clast / lam**2
        mrt = aumc_inf / auc_inf - (duration / 2 if route == "infusion" else 0.0)
        cl = dose / auc_inf
        vz = cl / lam
        vss = cl * mrt if route != "oral" else nan
        half_life = float(np.log(2) / lam)
        pct = 100 * (auc_inf - auc_last) / auc_inf
    else:
        auc_inf = aumc_inf = mrt = cl = vz = vss = half_life = pct = nan
    return NCAResult(route, float(dose), cmax, tmax, clast, tlast, auc_last, auc_inf, pct,
                     aumc_inf, lam, half_life, r2_adj, n_pts, mrt, cl, vz, vss)


def nca_subjects(subjects: list[Subject], method: str = "linear-up/log-down") -> pd.DataFrame:
    """NCA for every subject with a single dose at t = 0. Others are skipped with a reason."""
    rows = []
    for s in subjects:
        row = {"ID": s.id}
        if len(s.doses) != 1 or s.doses[0].time != 0 or s.dv is None:
            rows.append({**row, "note": "skipped: NCA here needs one dose at t = 0"})
            continue
        d = s.doses[0]
        try:
            res = nca(s.times, s.dv, d.amount, d.route, d.duration, method)
        except ValueError as exc:
            rows.append({**row, "note": f"skipped: {exc}"})
            continue
        rows.append({**row, **res.as_dict(), "note": ""})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Geometric means and CV% of the main exposure parameters across subjects."""
    cols = ["cmax", "tmax", "auc_last", "auc_inf", "half_life", "cl", "vz"]
    rows = []
    for col in cols:
        x = pd.to_numeric(table.get(col), errors="coerce").dropna()
        x = x[x > 0]
        if x.empty:
            continue
        logs = np.log(x)
        gcv = 100 * np.sqrt(np.expm1(logs.var(ddof=1))) if len(x) > 1 else float("nan")
        rows.append({"parameter": col, "n": len(x), "geometric_mean": float(np.exp(logs.mean())),
                     "geometric_cv_pct": float(gcv), "min": float(x.min()),
                     "max": float(x.max())})
    return pd.DataFrame(rows)
