"""Numerical ODE solution of compartment models.

Slower than :mod:`pkforge.models`, but it handles what has no closed form,
such as saturable (Michaelis-Menten) elimination, and it serves as an
independent check on the analytic engine: the two are implemented differently
and must agree to solver tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from pkforge.dosing import Dose
from pkforge.models import CENTRAL_VOLUME, PERIPHERALS

ELIMINATION = ("linear", "michaelis-menten")


def _get(params: dict, name: str) -> float:
    if name not in params:
        raise ValueError(f"missing parameter {name}")
    value = float(params[name])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"parameter {name} must be finite and positive")
    return value


def simulate_ode(params: dict, doses: list[Dose], times, n_cmt: int = 1,
                 elimination: str = "linear", rtol: float = 1e-9,
                 atol: float = 1e-12) -> np.ndarray:
    """Central concentrations at ``times`` by integrating the model ODEs.

    Linear elimination uses ``CL``; Michaelis-Menten elimination uses ``VMAX``
    (amount/time) and ``KM`` (concentration). Oral doses need ``KA`` and
    optionally ``F``.
    """
    if elimination not in ELIMINATION:
        raise ValueError(f"elimination must be one of {ELIMINATION}")
    if n_cmt not in PERIPHERALS:
        raise ValueError(f"n_cmt must be 1, 2 or 3, got {n_cmt}")
    vc = _get(params, CENTRAL_VOLUME[n_cmt])
    periph = [(_get(params, q), _get(params, v)) for q, v in PERIPHERALS[n_cmt]]
    if elimination == "linear":
        cl = _get(params, "CL")
        vmax = km = 0.0
    else:
        cl = 0.0
        vmax, km = _get(params, "VMAX"), _get(params, "KM")
    oral = any(d.route == "oral" for d in doses)
    ka = _get(params, "KA") if oral else 0.0
    bio = float(params.get("F", 1.0))

    def rhs(t, y, infusion_rate):
        conc = y[1] / vc
        elim = cl * conc if elimination == "linear" else vmax * conc / (km + conc)
        dy = np.zeros_like(y)
        dy[0] = -ka * y[0]
        dy[1] = ka * y[0] - elim + infusion_rate
        for j, (q, vp) in enumerate(periph):
            flow = q * (conc - y[2 + j] / vp)
            dy[1] -= flow
            dy[2 + j] += flow
        return dy

    times = np.asarray(times, dtype=float)
    out = np.zeros_like(times)
    t_end = float(times.max()) if times.size else 0.0
    breaks = {0.0, t_end}
    for d in doses:
        breaks.add(d.time)
        if d.route == "infusion":
            breaks.add(d.time + d.duration)
    breaks = sorted(b for b in breaks if b <= t_end)

    y = np.zeros(2 + len(periph))
    for t0, t1 in zip(breaks[:-1], breaks[1:], strict=True):
        for d in doses:  # doses at t0 act after any observation at t0 (pre-dose rule)
            if d.time == t0 and d.route == "iv_bolus":
                y[1] += d.amount
            elif d.time == t0 and d.route == "oral":
                y[0] += bio * d.amount
        rate = sum(d.rate for d in doses
                   if d.route == "infusion" and d.time <= t0 < d.time + d.duration)
        in_segment = (times > t0) & (times <= t1)
        seg_times = times[in_segment]
        t_eval = np.unique(np.append(seg_times, t1))  # always ends at t1: the next start state
        sol = solve_ivp(rhs, (t0, t1), y, method="LSODA", rtol=rtol, atol=atol,
                        t_eval=t_eval, args=(rate,))
        if not sol.success:
            raise RuntimeError(f"ODE solver failed between t={t0} and t={t1}: {sol.message}")
        if seg_times.size:
            out[in_segment] = sol.y[1, np.searchsorted(t_eval, seg_times)] / vc
        y = sol.y[:, -1]
    return out
