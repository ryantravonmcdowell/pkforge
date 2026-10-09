"""Visual predictive checks (VPC), optionally prediction-corrected (pcVPC).

The fitted model is used to simulate the study design many times. In each time
bin the percentiles of the observed data are compared with the distribution
of the same percentiles across simulated replicates. A model that describes
the data puts the observed percentiles inside the simulated intervals.

Prediction correction (Bergstrand et al., AAPS J 2011) rescales each value by
the median population prediction of its bin divided by its own population
prediction, removing variability that comes from the design (different doses,
covariates) rather than the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pkforge.dosing import Subject
from pkforge.estimation.population import PopFit, PopModel, Population


@dataclass
class VPCResult:
    bins: pd.DataFrame
    observed: pd.DataFrame
    percentiles: tuple[float, ...]
    ci: float
    n_sim: int
    prediction_corrected: bool

    def coverage(self) -> float:
        """Share of observed percentiles inside their simulated confidence interval."""
        inside = []
        for p in self.percentiles:
            col = _label(p)
            inside.append(self.bins[f"obs_{col}"].between(self.bins[f"sim_{col}_lo"],
                                                          self.bins[f"sim_{col}_hi"]))
        return float(pd.concat(inside).mean())


def _label(p: float) -> str:
    return f"p{p:g}".replace(".", "_")


def vpc(model: PopModel, subjects: list[Subject], theta: dict, omega: dict, sigma: dict,
        n_sim: int = 200, n_bins: int = 8, percentiles=(5, 50, 95), ci: float = 95,
        prediction_corrected: bool = False, seed: int = 0) -> VPCResult:
    pop = Population(model, subjects)
    mask = pop.mask
    times, dv = pop.design.times[mask], pop.y[mask]
    nominal = np.unique(times)
    if len(nominal) < 2:
        raise ValueError("need observations at more than one time for a VPC")
    if len(nominal) <= max(n_bins, 12):
        # a fixed sampling schedule: one bin per nominal time, never merging two
        edges = np.r_[nominal, nominal[-1]]
        bin_of = np.searchsorted(nominal, times)
    else:
        # irregular times: bins holding roughly equal numbers of observations
        edges = np.unique(np.quantile(times, np.linspace(0, 1, n_bins + 1)))
        bin_of = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)

    rng = np.random.default_rng(seed)
    sd = np.sqrt([omega[p.name] for p in model.iiv])
    sig = [sigma[n] for n in model.error.names]
    sims = np.empty((n_sim, int(mask.sum())))
    for r in range(n_sim):
        eta = rng.standard_normal((len(subjects), len(sd))) * sd
        f = pop.predict(theta, eta)
        y = f + np.sqrt(model.error.variance(f, sig)) * rng.standard_normal(f.shape)
        sims[r] = y[mask]

    if prediction_corrected:
        pred = pop.predict(theta, np.zeros((len(subjects), len(sd))))[mask]
        factor = np.ones_like(pred)
        for b in np.unique(bin_of):
            sel = bin_of == b
            factor[sel] = np.median(pred[sel]) / pred[sel]
        dv, sims = dv * factor, sims * factor

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    rows = []
    for b in range(len(edges) - 1):
        sel = bin_of == b
        if not sel.any():
            continue
        row = {"bin": b, "t_lo": edges[b], "t_hi": edges[b + 1],
               "t_mid": float(np.median(times[sel])), "n_obs": int(sel.sum())}
        sim_pct = np.percentile(sims[:, sel], percentiles, axis=1)  # (len(p), n_sim)
        obs_pct = np.percentile(dv[sel], percentiles)
        for p, obs, sim in zip(percentiles, obs_pct, sim_pct, strict=True):
            col = _label(p)
            row[f"obs_{col}"] = float(obs)
            row[f"sim_{col}_lo"], row[f"sim_{col}_med"], row[f"sim_{col}_hi"] = (
                float(v) for v in np.percentile(sim, [lo_q, 50, hi_q]))
        rows.append(row)
    observed = pd.DataFrame({"TIME": times, "DV": dv, "bin": bin_of})
    return VPCResult(pd.DataFrame(rows), observed, tuple(percentiles), ci, n_sim,
                     prediction_corrected)


def vpc_from_fit(fit: PopFit, subjects: list[Subject], **kwargs) -> VPCResult:
    return vpc(fit.model, subjects, fit.theta, fit.omega, fit.sigma, **kwargs)
