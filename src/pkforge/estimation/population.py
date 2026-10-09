"""Population PK: nonlinear mixed-effects models fitted by FOCE-I.

Individual parameters follow the usual log-normal model with optional power
covariate effects (fixed exponents, e.g. allometric weight scaling):

    P_i = theta_P * (COV_i / reference) ** exponent * exp(eta_iP),
    eta_i ~ N(0, diag(omega^2)).

Estimation minimises the FOCE-I objective (:mod:`pkforge.estimation.laplace`)
over log(theta), log(omega^2) and log(sigma) with L-BFGS-B, warm-starting the
inner conditional-mode searches from the previous evaluation. The covariance
step inverts a finite-difference Hessian of the objective, as NONMEM's
``$COVARIANCE MATRIX=R`` does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pkforge.dosing import Subject
from pkforge.estimation.error import ErrorModel
from pkforge.estimation.laplace import LOG_2PI, LaplaceResult, foce_ofv
from pkforge.models import Design, predict_design, required_parameters


@dataclass(frozen=True)
class Param:
    """One structural parameter: initial typical value, variability and covariate."""

    name: str
    value: float
    omega: float | None = 0.09  # initial variance of eta; None = no random effect
    covariate: str | None = None
    reference: float = 1.0
    exponent: float = 0.0
    fixed: bool = False

    def __post_init__(self):
        if not self.value > 0:
            raise ValueError(f"{self.name}: initial value must be positive")
        if self.omega is not None and not self.omega > 0:
            raise ValueError(f"{self.name}: omega must be positive or None")
        if self.covariate is not None and not self.reference > 0:
            raise ValueError(f"{self.name}: covariate reference must be positive")


@dataclass(frozen=True)
class PopModel:
    n_cmt: int
    params: tuple[Param, ...]
    error: ErrorModel = field(default_factory=ErrorModel)
    sigma: tuple[float, ...] = (0.2,)

    def __post_init__(self):
        if len(self.sigma) != len(self.error.names):
            raise ValueError(f"{self.error.kind} error needs {len(self.error.names)} sigma(s)")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names")
        if not any(p.omega is not None for p in self.params):
            raise ValueError("a population model needs at least one random effect")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def iiv(self) -> list[Param]:
        return [p for p in self.params if p.omega is not None]

    def with_values(self, theta: dict, omega: dict, sigma: dict) -> PopModel:
        """A copy whose initial values are the given estimates (e.g. for simulation)."""
        params = tuple(
            Param(p.name, theta[p.name], omega.get(p.name) if p.omega is not None else None,
                  p.covariate, p.reference, p.exponent, p.fixed) for p in self.params)
        return PopModel(self.n_cmt, params, self.error,
                        tuple(sigma[n] for n in self.error.names))


class Population:
    """A model bound to a dataset: individual parameters and predictions."""

    def __init__(self, model: PopModel, subjects: list[Subject]):
        self.model = model
        self.subjects = subjects
        self.design = Design.from_subjects(subjects)
        required = required_parameters(model.n_cmt, self.design.has_oral)
        missing = [r for r in required if r not in model.names]
        if missing:
            raise ValueError(f"model is missing parameter(s) {missing} for this dataset")
        unknown = [n for n in model.names if n not in required + ("F",)]
        if unknown:
            raise ValueError(f"parameter(s) {unknown} are not used by a {model.n_cmt}-compartment "
                             f"model with {'oral' if self.design.has_oral else 'IV'} dosing")
        N, n = self.design.times.shape
        self.y = np.zeros((N, n))
        for i, s in enumerate(subjects):
            if s.dv is not None:
                self.y[i, : len(s.dv)] = np.nan_to_num(s.dv, nan=0.0)
        has_dv = np.zeros((N, n), dtype=bool)
        for i, s in enumerate(subjects):
            if s.dv is not None:
                has_dv[i, : len(s.dv)] = np.isfinite(s.dv)
        self.mask = self.design.mask & has_dv
        self.cov_factor = {}
        for p in model.params:
            if p.covariate is None:
                self.cov_factor[p.name] = np.ones(N)
                continue
            values = np.array([s.covariates.get(p.covariate, np.nan) for s in subjects])
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"covariate {p.covariate} is missing or non-positive for some "
                                 f"subjects")
            self.cov_factor[p.name] = (values / p.reference) ** p.exponent
        positive = self.y[self.mask & (self.y > 0)]
        scale = float(np.median(positive)) if positive.size else 1.0
        self.variance_floor = (1e-4 * scale) ** 2
        self.data_scale = scale

    @property
    def n_obs(self) -> int:
        return int(self.mask.sum())

    def individual(self, theta: dict, eta: np.ndarray) -> dict[str, np.ndarray]:
        out, k = {}, 0
        for p in self.model.params:
            value = theta[p.name] * self.cov_factor[p.name]
            if p.omega is not None:
                value = value * np.exp(eta[:, k])
                k += 1
            out[p.name] = value
        return out

    def predict(self, theta: dict, eta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return predict_design(self.individual(theta, eta), self.design, self.model.n_cmt,
                                  validate=False)


class FOCEI:
    """FOCE-I estimation for one model and dataset."""

    def __init__(self, model: PopModel, subjects: list[Subject]):
        self.pop = Population(model, subjects)
        self.model = model
        self.free = [p for p in model.params if not p.fixed]
        self.q = len(model.iiv)
        self._eta = np.zeros((len(subjects), self.q))
        self.n_evaluations = 0

    # parameter vector <-> named values ------------------------------------------------
    def pack(self, theta: dict, omega: dict, sigma: dict) -> np.ndarray:
        return np.log(np.r_[[theta[p.name] for p in self.free],
                            [omega[p.name] for p in self.model.iiv],
                            [sigma[n] for n in self.model.error.names]])

    def unpack(self, phi: np.ndarray) -> tuple[dict, dict, dict]:
        values = np.exp(np.asarray(phi, dtype=float))
        nf, q = len(self.free), self.q
        theta = {p.name: p.value for p in self.model.params}
        theta.update({p.name: float(v) for p, v in zip(self.free, values[:nf], strict=True)})
        omega = {p.name: float(v) for p, v in zip(self.model.iiv, values[nf:nf + q], strict=True)}
        sigma = {n: float(v) for n, v in zip(self.model.error.names, values[nf + q:], strict=True)}
        return theta, omega, sigma

    def initial(self) -> np.ndarray:
        m = self.model
        return self.pack({p.name: p.value for p in m.params},
                         {p.name: p.omega for p in m.iiv},
                         dict(zip(m.error.names, m.sigma, strict=True)))

    def bounds(self) -> list[tuple[float, float]]:
        out = [(np.log(p.value) - 7, np.log(p.value) + 7) for p in self.free]
        out += [(np.log(1e-5), np.log(10.0))] * self.q
        for name in self.model.error.names:
            scale = self.pop.data_scale if name == "sigma_add" else 1.0
            out.append((np.log(1e-6 * scale), np.log(10 * scale)))
        return out

    # objective --------------------------------------------------------------------------
    def evaluate(self, phi: np.ndarray, eta0: np.ndarray | None = None) -> LaplaceResult:
        theta, omega, sigma = self.unpack(phi)
        omega_mat = np.diag([omega[p.name] for p in self.model.iiv])
        sig = np.array([sigma[n] for n in self.model.error.names])
        self.n_evaluations += 1
        start = self._eta if eta0 is None else eta0
        return foce_ofv(self.pop.y, self.pop.mask, lambda e: self.pop.predict(theta, e),
                        omega_mat, self.model.error, sig, start, self.pop.variance_floor,
                        extra_starts=(np.zeros_like(start),))

    def objective(self, phi: np.ndarray, store: bool = True) -> float:
        base = self._eta.copy()
        try:
            res = self.evaluate(phi, base)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float("inf")
        if store and np.isfinite(res.ofv):
            self._eta = res.eta
        return res.ofv

    def _value_and_gradient(self, phi: np.ndarray, h: float = 1e-4):
        f0 = self.objective(phi, store=True)
        if not np.isfinite(f0):
            return 1e12, np.zeros_like(phi)
        grad = np.zeros_like(phi)
        for k in range(len(phi)):
            step = np.zeros_like(phi)
            step[k] = h
            hi, lo = self.objective(phi + step, store=False), self.objective(phi - step,
                                                                             store=False)
            grad[k] = (hi - lo) / (2 * h) if np.isfinite(hi) and np.isfinite(lo) else 0.0
        return f0, grad

    def hessian(self, phi: np.ndarray, h: float = 1e-2) -> np.ndarray:
        """Finite-difference Hessian of the objective in log-parameter space.

        The step (1% on the natural scale) is deliberately not tiny: the inner
        optimisation leaves noise of order 1e-6 in the objective, and a second
        difference divides that noise by h^2.
        """
        k = len(phi)
        f0 = self.objective(phi, store=False)
        H = np.zeros((k, k))
        ev = np.eye(k) * h
        f_plus = [self.objective(phi + ev[a], store=False) for a in range(k)]
        f_minus = [self.objective(phi - ev[a], store=False) for a in range(k)]
        for a in range(k):
            H[a, a] = (f_plus[a] - 2 * f0 + f_minus[a]) / h**2
            for b in range(a + 1, k):
                fpp = self.objective(phi + ev[a] + ev[b], store=False)
                fpm = self.objective(phi + ev[a] - ev[b], store=False)
                fmp = self.objective(phi - ev[a] + ev[b], store=False)
                fmm = self.objective(phi - ev[a] - ev[b], store=False)
                H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4 * h * h)
        return H

    # fit --------------------------------------------------------------------------------
    def fit(self, maxiter: int = 300, covariance: bool = True) -> PopFit:
        start = time.perf_counter()
        phi0 = self.initial()
        f0 = self.objective(phi0)  # also warm-starts the conditional modes
        # Precondition by the diagonal curvature at the start. Unscaled, L-BFGS-B's
        # first step is the raw gradient, which for an objective summed over
        # hundreds of observations is a jump of e^20 in log-parameter space.
        h = 1e-2
        curvature = np.array([
            (self.objective(phi0 + h * e, store=False) - 2 * f0
             + self.objective(phi0 - h * e, store=False)) / h**2
            for e in np.eye(len(phi0))])
        d = np.sqrt(np.clip(np.nan_to_num(np.abs(curvature), nan=1.0), 1.0, None))

        def scaled(x):
            f, g = self._value_and_gradient(phi0 + x / d)
            return f, g / d

        bounds = [((lo - p) * dk, (hi - p) * dk)
                  for (lo, hi), p, dk in zip(self.bounds(), phi0, d, strict=True)]
        opt = minimize(scaled, np.zeros_like(phi0), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6})
        phi = phi0 + opt.x / d
        res = self.evaluate(phi)
        self._eta = res.eta
        theta, omega, sigma = self.unpack(phi)

        se, cov_ok, cond = {}, False, float("nan")
        if covariance:
            H = self.hessian(phi)
            eig = np.linalg.eigvalsh((H + H.T) / 2) if np.all(np.isfinite(H)) else None
            if eig is not None and eig.min() > 0:
                cov_ok, cond = True, float(eig.max() / eig.min())
                cov_phi = 2 * np.linalg.inv(H)
                se_log = np.sqrt(np.clip(np.diag(cov_phi), 0, None))
                labels = ([f"theta:{p.name}" for p in self.free]
                          + [f"omega:{p.name}" for p in self.model.iiv]
                          + [f"sigma:{n}" for n in self.model.error.names])
                values = np.exp(phi)
                se = {lab: float(v * s) for lab, v, s in zip(labels, values, se_log, strict=True)}
        return PopFit(
            model=self.model, theta=theta, omega=omega, sigma=sigma, minus2ll=res.ofv,
            n_subjects=len(self.pop.subjects), n_obs=self.pop.n_obs, n_estimated=len(phi),
            converged=bool(opt.success), message=str(opt.message),
            n_evaluations=self.n_evaluations, seconds=time.perf_counter() - start,
            se=se, covariance_ok=cov_ok, condition_number=cond,
            **_diagnostics(self.pop, theta, omega, sigma, res))


def _diagnostics(pop: Population, theta, omega, sigma, res: LaplaceResult) -> dict:
    model = pop.model
    sig = np.array([sigma[n] for n in model.error.names])
    omega_mat = np.diag([omega[p.name] for p in model.iiv])
    pred = pop.predict(theta, np.zeros_like(res.eta))
    ipred, G = res.f, res.G
    v = model.error.variance(ipred, sig, pop.variance_floor)
    rows = []
    for i, s in enumerate(pop.subjects):
        m = pop.mask[i]
        if not m.any():
            continue
        Gi = G[i][m]
        C = Gi @ omega_mat @ Gi.T + np.diag(v[i][m])
        r = pop.y[i][m] - ipred[i][m] + Gi @ res.eta[i]
        cwres = np.linalg.solve(np.linalg.cholesky(C), r)
        iwres = (pop.y[i][m] - ipred[i][m]) / np.sqrt(v[i][m])
        for t, dv, p0, ip, iw, cw in zip(pop.design.times[i][m], pop.y[i][m], pred[i][m],
                                         ipred[i][m], iwres, cwres, strict=True):
            rows.append({"ID": s.id, "TIME": t, "DV": dv, "PRED": p0, "IPRED": ip,
                         "IWRES": iw, "CWRES": cw})
    eta_df = pd.DataFrame(res.eta, columns=[f"ETA_{p.name}" for p in model.iiv])
    eta_df.insert(0, "ID", [s.id for s in pop.subjects])
    ind = pop.individual(theta, res.eta)
    ind_df = pd.DataFrame({"ID": [s.id for s in pop.subjects], **ind})
    shrinkage = {p.name: float(1 - np.std(res.eta[:, k], ddof=1) / np.sqrt(omega[p.name]))
                 for k, p in enumerate(model.iiv)}
    return {"diagnostics": pd.DataFrame(rows), "eta": eta_df, "individual": ind_df,
            "shrinkage": shrinkage}


@dataclass
class PopFit:
    model: PopModel
    theta: dict[str, float]
    omega: dict[str, float]  # variances of eta
    sigma: dict[str, float]  # residual standard deviations
    minus2ll: float  # -2 log-likelihood including constants
    n_subjects: int
    n_obs: int
    n_estimated: int
    converged: bool
    message: str
    n_evaluations: int
    seconds: float
    se: dict[str, float]
    covariance_ok: bool
    condition_number: float
    diagnostics: pd.DataFrame
    eta: pd.DataFrame
    individual: pd.DataFrame
    shrinkage: dict[str, float]

    @property
    def ofv(self) -> float:
        """Objective function value in NONMEM's convention (without n*log(2*pi))."""
        return self.minus2ll - self.n_obs * LOG_2PI

    @property
    def aic(self) -> float:
        return self.minus2ll + 2 * self.n_estimated

    @property
    def bic(self) -> float:
        return self.minus2ll + self.n_estimated * np.log(self.n_obs)

    def parameter_table(self) -> pd.DataFrame:
        rows = []
        for p in self.model.params:
            se = self.se.get(f"theta:{p.name}")
            rows.append({"parameter": p.name, "kind": "theta", "estimate": self.theta[p.name],
                         "se": se, "rse_pct": _rse(se, self.theta[p.name]),
                         "note": "fixed" if p.fixed else ""})
        for name, var in self.omega.items():
            se = self.se.get(f"omega:{name}")
            cv = 100 * np.sqrt(np.expm1(var))
            rows.append({"parameter": f"omega2({name})", "kind": "omega", "estimate": var,
                         "se": se, "rse_pct": _rse(se, var),
                         "note": f"CV {cv:.1f}%, shrinkage {100 * self.shrinkage[name]:.1f}%"})
        for name, s in self.sigma.items():
            se = self.se.get(f"sigma:{name}")
            rows.append({"parameter": name, "kind": "sigma", "estimate": s, "se": se,
                         "rse_pct": _rse(se, s), "note": ""})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"FOCE-I fit: {self.n_subjects} subjects, {self.n_obs} observations, "
            f"{self.n_estimated} estimated parameters",
            f"converged: {self.converged} ({self.message}); {self.n_evaluations} objective "
            f"evaluations in {self.seconds:.1f} s",
            f"OFV (NONMEM convention) {self.ofv:.3f}   -2LL {self.minus2ll:.3f}   "
            f"AIC {self.aic:.3f}   BIC {self.bic:.3f}",
            "covariance step: " + (f"OK, condition number {self.condition_number:.3g}"
                                   if self.covariance_ok else "FAILED (Hessian not positive "
                                   "definite); standard errors unavailable"),
            "",
            self.parameter_table().to_string(index=False, float_format=lambda x: f"{x:.4g}"),
        ]
        return "\n".join(lines)


def _rse(se: float | None, estimate: float) -> float | None:
    return None if se is None else 100 * se / abs(estimate)


def fit(model: PopModel, subjects: list[Subject], **kwargs) -> PopFit:
    return FOCEI(model, subjects).fit(**kwargs)


def simulate(model: PopModel, subjects: list[Subject], seed: int = 0,
             theta: dict | None = None, omega: dict | None = None,
             sigma: dict | None = None) -> list[Subject]:
    """New observations for the same design, drawn from the model.

    Parameter values default to the model's initial values.
    """
    rng = np.random.default_rng(seed)
    pop = Population(model, subjects)
    theta = theta or {p.name: p.value for p in model.params}
    omega = omega or {p.name: p.omega for p in model.iiv}
    sigma = sigma or dict(zip(model.error.names, model.sigma, strict=True))
    sd = np.sqrt([omega[p.name] for p in model.iiv])
    eta = rng.standard_normal((len(subjects), len(sd))) * sd
    f = pop.predict(theta, eta)
    v = model.error.variance(f, [sigma[n] for n in model.error.names])
    y = f + np.sqrt(v) * rng.standard_normal(f.shape)
    return [s.with_dv(y[i, : len(s.times)]) for i, s in enumerate(subjects)]
