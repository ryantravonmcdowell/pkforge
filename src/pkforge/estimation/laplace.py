"""The FOCE-I likelihood, independent of any particular PK model.

For subject i with random effects eta ~ N(0, Omega) and observations
y_ij ~ N(f_ij(eta), v_ij(eta)), the marginal likelihood integrates eta out.
FOCE-I approximates that integral by Laplace's method at the conditional mode
eta_hat, with the Hessian replaced by its expected (first-order) value:

    -2 log L_i ~= sum_j [log(2 pi v_ij) + r_ij^2 / v_ij] + eta_hat' Omega^-1 eta_hat
                  + log |I + Omega M_i|,
    M_i = sum_j g_j g_j' (1 / v_ij + v'_ij^2 / (2 v_ij^2)),   g_j = d f_ij / d eta,

with every term evaluated at eta_hat ("interaction": the residual variance is
evaluated at the individual prediction). For a model that is linear in eta
with additive error this is the exact Gaussian marginal likelihood, which is
how the implementation is tested.

All functions work on a batch of N subjects padded to n observations, with a
boolean mask marking the real ones; ``predict`` maps eta (N, q) to f (N, n).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pkforge.estimation.error import ErrorModel

LOG_2PI = float(np.log(2 * np.pi))
Predict = Callable[[np.ndarray], np.ndarray]


@dataclass
class LaplaceResult:
    ofv: float  # total -2 log-likelihood, including the 2*pi constants
    ofv_i: np.ndarray  # (N,)
    eta: np.ndarray  # (N, q) conditional modes
    f: np.ndarray  # (N, n) individual predictions at the modes
    G: np.ndarray  # (N, n, q) d f / d eta at the modes
    v: np.ndarray  # (N, n) residual variances at the modes
    converged: np.ndarray  # (N,) inner optimisation converged
    iterations: int


def jacobian(predict: Predict, eta: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference d f / d eta, shape (N, n, q)."""
    cols = []
    for k in range(eta.shape[1]):
        step = np.zeros_like(eta)
        step[:, k] = h
        cols.append((predict(eta + step) - predict(eta - step)) / (2 * h))
    return np.stack(cols, axis=-1)


def _conditional_objective(y, mask, f, eta, omega_inv, error, sigma, floor):
    v = error.variance(f, sigma, floor)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        terms = np.where(mask, (y - f) ** 2 / v + np.log(v), 0.0).sum(axis=1)
    terms = terms + np.einsum("nk,kl,nl->n", eta, omega_inv, eta)
    return np.where(np.isfinite(terms), terms, np.inf)


def conditional_modes(y, mask, predict: Predict, omega: np.ndarray, error: ErrorModel, sigma,
                      eta0: np.ndarray, floor: float = 0.0, max_iter: int = 100,
                      tol: float = 1e-12, max_step: float = 3.0,
                      extra_starts: tuple = ()) -> tuple[np.ndarray, np.ndarray, int]:
    """Batched damped Newton search for each subject's most likely eta.

    Minimises sum_j [r^2/v + log v] + eta' Omega^-1 eta. Far from the optimum
    it takes Fisher-scoring steps (exact gradient, expected Hessian), which are
    robust but converge only linearly when residuals are large; once a
    subject's Newton decrement is small it switches to the exact Hessian (by
    differencing the analytic gradient) for quadratic convergence. A
    backtracking (Armijo) line search keeps every accepted step downhill.

    Each subject starts from whichever of ``eta0`` and ``extra_starts`` scores
    best, so a warm start inherited from a wild trial point cannot trap it.
    """
    omega_inv = np.linalg.inv(omega)
    eta = eta0.copy()
    N, q = eta.shape
    if extra_starts:
        with np.errstate(all="ignore"):
            best = _conditional_objective(y, mask, predict(eta), eta, omega_inv, error, sigma,
                                          floor)
            for start in extra_starts:
                score = _conditional_objective(y, mask, predict(start), start, omega_inv, error,
                                               sigma, floor)
                better = score < best
                eta = np.where(better[:, None], start, eta)
                best = np.where(better, score, best)

    def local_terms(e):
        """Gradient, expected Hessian and what the exact Hessian reuses."""
        with np.errstate(all="ignore"):
            f = predict(e)
            G = jacobian(predict, e)
            v = error.variance(f, sigma, floor)
            dv = error.dvariance(f, sigma, floor)
            r = y - f
            # d/df and expected d2/df2 of  r^2/v + log v
            d1 = np.where(mask, -2 * r / v + (1 - r**2 / v) * dv / v, 0.0)
            w = np.where(mask, 2 / v + dv**2 / v**2, 0.0)
        grad = np.einsum("nj,njk->nk", d1, G) + 2 * e @ omega_inv
        fisher = np.einsum("nj,nja,njb->nab", w, G, G) + 2 * omega_inv
        return grad, fisher, (f, G, v, dv, r, d1)

    def exact_hessian(e, f, G, v, dv, r, d1, h=1e-4):
        """sum_j [phi''(f) g g' + phi'(f) d2f/deta2] + 2 Omega^-1, phi = r^2/v + log v.

        phi' and phi'' are analytic; only the model's second derivatives are
        taken by finite differences (2q + 4 * q(q-1)/2 predictions).
        """
        d2v = error.d2variance(f, sigma, floor)
        with np.errstate(all="ignore"):
            d2 = np.where(mask, 2 / v + 4 * r * dv / v**2 - r**2 * d2v / v**2
                          + 2 * r**2 * dv**2 / v**3 + d2v / v - dv**2 / v**2, 0.0)
            F = np.empty(f.shape + (q, q))
            unit = np.eye(q) * h
            for a in range(q):
                F[..., a, a] = (predict(e + unit[a]) - 2 * f + predict(e - unit[a])) / h**2
                for b in range(a + 1, q):
                    F[..., a, b] = F[..., b, a] = (
                        predict(e + unit[a] + unit[b]) - predict(e + unit[a] - unit[b])
                        - predict(e - unit[a] + unit[b]) + predict(e - unit[a] - unit[b])
                    ) / (4 * h * h)
        return (np.einsum("nj,nja,njb->nab", d2, G, G) + np.einsum("nj,njab->nab", d1, F)
                + 2 * omega_inv)

    with np.errstate(all="ignore"):
        h_cur = _conditional_objective(y, mask, predict(eta), eta, omega_inv, error, sigma,
                                       floor)
    converged = np.zeros(N, dtype=bool)
    near = np.zeros(N, dtype=bool)
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        grad, H, terms = local_terms(eta)
        if near.any():
            exact = exact_hessian(eta, *terms)
            with np.errstate(all="ignore"):
                finite = np.isfinite(exact).all(axis=(1, 2))
                pd = finite & (np.linalg.eigvalsh(np.where(finite[:, None, None], exact,
                                                           np.eye(q))) > 0).all(axis=1)
            use = near & pd
            H = np.where(use[:, None, None], exact, H)
        bad = ~(np.isfinite(H).all(axis=(1, 2)) & np.isfinite(grad).all(axis=1))
        H[bad], grad[bad] = 2 * omega_inv, 0.0  # a subject with a broken prediction stays put
        step = -np.linalg.solve(H, grad[..., None])[..., 0]
        decrement = -(grad * step).sum(axis=1)
        # cap each subject's step at max_step in eta units (a factor of e^3 on a
        # log-normal parameter); the decrement keeps the uncapped Newton value
        largest = np.abs(step).max(axis=1)
        step = step * np.minimum(1.0, max_step / np.where(largest > 0, largest, 1.0))[:, None]
        slope = -(grad * step).sum(axis=1)  # expected decrease along the capped step
        converged = np.isfinite(h_cur) & (decrement < tol * (1 + np.abs(h_cur)))
        if converged.all():
            break
        near = np.isfinite(h_cur) & (decrement < 1e-3 * (1 + np.abs(h_cur)))
        t = np.ones(N)
        for _ in range(40):
            cand = eta + t[:, None] * step
            with np.errstate(all="ignore"):
                h_new = _conditional_objective(y, mask, predict(cand), cand, omega_inv, error,
                                               sigma, floor)
            ok = (h_new <= h_cur - 1e-4 * t * slope) | converged
            if ok.all():
                break
            t = np.where(ok, t, t / 2)
        move = ok & ~converged
        eta = np.where(move[:, None], cand, eta)
        h_cur = np.where(move, h_new, h_cur)
        if not move.any():
            break
    return eta, converged, iterations


def foce_ofv(y, mask, predict: Predict, omega: np.ndarray, error: ErrorModel, sigma,
             eta0: np.ndarray, floor: float = 0.0, extra_starts: tuple = ()) -> LaplaceResult:
    """FOCE-I objective (-2 log-likelihood) at the conditional modes."""
    eta, converged, iterations = conditional_modes(y, mask, predict, omega, error, sigma, eta0,
                                                   floor, extra_starts=extra_starts)
    with np.errstate(all="ignore"):
        f = predict(eta)
        G = jacobian(predict, eta)
    v = error.variance(f, sigma, floor)
    dv = error.dvariance(f, sigma, floor)
    r = y - f
    with np.errstate(divide="ignore", invalid="ignore"):
        data = np.where(mask, LOG_2PI + np.log(v) + r**2 / v, 0.0).sum(axis=1)
    quad = np.einsum("nk,kl,nl->n", eta, np.linalg.inv(omega), eta)
    w = np.where(mask, 1 / v + dv**2 / (2 * v**2), 0.0)
    M = np.einsum("nj,nja,njb->nab", w, G, G)
    q = omega.shape[0]
    sign, logdet = np.linalg.slogdet(np.eye(q) + omega @ M)
    ofv_i = data + quad + np.where(sign > 0, logdet, np.inf)
    total = float(ofv_i.sum()) if np.all(np.isfinite(ofv_i)) else float("inf")
    return LaplaceResult(total, ofv_i, eta, f, G, v, converged, iterations)
