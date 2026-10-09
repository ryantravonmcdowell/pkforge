"""Linear compartment models, solved exactly as sums of exponentials.

For any linear mammillary model, the central concentration after a unit bolus
into the central compartment is ``sum_i c_i * exp(-lambda_i * t)``. Bolus,
zero-order infusion and first-order absorption are then convolutions of that
impulse response with the input, each of which has a closed form, and multiple
doses add by superposition. Everything here is vectorised over a batch of
subjects so the population estimator can evaluate a whole study at once.

The coefficients come from closed forms for one and two compartments and from
an eigendecomposition of the rate matrix in general; the two are tested against
each other and against a numerical ODE solution (``pkforge.ode``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkforge.dosing import ROUTE_CODE, Dose, Subject

PARAMETERS = {
    1: ("CL", "V"),
    2: ("CL", "V1", "Q", "V2"),
    3: ("CL", "V1", "Q2", "V2", "Q3", "V3"),
}
PERIPHERALS = {1: (), 2: (("Q", "V2"),), 3: (("Q2", "V2"), ("Q3", "V3"))}
CENTRAL_VOLUME = {1: "V", 2: "V1", 3: "V1"}
ABSORPTION = ("KA", "F")


def required_parameters(n_cmt: int, oral: bool) -> tuple[str, ...]:
    if n_cmt not in PARAMETERS:
        raise ValueError(f"n_cmt must be 1, 2 or 3, got {n_cmt}")
    return PARAMETERS[n_cmt] + (("KA",) if oral else ())


@dataclass
class Design:
    """Observation times and doses for N subjects, padded to rectangular arrays."""

    times: np.ndarray  # (N, n)
    mask: np.ndarray  # (N, n) True where a real observation exists
    dose_time: np.ndarray  # (N, d)
    dose_amt: np.ndarray  # (N, d), 0 for padding
    dose_route: np.ndarray  # (N, d) route codes
    dose_dur: np.ndarray  # (N, d) infusion durations, 0 otherwise

    @classmethod
    def from_subjects(cls, subjects: list[Subject]) -> Design:
        n = max(1, max(len(s.times) for s in subjects))
        d = max(len(s.doses) for s in subjects)
        N = len(subjects)
        times, mask = np.zeros((N, n)), np.zeros((N, n), dtype=bool)
        dt, da, dr, dd = np.zeros((N, d)), np.zeros((N, d)), np.zeros((N, d), int), np.zeros((N, d))
        for i, s in enumerate(subjects):
            times[i, : len(s.times)] = s.times
            mask[i, : len(s.times)] = True
            for j, dose in enumerate(s.doses):
                dt[i, j], da[i, j] = dose.time, dose.amount
                dr[i, j], dd[i, j] = ROUTE_CODE[dose.route], dose.duration
        return cls(times, mask, dt, da, dr, dd)

    @classmethod
    def single(cls, doses: list[Dose], times) -> Design:
        return cls.from_subjects([Subject("1", doses, np.sort(np.asarray(times, float)))])

    @property
    def n_subjects(self) -> int:
        return self.times.shape[0]

    @property
    def has_oral(self) -> bool:
        return bool(np.any((self.dose_route == ROUTE_CODE["oral"]) & (self.dose_amt > 0)))


def _as_batch(params: dict, names, N: int, validate: bool = True) -> dict[str, np.ndarray]:
    missing = [p for p in names if p not in params]
    if missing:
        raise ValueError(f"missing parameter(s): {', '.join(missing)}")
    out = {}
    for name in names:
        value = np.broadcast_to(np.asarray(params[name], dtype=float), (N,))
        if validate and (not np.all(np.isfinite(value)) or np.any(value <= 0)):
            raise ValueError(f"parameter {name} must be finite and positive")
        out[name] = value
    return out


def rate_matrix(params: dict, n_cmt: int, N: int, validate: bool = True) -> np.ndarray:
    """First-order rate matrix K (N, n, n) for amounts: dA/dt = K A."""
    p = _as_batch(params, PARAMETERS[n_cmt], N, validate)
    v1 = p[CENTRAL_VOLUME[n_cmt]]
    K = np.zeros((N, n_cmt, n_cmt))
    K[:, 0, 0] = -p["CL"] / v1
    for j, (q, vp) in enumerate(PERIPHERALS[n_cmt], start=1):
        k_out, k_back = p[q] / v1, p[q] / p[vp]
        K[:, 0, 0] -= k_out
        K[:, j, 0] = k_out
        K[:, 0, j] = k_back
        K[:, j, j] = -k_back
    return K


def disposition_eigen(params: dict, n_cmt: int, N: int,
                      validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Impulse-response coefficients and rates from an eigendecomposition of K."""
    K = rate_matrix(params, n_cmt, N, validate)
    if not validate:  # keep the decomposition defined for out-of-range trial values
        K = np.where(np.isfinite(K), K, 0.0)
    mu, S = np.linalg.eig(K)
    S_inv = np.linalg.inv(S)
    v1 = _as_batch(params, (CENTRAL_VOLUME[n_cmt],), N, validate)[CENTRAL_VOLUME[n_cmt]]
    coefs = (S[:, 0, :] * S_inv[:, :, 0]).real / v1[:, None]
    return coefs, -mu.real


def disposition(params: dict, n_cmt: int, N: int,
                validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Impulse-response coefficients (N, m) and rate constants (N, m)."""
    if n_cmt == 1:
        p = _as_batch(params, PARAMETERS[1], N, validate)
        return (1 / p["V"])[:, None], (p["CL"] / p["V"])[:, None]
    if n_cmt == 2:
        p = _as_batch(params, PARAMETERS[2], N, validate)
        k10, k12, k21 = p["CL"] / p["V1"], p["Q"] / p["V1"], p["Q"] / p["V2"]
        a = k10 + k12 + k21
        alpha = (a + np.sqrt(a * a - 4 * k10 * k21)) / 2
        beta = k10 * k21 / alpha  # avoids cancellation in (a - sqrt(...)) / 2
        span = p["V1"] * (alpha - beta)
        coefs = np.stack([(alpha - k21) / span, (k21 - beta) / span], axis=1)
        return coefs, np.stack([alpha, beta], axis=1)
    if n_cmt == 3:
        return disposition_eigen(params, 3, N, validate)
    raise ValueError(f"n_cmt must be 1, 2 or 3, got {n_cmt}")


def _phi(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(1 - exp(-x s)) / x, continuous at x = 0 where it equals s."""
    small = np.abs(x) < 1e-10
    x_safe = np.where(small, 1.0, x)
    return np.where(small, s * (1 - 0.5 * x * s), -np.expm1(-x_safe * s) / x_safe)


def predict_design(params: dict, design: Design, n_cmt: int = 1,
                   validate: bool = True) -> np.ndarray:
    """Central concentrations (N, n) for every subject in ``design``.

    ``params`` maps parameter names to scalars or (N,) arrays. With
    ``validate=False`` (used inside the estimator) out-of-range values give
    non-finite predictions instead of an exception, so a line search can reject
    them.
    """
    N = design.n_subjects
    coefs, lambdas = disposition(params, n_cmt, N, validate)
    oral = design.has_oral
    if oral:
        absorption = _as_batch({"F": 1.0, **params}, ABSORPTION, N, validate)
        ka, F = absorption["KA"], absorption["F"]
    else:
        ka, F = np.ones(N), np.ones(N)

    s = design.times[:, :, None] - design.dose_time[:, None, :]  # (N, n, d)
    active = (s > 0) & (design.dose_amt[:, None, :] > 0)
    s = np.where(active, s, 0.0)[..., None]  # (N, n, d, 1)
    lam = lambdas[:, None, None, :]  # (N, 1, 1, m)
    c = coefs[:, None, None, :]
    amt = design.dose_amt[:, None, :, None]
    route = design.dose_route[:, None, :, None]

    response = amt * c * np.exp(-lam * s)  # bolus
    if np.any(design.dose_route == ROUTE_CODE["infusion"]):
        dur = design.dose_dur[:, None, :, None]
        rate = amt / np.where(dur > 0, dur, 1.0)
        during, after = np.minimum(s, dur), np.maximum(s - dur, 0.0)
        infusion = rate * c * _phi(lam, during) * np.exp(-lam * after)
        response = np.where(route == ROUTE_CODE["infusion"], infusion, response)
    if oral:
        ka4, F4 = ka[:, None, None, None], F[:, None, None, None]
        lo, hi = np.minimum(ka4, lam), np.maximum(ka4, lam)
        # (exp(-lam s) - exp(-ka s)) / (ka - lam), written so it never overflows
        absorbed = F4 * amt * ka4 * c * np.exp(-lo * s) * _phi(hi - lo, s)
        response = np.where(route == ROUTE_CODE["oral"], absorbed, response)
    response = np.where(active[..., None], response, 0.0)
    return response.sum(axis=(2, 3))


def predict(params: dict, doses: list[Dose], times, n_cmt: int = 1) -> np.ndarray:
    """Concentrations at ``times`` for one subject. Times need not be sorted."""
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    design = Design.single(doses, times)
    out = np.empty_like(times)
    out[order] = predict_design(params, design, n_cmt)[0, : len(times)]
    return out


def terminal_half_life(params: dict, n_cmt: int) -> float:
    """ln 2 divided by the smallest disposition rate constant."""
    _, lambdas = disposition(params, n_cmt, 1)
    return float(np.log(2) / lambdas.min())
