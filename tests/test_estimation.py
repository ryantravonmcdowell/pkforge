import numpy as np
import pytest
from scipy.integrate import quad

from pkforge.dosing import Dose, Subject
from pkforge.estimation import ErrorModel, Param, PopModel, fit, simulate
from pkforge.estimation.laplace import LOG_2PI, conditional_modes, foce_ofv
from pkforge.estimation.population import FOCEI


class TestLaplaceCore:
    """FOCE-I is exact when predictions are linear in eta and the error is additive."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.N, self.n, self.q = 6, 8, 2
        self.Z = rng.normal(size=(self.n, self.q))
        self.base = rng.normal(size=(self.N, self.n)) + 5
        self.omega = np.array([[0.5, 0.0], [0.0, 0.2]])
        self.sigma = 0.3
        eta = rng.multivariate_normal(np.zeros(2), self.omega, size=self.N)
        self.y = self.base + eta @ self.Z.T + rng.normal(scale=self.sigma, size=(self.N, self.n))
        self.mask = np.ones((self.N, self.n), dtype=bool)
        self.mask[0, -3:] = False  # ragged data
        self.predict = lambda e: self.base + e @ self.Z.T

    def test_objective_equals_exact_gaussian_marginal(self):
        res = foce_ofv(self.y, self.mask, self.predict, self.omega, ErrorModel("additive"),
                       [self.sigma], np.zeros((self.N, self.q)))
        exact = 0.0
        for i in range(self.N):
            m = self.mask[i]
            Z = self.Z[m]
            C = Z @ self.omega @ Z.T + self.sigma**2 * np.eye(m.sum())
            r = self.y[i, m] - self.base[i, m]
            exact += m.sum() * LOG_2PI + np.linalg.slogdet(C)[1] + r @ np.linalg.solve(C, r)
        assert res.ofv == pytest.approx(exact, rel=1e-9)

    def test_conditional_modes_equal_the_posterior_mean(self):
        eta, converged, _ = conditional_modes(self.y, self.mask, self.predict, self.omega,
                                              ErrorModel("additive"), [self.sigma],
                                              np.zeros((self.N, self.q)))
        assert converged.all()
        for i in range(self.N):
            m = self.mask[i]
            Z = self.Z[m]
            A = Z.T @ Z / self.sigma**2 + np.linalg.inv(self.omega)
            expected = np.linalg.solve(A, Z.T @ (self.y[i, m] - self.base[i, m]) / self.sigma**2)
            np.testing.assert_allclose(eta[i], expected, rtol=1e-7, atol=1e-10)

    def test_close_to_numerical_integration_for_a_nonlinear_model(self):
        # one subject, one random effect on clearance, proportional error
        times = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
        omega, sig = 0.1, 0.1
        y = 100 / 30 * np.exp(-0.2 * np.exp(0.15) * times) * (1 + 0.05 * np.sin(times))

        def f(e):
            return 100 / 30 * np.exp(-0.2 * np.exp(e) * times)

        def integrand(e):
            v = (sig * f(e)) ** 2
            loglik = -0.5 * np.sum(np.log(2 * np.pi * v) + (y - f(e)) ** 2 / v)
            return np.exp(loglik - e**2 / (2 * omega)) / np.sqrt(2 * np.pi * omega)

        exact = -2 * np.log(quad(integrand, -2, 2, points=[0.15], limit=200)[0])
        res = foce_ofv(y[None], np.ones((1, len(times)), bool), lambda e: f(e[:, :1]),
                       np.array([[omega]]), ErrorModel("proportional"), [sig], np.zeros((1, 1)))
        assert res.ofv == pytest.approx(exact, abs=0.1)


TRUE = PopModel(
    n_cmt=1,
    params=(
        Param("CL", 4.0, 0.09, covariate="WT", reference=70, exponent=0.75),
        Param("V", 40.0, 0.04, covariate="WT", reference=70, exponent=1.0),
        Param("KA", 1.2, 0.16),
    ),
    error=ErrorModel("proportional"),
    sigma=(0.12,),
)
SAMPLES = [0.5, 1, 2, 3, 4, 6, 8, 12, 24]


def study(n=40, seed=1):
    rng = np.random.default_rng(seed)
    design = [Subject(str(i + 1), [Dose(0, 100.0, "oral")], SAMPLES,
                      covariates={"WT": float(np.clip(rng.normal(72, 14), 40, 120))})
              for i in range(n)]
    return simulate(TRUE, design, seed=seed)


def start_model():
    """The true model with initial estimates deliberately off by 30-60%."""
    return PopModel(1, (Param("CL", 6.0, 0.2, "WT", 70, 0.75), Param("V", 28.0, 0.2, "WT", 70, 1.0),
                        Param("KA", 0.7, 0.2)), ErrorModel("proportional"), (0.3,))


@pytest.fixture(scope="module")
def recovered():
    return fit(start_model(), study())


@pytest.mark.slow
class TestParameterRecovery:
    def test_fixed_effects(self, recovered):
        for name, true in [("CL", 4.0), ("V", 40.0), ("KA", 1.2)]:
            assert recovered.theta[name] == pytest.approx(true, rel=0.15), name

    def test_variability_and_residual_error(self, recovered):
        assert recovered.sigma["sigma_prop"] == pytest.approx(0.12, rel=0.2)
        for name, true in [("CL", 0.09), ("V", 0.04), ("KA", 0.16)]:
            assert recovered.omega[name] == pytest.approx(true, rel=0.6), name

    def test_covariance_step(self, recovered):
        assert recovered.converged
        assert recovered.covariance_ok
        table = recovered.parameter_table().set_index("parameter")
        assert table.loc["CL", "rse_pct"] < 15
        assert table["se"].notna().all()

    def test_cwres_look_standard_normal(self, recovered):
        cwres = recovered.diagnostics["CWRES"]
        assert abs(cwres.mean()) < 0.2
        assert 0.8 < cwres.std() < 1.2

    def test_diagnostic_tables(self, recovered):
        assert len(recovered.diagnostics) == recovered.n_obs == 40 * len(SAMPLES)
        assert set(recovered.eta.columns) == {"ID", "ETA_CL", "ETA_V", "ETA_KA"}
        assert recovered.individual["CL"].gt(0).all()
        assert all(-1 < s < 1 for s in recovered.shrinkage.values())
        assert "covariance step: OK" in recovered.summary()

    def test_information_criteria(self, recovered):
        assert recovered.aic == pytest.approx(recovered.minus2ll + 2 * 7)
        assert recovered.ofv == pytest.approx(recovered.minus2ll - recovered.n_obs * LOG_2PI)


def test_simulation_is_reproducible_and_seed_dependent():
    a, b, c = study(5, seed=3), study(5, seed=3), study(5, seed=4)
    np.testing.assert_array_equal(a[0].dv, b[0].dv)
    assert not np.allclose(a[0].dv, c[0].dv)


def test_objective_is_lower_at_the_truth_than_far_from_it():
    data = study(20, seed=5)
    est = FOCEI(TRUE, data)
    at_truth = est.objective(est.initial())
    far = est.initial() + np.r_[np.log(2.0), 0, 0, 0, 0, 0, 0]
    assert est.objective(far) > at_truth + 10


class TestModelValidation:
    def test_missing_absorption_parameter(self):
        model = PopModel(1, (Param("CL", 4.0), Param("V", 40.0)))
        with pytest.raises(ValueError, match="missing parameter"):
            FOCEI(model, study(3))

    def test_parameter_the_model_does_not_use(self):
        iv = [Subject("1", [Dose(0, 100.0)], [1.0, 2.0], [2.0, 1.5])]
        model = PopModel(1, (Param("CL", 4.0), Param("V", 40.0), Param("KA", 1.0)))
        with pytest.raises(ValueError, match="not used"):
            FOCEI(model, iv)

    def test_missing_covariate(self):
        data = [Subject("1", [Dose(0, 100.0, "oral")], [1.0, 2.0], [2.0, 1.5])]
        with pytest.raises(ValueError, match="covariate WT"):
            FOCEI(TRUE, data)

    def test_needs_a_random_effect(self):
        with pytest.raises(ValueError, match="random effect"):
            PopModel(1, (Param("CL", 4.0, None), Param("V", 40.0, None)))

    def test_sigma_count_must_match_error_model(self):
        with pytest.raises(ValueError, match="sigma"):
            PopModel(1, (Param("CL", 4.0), Param("V", 40.0)), ErrorModel("combined"), (0.1,))
