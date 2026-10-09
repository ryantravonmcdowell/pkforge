import numpy as np
import pandas as pd
import pytest

from pkforge.dosing import Dose, Subject, read_dataset, regimen, to_frame
from pkforge.models import (
    Design,
    disposition,
    disposition_eigen,
    predict,
    predict_design,
    terminal_half_life,
)
from pkforge.ode import simulate_ode

PARAMS = {
    1: {"CL": 4.0, "V": 30.0, "KA": 1.3, "F": 0.7},
    2: {"CL": 4.0, "V1": 20.0, "Q": 6.0, "V2": 45.0, "KA": 1.3, "F": 0.7},
    3: {"CL": 4.0, "V1": 20.0, "Q2": 6.0, "V2": 45.0, "Q3": 1.5, "V3": 120.0, "KA": 1.3,
        "F": 0.7},
}
MIXED_DOSES = [
    Dose(0.0, 100.0, "iv_bolus"),
    Dose(6.0, 80.0, "infusion", 2.0),
    Dose(7.0, 40.0, "infusion", 3.0),  # overlaps the first infusion
    Dose(12.0, 150.0, "oral"),
    Dose(24.0, 150.0, "oral"),
]
GRID = np.array([0.0, 0.1, 0.5, 1, 2, 6, 6.5, 8, 9, 10, 12, 12.25, 13, 16, 24, 25, 30, 48, 72])


class TestDisposition:
    @pytest.mark.parametrize("n_cmt", [1, 2])
    def test_closed_form_matches_eigendecomposition(self, n_cmt):
        rng = np.random.default_rng(n_cmt)
        for _ in range(20):
            params = {k: rng.uniform(0.5, 50) for k in PARAMS[n_cmt]}
            c1, l1 = disposition(params, n_cmt, 1)
            c2, l2 = disposition_eigen(params, n_cmt, 1)
            order1, order2 = np.argsort(l1[0]), np.argsort(l2[0])
            np.testing.assert_allclose(l1[0][order1], l2[0][order2], rtol=1e-10)
            np.testing.assert_allclose(c1[0][order1], c2[0][order2], rtol=1e-8)

    @pytest.mark.parametrize("n_cmt", [1, 2, 3])
    def test_coefficients_sum_to_inverse_central_volume(self, n_cmt):
        coefs, lambdas = disposition(PARAMS[n_cmt], n_cmt, 1)
        v1 = PARAMS[n_cmt]["V" if n_cmt == 1 else "V1"]
        assert coefs.sum() == pytest.approx(1 / v1, rel=1e-10)
        assert np.all(lambdas > 0)

    def test_area_under_impulse_response_is_one_over_clearance(self):
        for n_cmt in (1, 2, 3):
            coefs, lambdas = disposition(PARAMS[n_cmt], n_cmt, 1)
            assert (coefs / lambdas).sum() == pytest.approx(1 / PARAMS[n_cmt]["CL"], rel=1e-10)


class TestAnalyticAgainstODE:
    @pytest.mark.parametrize("n_cmt", [1, 2, 3])
    def test_mixed_routes_and_overlapping_infusions(self, n_cmt):
        analytic = predict(PARAMS[n_cmt], MIXED_DOSES, GRID, n_cmt)
        numeric = simulate_ode(PARAMS[n_cmt], MIXED_DOSES, GRID, n_cmt)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_absorption_rate_equal_to_elimination_rate(self):
        params = {"CL": 3.0, "V": 30.0, "KA": 0.1, "F": 1.0}  # ka == k exactly
        doses = [Dose(0.0, 100.0, "oral")]
        times = np.linspace(0.5, 60, 25)
        analytic = predict(params, doses, times)
        np.testing.assert_allclose(analytic, simulate_ode(params, doses, times), rtol=1e-6)
        expected = 100 * 0.1 * times * np.exp(-0.1 * times) / 30  # the ka -> k limit
        np.testing.assert_allclose(analytic, expected, rtol=1e-9)

    def test_continuity_near_equal_rates(self):
        doses = [Dose(0.0, 100.0, "oral")]
        exact = predict({"CL": 3.0, "V": 30.0, "KA": 0.1}, doses, [5.0, 20.0])
        nearby = predict({"CL": 3.0, "V": 30.0, "KA": 0.1 * (1 + 1e-9)}, doses, [5.0, 20.0])
        np.testing.assert_allclose(exact, nearby, rtol=1e-7)

    def test_flip_flop_kinetics_do_not_overflow(self):
        params = {"CL": 30.0, "V": 10.0, "KA": 0.01}  # absorption far slower than elimination
        out = predict(params, [Dose(0.0, 100.0, "oral")], [1.0, 100.0, 5000.0])
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(
            out, simulate_ode(params, [Dose(0.0, 100.0, "oral")], [1.0, 100.0, 5000.0]),
            rtol=1e-5, atol=1e-12)

    def test_saturable_elimination_reduces_to_linear_when_km_is_large(self):
        doses = regimen(200.0, 12.0, 4)
        times = np.linspace(1, 60, 30)
        linear = predict({"CL": 5.0, "V": 40.0}, doses, times)
        mm = simulate_ode({"VMAX": 5.0e6, "KM": 1.0e6, "V": 40.0}, doses, times,
                          elimination="michaelis-menten")
        np.testing.assert_allclose(mm, linear, rtol=1e-4)

    def test_saturable_elimination_is_slower_at_high_concentration(self):
        times = [24.0]
        params = {"VMAX": 50.0, "KM": 2.0, "V": 40.0}
        low = simulate_ode(params, [Dose(0, 10.0)], times, elimination="michaelis-menten")
        high = simulate_ode(params, [Dose(0, 1000.0)], times, elimination="michaelis-menten")
        assert high[0] / 1000 > low[0] / 10  # dose-normalised exposure rises with dose


class TestDosingConventions:
    def test_observation_at_dose_time_is_pre_dose(self):
        doses = regimen(100.0, 12.0, 3)
        c = predict({"CL": 5.0, "V": 50.0}, doses, [0.0, 12.0, 24.0])
        k = 0.1
        assert c[0] == 0.0
        assert c[1] == pytest.approx(2.0 * np.exp(-k * 12))
        assert c[2] == pytest.approx(2.0 * (np.exp(-k * 12) + np.exp(-k * 24)))

    def test_superposition(self):
        params = PARAMS[2]
        a, b = [Dose(0, 100.0, "oral")], [Dose(5, 60.0, "infusion", 1.5)]
        np.testing.assert_allclose(predict(params, a + b, GRID, 2),
                                   predict(params, a, GRID, 2) + predict(params, b, GRID, 2),
                                   rtol=1e-12)

    def test_steady_state_trough_matches_textbook_formula(self):
        CL, V, D, tau = 5.0, 50.0, 100.0, 12.0
        k = CL / V
        trough = predict({"CL": CL, "V": V}, regimen(D, tau, 60), [60 * tau])[0]
        assert trough == pytest.approx(D / V * np.exp(-k * tau) / (1 - np.exp(-k * tau)),
                                       rel=1e-9)

    def test_unsorted_times_are_returned_in_input_order(self):
        params, doses = {"CL": 5.0, "V": 50.0}, [Dose(0, 100.0)]
        out = predict(params, doses, [10.0, 1.0, 5.0])
        np.testing.assert_allclose(out, 2.0 * np.exp(-0.1 * np.array([10.0, 1.0, 5.0])))

    def test_batched_design_matches_single_subject_predictions(self):
        subjects = [Subject("a", [Dose(0, 100.0, "oral")], [1.0, 2.0, 8.0]),
                    Subject("b", regimen(50.0, 8.0, 3, "infusion", duration=1.0),
                            [0.5, 4.0, 9.0, 16.5, 30.0])]
        cl, v, ka = np.array([4.0, 6.0]), np.array([30.0, 45.0]), np.array([1.0, 1.0])
        batch = predict_design({"CL": cl, "V": v, "KA": ka}, Design.from_subjects(subjects))
        for i, s in enumerate(subjects):
            single = predict({"CL": cl[i], "V": v[i], "KA": ka[i]}, s.doses, s.times)
            np.testing.assert_allclose(batch[i, : len(s.times)], single, rtol=1e-12)

    def test_terminal_half_life(self):
        assert terminal_half_life({"CL": 5.0, "V": 50.0}, 1) == pytest.approx(np.log(2) / 0.1)
        two = terminal_half_life(PARAMS[2], 2)
        assert two > np.log(2) * PARAMS[2]["V1"] / PARAMS[2]["CL"]


class TestValidation:
    def test_missing_parameter(self):
        with pytest.raises(ValueError, match="missing parameter"):
            predict({"CL": 5.0}, [Dose(0, 100.0)], [1.0])

    def test_oral_dose_needs_ka(self):
        with pytest.raises(ValueError, match="KA"):
            predict({"CL": 5.0, "V": 50.0}, [Dose(0, 100.0, "oral")], [1.0])

    def test_non_positive_parameter(self):
        with pytest.raises(ValueError, match="positive"):
            predict({"CL": -5.0, "V": 50.0}, [Dose(0, 100.0)], [1.0])

    @pytest.mark.parametrize("kwargs", [
        {"time": -1, "amount": 10}, {"time": 0, "amount": 0},
        {"time": 0, "amount": 10, "route": "infusion"},
        {"time": 0, "amount": 10, "duration": 2.0}, {"time": 0, "amount": 10, "route": "nasal"},
    ])
    def test_invalid_doses(self, kwargs):
        with pytest.raises(ValueError):
            Dose(**kwargs)


class TestDataset:
    def test_round_trip(self):
        subjects = [Subject("7", [Dose(0, 100.0, "oral"), Dose(12, 50.0, "infusion", 2.0)],
                            [1.0, 12.0, 13.0], [3.2, 1.1, 4.0], {"WT": 72.5}),
                    Subject("10", [Dose(0, 20.0)], [0.5, 2.0], [0.4, 0.3], {"WT": 60.0})]
        back = read_dataset(to_frame(subjects))
        assert [s.id for s in back] == ["7", "10"]
        assert back[0].doses == subjects[0].doses
        np.testing.assert_allclose(back[0].dv, subjects[0].dv)
        assert back[1].covariates == {"WT": 60.0}

    def test_errors_name_the_row(self):
        df = pd.DataFrame({"ID": [1, 1], "TIME": [0, 1], "EVID": [1, 0], "AMT": [100, 0],
                           "RATE": [5, 0], "CMT": [1, 2], "DV": [np.nan, 2.0]})
        with pytest.raises(ValueError, match="row 2 .*oral dose"):
            read_dataset(df)

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="missing column"):
            read_dataset(pd.DataFrame({"ID": [1], "TIME": [0]}))

    def test_time_varying_covariate_is_rejected(self):
        df = pd.DataFrame({"ID": [1, 1], "TIME": [0, 1], "EVID": [1, 0], "AMT": [100, 0],
                           "RATE": [0, 0], "CMT": [2, 2], "DV": [np.nan, 2.0], "WT": [70, 71]})
        with pytest.raises(ValueError, match="changes within subject"):
            read_dataset(df)
