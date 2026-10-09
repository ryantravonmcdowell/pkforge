import numpy as np
import pytest

from pkforge.dosing import Dose, Subject
from pkforge.models import predict
from pkforge.nca import auc, lambda_z, nca, nca_subjects, summarize

D, CL, V = 100.0, 5.0, 50.0
K = CL / V
TIMES = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])


def bolus_profile(times=TIMES):
    return D / V * np.exp(-K * times)


class TestExactOnMonoexponential:
    """The log-down rule and log-linear back-extrapolation are exact for exp(-kt)."""

    def test_iv_bolus_parameters(self):
        r = nca(TIMES, bolus_profile(), D, "iv_bolus")
        assert r.lambda_z == pytest.approx(K, rel=1e-10)
        assert r.half_life == pytest.approx(np.log(2) / K, rel=1e-10)
        assert r.auc_inf == pytest.approx(D / CL, rel=1e-10)
        assert r.mrt == pytest.approx(1 / K, rel=1e-9)
        assert r.cl == pytest.approx(CL, rel=1e-10)
        assert r.vz == pytest.approx(V, rel=1e-10)
        assert r.vss == pytest.approx(V, rel=1e-9)
        assert r.n_lambda_points == len(TIMES)  # every point lies on the line
        assert r.r2_adjusted == pytest.approx(1.0)

    def test_pre_dose_sample_at_time_zero_is_ignored_for_bolus(self):
        t = np.r_[0.0, TIMES]
        c = np.r_[0.0, bolus_profile()]
        assert nca(t, c, D, "iv_bolus").auc_inf == pytest.approx(D / CL, rel=1e-10)

    def test_linear_rule_overestimates_a_declining_curve(self):
        exact = auc(TIMES, bolus_profile(), "linear-up/log-down")
        linear = auc(TIMES, bolus_profile(), "linear")
        assert linear > exact


class TestAgainstTheEngine:
    def test_oral_profile(self):
        params = {"CL": CL, "V": V, "KA": 1.5, "F": 0.8}
        t = np.r_[np.linspace(0.25, 4, 16), np.linspace(5, 72, 30)]
        c = predict(params, [Dose(0, D, "oral")], t)
        r = nca(t, c, D, "oral")
        assert r.auc_inf == pytest.approx(0.8 * D / CL, rel=5e-3)
        assert r.cl == pytest.approx(CL / 0.8, rel=5e-3)  # apparent clearance CL/F
        assert r.apparent
        # residual absorption early in the regression window biases lambda_z by ~0.2%:
        # a known property of NCA on extravascular data, not an implementation error
        assert r.lambda_z == pytest.approx(K, rel=5e-3)
        assert np.isnan(r.vss)

    def test_infusion_mrt_is_corrected_for_duration(self):
        t = np.r_[0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24, 36, 48, 72]
        c = predict({"CL": CL, "V": V}, [Dose(0, D, "infusion", 2.0)], t)
        r = nca(t, c, D, "infusion", duration=2.0)
        assert r.auc_inf == pytest.approx(D / CL, rel=5e-3)
        assert r.mrt == pytest.approx(1 / K, rel=1e-2)

    def test_lambda_z_finds_the_terminal_phase_of_a_two_compartment_profile(self):
        params = {"CL": 4.0, "V1": 20.0, "Q": 15.0, "V2": 80.0}
        t = np.array([0.25, 0.5, 1, 2, 4, 8, 12, 24, 36, 48, 72, 96])
        c = predict(params, [Dose(0, D)], t, n_cmt=2)
        r = nca(t, c, D, "iv_bolus")
        beta = np.log(2) / (np.log(2) / r.lambda_z)
        from pkforge.models import disposition
        _, lambdas = disposition(params, 2, 1)
        assert beta == pytest.approx(lambdas.min(), rel=1e-3)
        assert r.n_lambda_points < len(t)  # the distribution phase is left out


class TestEdgeCases:
    def test_trailing_zeros_are_below_quantification(self):
        c = np.r_[bolus_profile()[:-2], 0.0, 0.0]
        r = nca(TIMES, c, D, "iv_bolus")
        assert r.tlast == TIMES[-3]
        assert np.isfinite(r.auc_inf)

    def test_cmax_is_excluded_from_lambda_z_after_oral_dosing(self):
        t = np.array([1.0, 2.0, 4.0, 6.0])
        c = np.array([5.0, 4.0, 2.0, 1.0])  # Cmax first; only 3 points after it
        r = nca(t, c, D, "oral")
        assert r.n_lambda_points == 3

    def test_not_enough_terminal_points_gives_nan_not_an_error(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        c = np.array([1.0, 2.0, 3.0, 2.5])  # still rising: no terminal phase
        r = nca(t, c, D, "oral")
        assert np.isnan(r.lambda_z) and np.isnan(r.auc_inf)
        assert r.auc_last > 0

    def test_lambda_z_rejects_rising_tail(self):
        lam, r2, n = lambda_z([1, 2, 3, 4], [1, 2, 3, 4])
        assert np.isnan(lam) and n == 0

    @pytest.mark.parametrize("t, c", [([1, 2], [1, 0.5]), ([2, 1, 3], [1, 2, 1]),
                                      ([1, 2, 3], [1, -1, 0.5])])
    def test_invalid_input(self, t, c):
        with pytest.raises(ValueError):
            nca(t, c, D, "oral")


def test_subject_table_and_summary():
    subjects = [Subject(str(i), [Dose(0, D, "oral")], TIMES,
                        predict({"CL": CL * f, "V": V, "KA": 1.0}, [Dose(0, D, "oral")], TIMES))
                for i, f in enumerate([0.8, 1.0, 1.25])]
    subjects.append(Subject("multi", [Dose(0, D), Dose(12, D)], TIMES, bolus_profile()))
    table = nca_subjects(subjects)
    assert table.loc[table.ID == "multi", "note"].item().startswith("skipped")
    summary = summarize(table).set_index("parameter")
    assert summary.loc["cl", "n"] == 3
    assert summary.loc["cl", "geometric_mean"] == pytest.approx(CL, rel=0.02)
