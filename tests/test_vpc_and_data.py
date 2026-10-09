import hashlib

import numpy as np
import pandas as pd
import pytest
from test_estimation import TRUE, study

from pkforge.datasets import THEOPH_SHA256, THEOPH_URL, ChecksumError, fetch, theoph
from pkforge.estimation.vpc import vpc

TRUE_VALUES = (
    {p.name: p.value for p in TRUE.params},
    {p.name: p.omega for p in TRUE.iiv},
    {"sigma_prop": TRUE.sigma[0]},
)


class TestVPC:
    def test_correct_model_covers_the_data(self):
        data = study(60, seed=11)
        result = vpc(TRUE, data, *TRUE_VALUES, n_sim=150, n_bins=6, seed=1)
        assert result.coverage() >= 0.8
        assert set(result.bins.columns) >= {"t_mid", "obs_p50", "sim_p50_lo", "sim_p50_hi"}
        assert result.bins["n_obs"].sum() == len(result.observed) == 60 * 9

    def test_misspecified_model_is_exposed(self):
        data = study(60, seed=11)
        theta, omega, sigma = TRUE_VALUES
        wrong = {**theta, "CL": theta["CL"] * 2.5}
        assert vpc(TRUE, data, wrong, omega, sigma, n_sim=150, n_bins=6).coverage() < 0.5

    def test_prediction_correction_runs_and_is_reproducible(self):
        data = study(30, seed=12)
        a = vpc(TRUE, data, *TRUE_VALUES, n_sim=50, prediction_corrected=True, seed=3)
        b = vpc(TRUE, data, *TRUE_VALUES, n_sim=50, prediction_corrected=True, seed=3)
        pd.testing.assert_frame_equal(a.bins, b.bins)
        assert a.prediction_corrected


THEOPH_SAMPLE = """\
"rownames","Subject","Wt","Dose","Time","conc"
"1",1,79.6,4.02,0,0.74
"2",1,79.6,4.02,0.25,2.84
"3",1,79.6,4.02,1.12,10.5
"4",1,79.6,4.02,24.37,3.28
"5",2,72.4,4.4,0,0
"6",2,72.4,4.4,0.27,1.72
"7",2,72.4,4.4,2.03,8.33
"""


class TestTheophylline:
    def test_conversion(self, tmp_path):
        path = tmp_path / "Theoph.csv"
        path.write_text(THEOPH_SAMPLE, encoding="utf-8")
        subjects = theoph(path)
        assert [s.id for s in subjects] == ["1", "2"]
        first = subjects[0]
        assert first.doses[0].amount == pytest.approx(4.02 * 79.6)  # mg/kg x kg
        assert first.doses[0].route == "oral"
        assert first.times.tolist() == [0.25, 1.12, 24.37]  # pre-dose t = 0 sample dropped
        assert first.covariates == {"WT": 79.6}

    def test_fetch_verifies_the_checksum(self, tmp_path):
        source = tmp_path / "source.csv"
        source.write_text(THEOPH_SAMPLE, encoding="utf-8")
        good = hashlib.sha256(source.read_bytes()).hexdigest()
        url = source.as_uri()
        with pytest.raises(ChecksumError, match="refusing"):
            fetch(url, "0" * 64, "copy.csv", tmp_path / "cache")
        assert not (tmp_path / "cache" / "copy.csv").exists()
        got = fetch(url, good, "copy.csv", tmp_path / "cache")
        assert got.read_bytes() == source.read_bytes()

    @pytest.mark.network
    def test_real_download(self, tmp_path):
        path = fetch(THEOPH_URL, THEOPH_SHA256, "Theoph.csv", tmp_path)
        subjects = theoph(path)
        assert len(subjects) == 12
        assert sum(len(s.times) for s in subjects) == 120
        assert all(np.all(s.dv >= 0) for s in subjects)
