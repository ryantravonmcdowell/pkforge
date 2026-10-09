"""Public datasets, downloaded on demand and verified by checksum.

The theophylline data (Boeckmann, Sheiner and Beal) record plasma concentrations
in 12 subjects over 24 hours after one oral dose. They are the classic test case
for population PK software and ship with R's ``datasets`` package. pkforge does
not redistribute them: they are downloaded from the Rdatasets mirror and pinned
by SHA-256, so an altered file is refused rather than silently analysed.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pkforge.dosing import Dose, Subject

THEOPH_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv"
THEOPH_SHA256 = "f9b77033090644cf38c83056c25c8137bca71cf629d7c0159a72ef18095726b0"


class ChecksumError(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("PKFORGE_DATA", Path.home() / ".cache" / "pkforge"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, sha256: str, filename: str, directory: Path | None = None,
          timeout: float = 30.0) -> Path:
    """Return a verified local copy of ``url``, downloading it if needed."""
    directory = Path(directory) if directory else cache_dir()
    target = directory / filename
    if target.exists() and _sha256(target) == sha256:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ChecksumError(f"{url} has SHA-256 {digest}, expected {sha256}; refusing to use it")
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def theoph_frame(path: str | Path | None = None) -> pd.DataFrame:
    """The raw theophylline table: Subject, Wt (kg), Dose (mg/kg), Time (h), conc (mg/L)."""
    path = Path(path) if path else fetch(THEOPH_URL, THEOPH_SHA256, "Theoph.csv")
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.lower() == "rownames"])


def theoph(path: str | Path | None = None) -> list[Subject]:
    """Theophylline as pkforge subjects.

    Doses are converted to mg (mg/kg x body weight) so clearance comes out in
    L/h and weight is available as the covariate ``WT``. The t = 0 samples are
    pre-dose by pkforge's convention; most are non-zero (drug left from earlier
    dosing) and a single-dose model cannot explain them, so they are dropped.
    """
    df = theoph_frame(path)
    subjects = []
    for sid, rows in df.groupby("Subject", sort=False):
        rows = rows.sort_values("Time")
        wt, dose = float(rows["Wt"].iloc[0]), float(rows["Dose"].iloc[0])
        obs = rows[rows["Time"] > 0]
        subjects.append(Subject(str(sid), [Dose(0.0, dose * wt, "oral")],
                                obs["Time"].to_numpy(float), obs["conc"].to_numpy(float),
                                {"WT": wt}))
    if not subjects or any(np.isnan(s.dv).any() for s in subjects):
        raise ValueError("unexpected content in the theophylline file")
    return subjects
