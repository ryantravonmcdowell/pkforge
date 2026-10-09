"""Doses, subjects, and the NONMEM-style dataset format.

Dataset columns (one row per event, as in NONMEM):

========  =====================================================================
ID        subject identifier
TIME      time since the start of the study (h)
EVID      1 = dose, 0 = observation
AMT       dose amount (mg); 0 on observation rows
RATE      infusion rate (mg/h); 0 = bolus or oral
CMT       1 = depot (oral dose), 2 = central (IV bolus or infusion)
DV        observed concentration (mg/L) on observation rows
other     numeric covariates, constant within a subject (e.g. WT)
========  =====================================================================

Convention used everywhere in pkforge: a dose affects concentrations strictly
*after* its time, so an observation recorded at a dose time is a pre-dose
(trough) sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = ("iv_bolus", "infusion", "oral")
ROUTE_CODE = {route: code for code, route in enumerate(ROUTES)}
REQUIRED_COLUMNS = ("ID", "TIME", "EVID", "AMT", "RATE", "CMT", "DV")


@dataclass(frozen=True)
class Dose:
    time: float
    amount: float
    route: str = "iv_bolus"
    duration: float = 0.0  # infusions only

    def __post_init__(self):
        if self.route not in ROUTES:
            raise ValueError(f"unknown route {self.route!r}; expected one of {ROUTES}")
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("dose time must be a finite number >= 0")
        if not self.amount > 0:
            raise ValueError("dose amount must be positive")
        if self.route == "infusion" and not self.duration > 0:
            raise ValueError("an infusion needs a positive duration")
        if self.route != "infusion" and self.duration != 0:
            raise ValueError("only infusions have a duration")

    @property
    def rate(self) -> float:
        return self.amount / self.duration if self.route == "infusion" else 0.0


def regimen(amount: float, interval: float, n_doses: int, route: str = "iv_bolus",
            start: float = 0.0, duration: float = 0.0) -> list[Dose]:
    """``n_doses`` equal doses every ``interval`` hours."""
    if n_doses < 1 or not interval > 0:
        raise ValueError("a regimen needs at least one dose and a positive interval")
    return [Dose(start + k * interval, amount, route, duration) for k in range(n_doses)]


@dataclass
class Subject:
    id: str
    doses: list[Dose]
    times: np.ndarray
    dv: np.ndarray | None = None
    covariates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        if not self.doses:
            raise ValueError(f"subject {self.id} has no doses")
        if self.times.ndim != 1 or np.any(np.diff(self.times) < 0):
            raise ValueError(f"subject {self.id}: observation times must be a sorted 1-D array")
        if self.dv is not None:
            self.dv = np.asarray(self.dv, dtype=float)
            if self.dv.shape != self.times.shape:
                raise ValueError(f"subject {self.id}: DV and TIME have different lengths")

    def with_dv(self, dv) -> Subject:
        return Subject(self.id, list(self.doses), self.times.copy(), np.asarray(dv, dtype=float),
                       dict(self.covariates))

    @property
    def total_dose(self) -> float:
        return float(sum(d.amount for d in self.doses))


def _dose_from_row(row, where: str) -> Dose:
    amt, rate, cmt = float(row.AMT), float(row.RATE), int(row.CMT)
    if cmt == 1:
        if rate:
            raise ValueError(f"{where}: an oral dose (CMT=1) cannot have a RATE")
        return Dose(float(row.TIME), amt, "oral")
    if cmt == 2:
        if rate < 0:
            raise ValueError(f"{where}: RATE must not be negative")
        if rate > 0:
            return Dose(float(row.TIME), amt, "infusion", amt / rate)
        return Dose(float(row.TIME), amt, "iv_bolus")
    raise ValueError(f"{where}: CMT must be 1 (depot) or 2 (central), got {cmt}")


def read_dataset(source: str | Path | pd.DataFrame) -> list[Subject]:
    """Read a NONMEM-style dataset (CSV path or DataFrame) into subjects."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing column(s): {', '.join(missing)}")
    covariate_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS and c != "MDV"]

    subjects = []
    for sid, rows in df.groupby("ID", sort=False):
        rows = rows.sort_values("TIME", kind="stable")
        doses, times, dv = [], [], []
        for idx, row in zip(rows.index, rows.itertuples(index=False), strict=True):
            where = f"row {idx + 2} (ID {sid})"  # +2: header line and 1-based numbering
            if int(row.EVID) == 1:
                doses.append(_dose_from_row(row, where))
            elif int(row.EVID) == 0:
                times.append(float(row.TIME))
                dv.append(float(row.DV) if pd.notna(row.DV) else np.nan)
            else:
                raise ValueError(f"{where}: EVID must be 0 or 1")
        covariates = {}
        for col in covariate_cols:
            values = pd.to_numeric(rows[col], errors="coerce").dropna().unique()
            if len(values) > 1:
                raise ValueError(f"covariate {col} changes within subject {sid}")
            if len(values) == 1:
                covariates[col] = float(values[0])
        subjects.append(Subject(str(sid), doses, np.array(times), np.array(dv), covariates))
    if not subjects:
        raise ValueError("dataset has no rows")
    return subjects


def to_frame(subjects: list[Subject]) -> pd.DataFrame:
    """Write subjects back to the NONMEM-style layout."""
    rows = []
    for order, s in enumerate(subjects):
        for d in s.doses:
            rows.append({"_order": order, "ID": s.id, "TIME": d.time, "EVID": 1, "AMT": d.amount,
                         "RATE": d.rate, "CMT": 1 if d.route == "oral" else 2, "DV": np.nan,
                         **s.covariates})
        dv = s.dv if s.dv is not None else np.full(s.times.shape, np.nan)
        for t, y in zip(s.times, dv, strict=True):
            rows.append({"_order": order, "ID": s.id, "TIME": t, "EVID": 0, "AMT": 0.0,
                         "RATE": 0.0, "CMT": 2, "DV": y, **s.covariates})
    df = pd.DataFrame(rows)
    # keep subject order; at equal times observations come first (they are pre-dose samples)
    df = df.sort_values(["_order", "TIME", "EVID"], kind="stable", ignore_index=True)
    return df.drop(columns="_order")
