"""A small unit system for grading answers: mass, volume and time.

Every unit is a dimension vector (mass, volume, time) and a factor to the
canonical units mg, L and h, so "4.2 mg/L", "4.2 ug/mL" and "4200 ng/mL" are
recognised as the same concentration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MASS = {"kg": 1e6, "g": 1e3, "mg": 1.0, "ug": 1e-3, "mcg": 1e-3, "ng": 1e-6}
_VOLUME = {"l": 1.0, "dl": 0.1, "ml": 1e-3, "ul": 1e-6}
_TIME = {"h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
         "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
         "s": 1 / 3600, "sec": 1 / 3600, "secs": 1 / 3600, "second": 1 / 3600,
         "seconds": 1 / 3600, "d": 24.0, "day": 24.0, "days": 24.0}
_TABLES = (_MASS, _VOLUME, _TIME)
_DIMENSIONLESS = {"", "none", "unitless", "dimensionless", "(unitless)", "-", "fold", "x"}
_TOKEN = re.compile(r"([a-z]+)(?:\^?\(?(-?\d)\)?)?")


@dataclass(frozen=True)
class Unit:
    dims: tuple[int, int, int]  # exponents of mass, volume, time
    factor: float  # value in this unit x factor = value in mg, L, h


DIMENSIONLESS = Unit((0, 0, 0), 1.0)


def _normalise(text: str) -> str:
    s = text.strip().lower()
    s = s.replace("µ", "u").replace("μ", "u").replace("−", "-")
    s = s.replace("·", "*").replace("×", "*").replace("⋅", "*")
    s = re.sub(r"\s+per\s+", "/", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s


def parse_unit(text: str) -> Unit | None:
    """Parse a unit string, or return None if it is not one."""
    s = _normalise(text)
    if s in _DIMENSIONLESS:
        return DIMENSIONLESS
    if s == "%":
        return Unit((0, 0, 0), 0.01)
    dims, factor = [0, 0, 0], 1.0
    for position, segment in enumerate(s.split("/")):
        sign = 1 if position == 0 else -1
        for token in (t for t in re.split(r"[*\s.]+", segment) if t):
            if token == "1" and position == 0:
                continue  # "1/h"
            m = _TOKEN.fullmatch(token)
            if not m:
                return None
            name, power = m.group(1), sign * int(m.group(2) or 1)
            for axis, table in enumerate(_TABLES):
                if name in table:
                    dims[axis] += power
                    factor *= table[name] ** power
                    break
            else:
                return None
    return Unit((dims[0], dims[1], dims[2]), factor)


def convert(value: float, source: Unit, target: Unit) -> float:
    if source.dims != target.dims:
        raise ValueError("units have different dimensions")
    return value * source.factor / target.factor
