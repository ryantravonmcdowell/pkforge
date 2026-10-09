"""Read a numeric answer out of free text and diagnose it.

Labels, checked in this order:

``correct``          within the problem's relative tolerance of the answer
``misconception``    matches what a known mistake produces (``misconception``
                     names which one: forgot_ln2, forgot_accumulation, ...)
``unit_error``       right number, wrong power of ten or time unit (x1000, x60...)
``wrong_dimension``  a unit of the wrong kind (hours where mg/L was asked)
``incorrect``        none of the above
``no_answer``        no number could be read
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from pkforge.bench.problems import Problem
from pkforge.bench.units import Unit, parse_unit

_ANSWER_LINE = re.compile(
    r"^[\s>*#_`-]*(?:final\s+)?answer\s*[*_`]*\s*[:=]\s*[*_`]*\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(
    r"(?<![\w.])(?P<mant>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)"
    r"(?:\s*(?:[eE](?P<e1>[-+]?\d+)|(?:x|\*|×)\s*10\s*\^?\s*\(?(?P<e2>[-+]?\d+)\)?))?")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
UNIT_SLIPS = {"x1000": 1e3, "/1000": 1e-3, "x1e6": 1e6, "/1e6": 1e-6, "x60": 60.0,
              "/60": 1 / 60, "x24": 24.0, "/24": 1 / 24}


@dataclass(frozen=True)
class ParsedAnswer:
    value: float
    unit_text: str
    unit: Unit | None  # None when the text after the number is not a unit
    source: str  # "answer_line" or "fallback"


@dataclass(frozen=True)
class Grade:
    label: str
    misconception: str | None
    value: float | None  # converted to the problem's unit
    rel_error: float | None
    unit_text: str
    unit_assumed: bool  # no usable unit, so the expected one was assumed
    source: str  # answer_line | fallback | none

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = text.translate(_SUPERSCRIPTS)
    text = re.sub(r"\\(?:text|mathrm|mathrm\s*)\{([^}]*)\}", r"\1", text)  # LaTeX units
    text = re.sub(r"\\times", "×", text)
    text = re.sub(r"\\[ ,;:!]", " ", text)  # LaTeX spacing commands
    text = re.sub(r"[{}$]", "", text)
    text = text.replace("≈", " ").replace("~", " ").replace("\u2212", "-")
    return text


def _unit_after(rest: str) -> tuple[str, Unit | None]:
    """The longest whitespace-delimited prefix of ``rest`` that is a unit."""
    tokens = rest.strip().split()
    for k in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:k]).strip(" .,;:()[]*_`'\"")
        if not candidate:
            continue
        unit = parse_unit(candidate)
        if unit is not None and candidate.lower() not in {"x", "-"}:
            return candidate, unit
    return "", None


def _read(segment: str, source: str, last: bool) -> ParsedAnswer | None:
    matches = list(_NUMBER.finditer(segment))
    if not matches:
        return None
    m = matches[-1] if last else matches[0]
    mant = float(m.group("mant").replace(",", ""))
    exp = m.group("e1") or m.group("e2")
    value = mant * 10 ** int(exp) if exp else mant
    unit_text, unit = _unit_after(segment[m.end():])
    return ParsedAnswer(value, unit_text, unit, source)


def parse_answer(text: str) -> ParsedAnswer | None:
    """The final answer: from the last 'ANSWER:' line, else the last number in the text."""
    text = _clean(text or "")
    lines = _ANSWER_LINE.findall(text)
    if lines:
        segment = lines[-1]
        if "=" in segment:  # "C(12 h) = 4.2 mg/L": the value follows the last '='
            segment = segment.rsplit("=", 1)[1]
        parsed = _read(segment, "answer_line", last=False)
        if parsed is not None:
            return parsed
    return _read(text, "fallback", last=True)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def grade(problem: Problem, text: str) -> Grade:
    parsed = parse_answer(text)
    if parsed is None:
        return Grade("no_answer", None, None, None, "", False, "none")
    expected = parse_unit(problem.unit)
    unit, assumed = parsed.unit, False
    if unit is None or (parsed.unit_text == "" and expected.dims != (0, 0, 0)):
        # no usable unit: grade the number in the expected unit, and record that
        unit, assumed = expected, expected.dims != (0, 0, 0)
    if unit.dims != expected.dims:
        return Grade("wrong_dimension", None, None, None, parsed.unit_text, False, parsed.source)
    value = parsed.value * unit.factor / expected.factor
    rel = abs(value - problem.answer) / abs(problem.answer)
    tol = problem.tolerance
    if rel <= tol:
        return Grade("correct", None, value, rel, parsed.unit_text, assumed, parsed.source)
    for name, wrong in problem.distractors.items():
        if _close(value, wrong, tol):
            return Grade("misconception", name, value, rel, parsed.unit_text, assumed,
                         parsed.source)
    for name, factor in UNIT_SLIPS.items():
        if _close(value, problem.answer * factor, tol):
            return Grade("unit_error", name, value, rel, parsed.unit_text, assumed,
                         parsed.source)
    return Grade("incorrect", None, value, rel, parsed.unit_text, assumed, parsed.source)
