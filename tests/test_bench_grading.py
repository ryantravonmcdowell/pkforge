import pytest

from pkforge.bench.grading import grade, parse_answer
from pkforge.bench.problems import Problem
from pkforge.bench.units import convert, parse_unit


class TestUnits:
    @pytest.mark.parametrize("text, dims, factor", [
        ("mg/L", (1, -1, 0), 1.0), ("ug/mL", (1, -1, 0), 1.0), ("µg/mL", (1, -1, 0), 1.0),
        ("ng/mL", (1, -1, 0), 1e-3), ("mg/dL", (1, -1, 0), 10.0), ("mg·h/L", (1, -1, 1), 1.0),
        ("mg*h/L", (1, -1, 1), 1.0), ("mg h/L", (1, -1, 1), 1.0), ("L/h", (0, 1, -1), 1.0),
        ("mL/min", (0, 1, -1), 0.06), ("h", (0, 0, 1), 1.0), ("hours", (0, 0, 1), 1.0),
        ("min", (0, 0, 1), 1 / 60), ("days", (0, 0, 1), 24.0), ("1/h", (0, 0, -1), 1.0),
        ("h^-1", (0, 0, -1), 1.0), ("mg per day", (1, 0, -1), 1 / 24), ("g", (1, 0, 0), 1e3),
        ("", (0, 0, 0), 1.0),
    ])
    def test_parse(self, text, dims, factor):
        unit = parse_unit(text)
        assert unit.dims == dims
        assert unit.factor == pytest.approx(factor)

    @pytest.mark.parametrize("text", ["furlongs", "mg/L/zz", "because"])
    def test_not_a_unit(self, text):
        assert parse_unit(text) is None

    def test_convert(self):
        assert convert(4200, parse_unit("ng/mL"), parse_unit("mg/L")) == pytest.approx(4.2)
        with pytest.raises(ValueError):
            convert(1, parse_unit("h"), parse_unit("mg"))


class TestParsing:
    @pytest.mark.parametrize("text, value, unit", [
        ("so...\nANSWER: 4.2 mg/L", 4.2, "mg/L"),
        ("**ANSWER:** 4.2 mg/L", 4.2, "mg/L"),
        ("Answer = 4,200 ng/mL", 4200.0, "ng/mL"),
        ("ANSWER: 1.2 × 10^-3 g/L", 1.2e-3, "g/L"),
        ("ANSWER: 1.2×10⁻³ g/L", 1.2e-3, "g/L"),
        ("ANSWER: 3.5e1 h", 35.0, "h"),
        ("ANSWER: C(12 h) = 4.2 mg/L (rounded)", 4.2, "mg/L"),
        ("Final answer: $4.2\\ \\text{mg/L}$", 4.2, "mg/L"),
        ("ANSWER: 7.5 hours.", 7.5, "hours"),
        ("ANSWER: 1.25", 1.25, ""),
    ])
    def test_answer_line(self, text, value, unit):
        parsed = parse_answer(text)
        assert parsed.value == pytest.approx(value)
        assert parsed.unit_text == unit
        assert parsed.source == "answer_line"

    def test_last_answer_line_wins(self):
        assert parse_answer("ANSWER: 1 h\nwait, correction\nANSWER: 2 h").value == 2.0

    def test_fallback_to_last_number(self):
        parsed = parse_answer("The concentration is about 4.2 mg/L.")
        assert (parsed.value, parsed.unit_text, parsed.source) == (4.2, "mg/L", "fallback")

    def test_nothing_to_read(self):
        assert parse_answer("I am not sure.") is None
        assert parse_answer("") is None


PROBLEM = Problem(id="p1", family="bolus_concentration", tier=1, prompt="...", answer=4.2,
                  unit="mg/L", tolerance=0.02,
                  distractors={"forgot_ln2": 3.1, "ignored_elimination": 8.0}, params={},
                  verified=4.2)


class TestGrading:
    @pytest.mark.parametrize("text", ["ANSWER: 4.2 mg/L", "ANSWER: 4.25 mg/L",
                                      "ANSWER: 4200 ng/mL", "ANSWER: 4.2 ug/mL",
                                      "ANSWER: 0.42 mg/dL"])
    def test_correct_in_any_equivalent_unit(self, text):
        g = grade(PROBLEM, text)
        assert g.label == "correct" and g.value == pytest.approx(4.2, rel=0.02)

    def test_misconception_is_named(self):
        g = grade(PROBLEM, "ANSWER: 3.1 mg/L")
        assert (g.label, g.misconception) == ("misconception", "forgot_ln2")

    def test_unit_slip(self):
        g = grade(PROBLEM, "ANSWER: 4200 mg/L")  # the ng/mL number with the mg/L label
        assert (g.label, g.misconception) == ("unit_error", "x1000")

    def test_wrong_dimension(self):
        assert grade(PROBLEM, "ANSWER: 4.2 h").label == "wrong_dimension"

    def test_incorrect_and_no_answer(self):
        assert grade(PROBLEM, "ANSWER: 5.5 mg/L").label == "incorrect"
        assert grade(PROBLEM, "No idea.").label == "no_answer"

    def test_missing_unit_is_assumed_and_flagged(self):
        g = grade(PROBLEM, "ANSWER: 4.2")
        assert g.label == "correct" and g.unit_assumed

    def test_dimensionless_answer_is_not_flagged(self):
        ratio = Problem("p2", "accumulation_ratio", 2, "...", 1.5, "", 0.02, {}, {}, 1.5)
        g = grade(ratio, "ANSWER: 1.50")
        assert g.label == "correct" and not g.unit_assumed
