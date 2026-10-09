import json

import numpy as np
import pytest

from pkforge.bench import CANARY, FAMILIES, generate, load, save
from pkforge.bench.families import LN2, HalfLife
from pkforge.bench.problems import VerificationError, _distinct, make_problem


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_family_verifies_against_the_engine(seed):
    # generate() raises VerificationError if any formula disagrees with the engine
    problems = generate(len(FAMILIES) * 12, seed=seed)
    assert {p.family for p in problems} == set(FAMILIES)
    for p in problems:
        assert p.answer > 0 and np.isfinite(p.answer)
        assert abs(p.verified - p.answer) <= FAMILIES[p.family].verify_tolerance * p.answer
        for value in p.distractors.values():
            assert abs(value - p.answer) > 3 * p.tolerance * p.answer


def test_verification_catches_a_wrong_formula():
    class Broken(HalfLife):
        def solve(self, p):
            return p["V"] / p["CL"]  # forgot ln 2

    family = Broken()
    params = family.sample(np.random.default_rng(0))
    with pytest.raises(VerificationError, match="engine gives"):
        make_problem(family, params, "broken-1")


def test_answers_use_the_displayed_numbers():
    p = generate(1, seed=4, families=["half_life"])[0]
    cl, v = p.params["CL"], p.params["V"]
    assert f"{cl:g} L/h" in p.prompt and f"{v:g} L" in p.prompt
    assert p.answer == pytest.approx(LN2 * v / cl, rel=1e-12)


def test_reproducible_and_seed_dependent():
    a = [p.to_json() for p in generate(30, seed=5)]
    b = [p.to_json() for p in generate(30, seed=5)]
    c = [p.to_json() for p in generate(30, seed=6)]
    assert a == b and a != c


def test_ids_are_unique():
    ids = [p.id for p in generate(300, seed=9)]
    assert len(ids) == len(set(ids))


def test_save_load_round_trip(tmp_path):
    problems = generate(20, seed=1)
    sha = save(problems, tmp_path / "p.jsonl")
    loaded, sha2 = load(tmp_path / "p.jsonl")
    assert loaded == problems and sha == sha2
    first = json.loads((tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["canary"] == CANARY


def test_selection_by_family_and_tier():
    assert {p.family for p in generate(10, families=["oral_tmax"])} == {"oral_tmax"}
    assert {p.tier for p in generate(30, tiers=[3])} == {3}
    with pytest.raises(ValueError):
        generate(5, families=["astrology"])


def test_distractor_filter():
    kept = _distinct(10.0, {"same": 10.3, "far": 14.0, "duplicate_of_far": 14.1, "neg": -1.0,
                            "other": 20.0}, tol=0.02)
    assert kept == {"far": 14.0, "other": 20.0}
