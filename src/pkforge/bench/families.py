"""Problem families for the PK reasoning benchmark.

Each family samples realistic, rounded inputs and writes a question, then:

* ``solve``       computes the answer from the textbook formula;
* ``verify``      recomputes it a second, independent way with the pkforge
                  engine (simulation, ODE integration, numerical root-finding,
                  NCA), so a wrong formula in the generator cannot ship;
* ``distractors`` computes what each known mistake would give (forgetting
                  ln 2, ignoring accumulation, using volume for clearance...),
                  so a wrong answer can be traced to the misconception behind it.

Answers are computed from the *displayed* (rounded) values, so a model that
works from the numbers it was shown can match them exactly.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from pkforge import nca
from pkforge.dosing import Dose, regimen
from pkforge.models import disposition_eigen, predict
from pkforge.ode import simulate_ode

LN2 = math.log(2)


def sig(x: float, digits: int = 2) -> float:
    """Round to significant figures."""
    return float(f"{x:.{digits}g}")


def fmt(x: float) -> str:
    return f"{x:g}"


class Family:
    name: ClassVar[str]
    tier: ClassVar[int]
    unit: ClassVar[str]
    summary: ClassVar[str]
    tolerance: ClassVar[float] = 0.02  # relative error accepted as correct
    verify_tolerance: ClassVar[float] = 1e-6  # agreement required between solve and verify

    def sample(self, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def prompt(self, p: dict) -> str:
        raise NotImplementedError

    def solve(self, p: dict) -> float:
        raise NotImplementedError

    def verify(self, p: dict) -> float:
        raise NotImplementedError

    def distractors(self, p: dict) -> dict[str, float]:
        return {}


def _one_cmt(cl, v):
    return {"CL": cl, "V": v}


# ---------------------------------------------------------------------- tier 1
class HalfLife(Family):
    name, tier, unit = "half_life", 1, "h"
    summary = "half-life from clearance and volume"

    def sample(self, rng):
        while True:
            p = {"CL": sig(rng.uniform(1, 20)), "V": sig(rng.uniform(10, 300))}
            if 0.5 < LN2 * p["V"] / p["CL"] < 150:
                return p

    def prompt(self, p):
        return (f"A drug has a clearance of {fmt(p['CL'])} L/h and a volume of distribution of "
                f"{fmt(p['V'])} L. Assuming one-compartment, first-order elimination, what is "
                f"its elimination half-life in hours?")

    def solve(self, p):
        return LN2 * p["V"] / p["CL"]

    def verify(self, p):
        t12 = LN2 * p["V"] / p["CL"]
        times = np.linspace(0.2, 5, 12) * t12
        conc = predict(_one_cmt(p["CL"], p["V"]), [Dose(0, 100.0)], times)
        return nca.nca(times, conc, 100.0, "iv_bolus").half_life

    def distractors(self, p):
        return {"forgot_ln2": p["V"] / p["CL"], "inverted_ratio": LN2 * p["CL"] / p["V"],
                "used_log10_of_2": math.log10(2) * p["V"] / p["CL"]}


class BolusConcentration(Family):
    name, tier, unit = "bolus_concentration", 1, "mg/L"
    summary = "concentration at time t after an IV bolus"

    def sample(self, rng):
        t12 = sig(rng.uniform(1, 24))
        return {"D": float(rng.choice([50, 100, 200, 250, 500, 1000])),
                "V": sig(rng.uniform(10, 200)), "t12": t12,
                "t": sig(t12 * rng.uniform(0.3, 4.0))}

    def prompt(self, p):
        return (f"A {fmt(p['D'])} mg IV bolus is given of a drug with a volume of distribution of "
                f"{fmt(p['V'])} L and an elimination half-life of {fmt(p['t12'])} h. Assuming "
                f"one-compartment, first-order elimination, what is the plasma concentration "
                f"{fmt(p['t'])} h after the dose, in mg/L?")

    def solve(self, p):
        return p["D"] / p["V"] * math.exp(-LN2 * p["t"] / p["t12"])

    def verify(self, p):
        cl = LN2 * p["V"] / p["t12"]
        return float(predict(_one_cmt(cl, p["V"]), [Dose(0, p["D"])], [p["t"]])[0])

    def distractors(self, p):
        c0, ratio = p["D"] / p["V"], p["t"] / p["t12"]
        out = {"forgot_ln2": c0 * math.exp(-ratio), "ignored_elimination": c0,
               "whole_half_lives_only": c0 * 0.5 ** math.floor(ratio)}
        if 1 - LN2 * ratio > 0:
            out["linear_decline"] = c0 * (1 - LN2 * ratio)
        return out


class InfusionSteadyState(Family):
    name, tier, unit = "infusion_steady_state", 1, "mg/L"
    summary = "steady-state concentration of a constant infusion"

    def sample(self, rng):
        return {"R": sig(rng.uniform(5, 200)), "CL": sig(rng.uniform(1, 20)),
                "V": sig(rng.uniform(10, 300))}

    def prompt(self, p):
        return (f"A drug is given by continuous IV infusion at {fmt(p['R'])} mg/h. Its clearance "
                f"is {fmt(p['CL'])} L/h and its volume of distribution is {fmt(p['V'])} L. What "
                f"is the steady-state plasma concentration in mg/L?")

    def solve(self, p):
        return p["R"] / p["CL"]

    def verify(self, p):
        t12 = LN2 * p["V"] / p["CL"]
        dur = 60 * t12  # effectively infinite: within exp(-41) of steady state
        dose = Dose(0, p["R"] * dur, "infusion", dur)
        return float(predict(_one_cmt(p["CL"], p["V"]), [dose], [dur])[0])

    def distractors(self, p):
        return {"used_volume": p["R"] / p["V"], "multiplied_by_clearance": p["R"] * p["CL"],
                "divided_by_rate_constant": p["R"] * p["V"] / p["CL"]}


class LoadingDose(Family):
    name, tier, unit = "loading_dose", 1, "mg"
    summary = "IV loading dose for a target concentration"

    def sample(self, rng):
        return {"C": sig(rng.uniform(1, 50)), "V": sig(rng.uniform(10, 300)),
                "CL": sig(rng.uniform(1, 20))}

    def prompt(self, p):
        return (f"A drug has a volume of distribution of {fmt(p['V'])} L and a clearance of "
                f"{fmt(p['CL'])} L/h. What IV bolus loading dose, in mg, immediately produces a "
                f"plasma concentration of {fmt(p['C'])} mg/L?")

    def solve(self, p):
        return p["C"] * p["V"]

    def verify(self, p):
        dose = p["C"] * p["V"]
        c = predict(_one_cmt(p["CL"], p["V"]), [Dose(0, dose)], [1e-9])[0]
        return dose * p["C"] / c  # the dose that makes the immediate concentration hit C

    def distractors(self, p):
        return {"used_clearance": p["C"] * p["CL"], "divided_by_volume": p["C"] / p["V"]}


# ---------------------------------------------------------------------- tier 2
class TimeToThreshold(Family):
    name, tier, unit = "time_to_threshold", 2, "h"
    summary = "time for a concentration to fall to a threshold"

    def sample(self, rng):
        c0 = sig(rng.uniform(5, 100))
        return {"C0": c0, "t12": sig(rng.uniform(1, 24)),
                "Ct": sig(c0 * rng.uniform(0.03, 0.6))}

    def prompt(self, p):
        return (f"Immediately after an IV bolus the plasma concentration of a drug is "
                f"{fmt(p['C0'])} mg/L. Its elimination half-life is {fmt(p['t12'])} h "
                f"(one-compartment, first-order elimination). How many hours after the dose does "
                f"the concentration fall to {fmt(p['Ct'])} mg/L?")

    def solve(self, p):
        return p["t12"] * math.log2(p["C0"] / p["Ct"])

    def verify(self, p):
        params = _one_cmt(LN2 / p["t12"], 1.0)  # V = 1 L, so the dose equals C0

        def gap(t):
            return predict(params, [Dose(0, p["C0"])], [t])[0] - p["Ct"]

        return brentq(gap, 1e-9, 200 * p["t12"], xtol=1e-12)

    def distractors(self, p):
        ratio, k = p["C0"] / p["Ct"], LN2 / p["t12"]
        return {"forgot_ln2": p["t12"] * math.log(ratio),
                "linear_decline": (p["C0"] - p["Ct"]) / (k * p["C0"]),
                "used_log10": math.log10(ratio) / k}


class SteadyStateAverage(Family):
    name, tier, unit = "steady_state_average", 2, "mg/L"
    summary = "average steady-state concentration of oral dosing"
    verify_tolerance = 1e-5

    def sample(self, rng):
        return {"D": float(rng.choice([25, 50, 100, 200, 250, 500])),
                "tau": float(rng.choice([6, 8, 12, 24])), "F": sig(rng.uniform(0.3, 0.95)),
                "CL": sig(rng.uniform(1, 20)), "V": sig(rng.uniform(10, 300))}

    def prompt(self, p):
        return (f"A drug is taken orally, {fmt(p['D'])} mg every {fmt(p['tau'])} h. Its oral "
                f"bioavailability is {fmt(p['F'])}, its clearance is {fmt(p['CL'])} L/h and its "
                f"volume of distribution is {fmt(p['V'])} L. What is the average plasma "
                f"concentration at steady state, in mg/L?")

    def solve(self, p):
        return p["F"] * p["D"] / (p["CL"] * p["tau"])

    def verify(self, p):
        # simulate to steady state (absorption rate is irrelevant to the average)
        params = {**_one_cmt(p["CL"], p["V"]), "KA": 1.0, "F": p["F"]}
        t12 = LN2 * p["V"] / p["CL"]
        n = int(math.ceil(40 * max(t12, LN2) / p["tau"])) + 1
        doses = regimen(p["D"], p["tau"], n, "oral")
        start = (n - 1) * p["tau"]
        area, _ = quad(lambda t: predict(params, doses, [t])[0], start, start + p["tau"],
                       epsabs=0, epsrel=1e-10, limit=200)
        return area / p["tau"]

    def distractors(self, p):
        base = p["D"] / (p["CL"] * p["tau"])
        return {"forgot_bioavailability": base, "divided_by_bioavailability": base / p["F"],
                "used_volume": p["F"] * p["D"] / (p["V"] * p["tau"])}


class SteadyStateTrough(Family):
    name, tier, unit = "steady_state_trough", 2, "mg/L"
    summary = "steady-state trough of repeated IV boluses"

    def sample(self, rng):
        while True:
            p = {"D": float(rng.choice([50, 100, 200, 250, 500, 1000])),
                 "tau": float(rng.choice([6, 8, 12, 24])), "CL": sig(rng.uniform(1, 20)),
                 "V": sig(rng.uniform(10, 300))}
            if 0.15 < p["CL"] / p["V"] * p["tau"] < 4:  # keeps troughs measurable
                return p

    def prompt(self, p):
        return (f"A drug is given as a {fmt(p['D'])} mg IV bolus every {fmt(p['tau'])} h. Its "
                f"clearance is {fmt(p['CL'])} L/h and its volume of distribution is "
                f"{fmt(p['V'])} L (one-compartment, first-order elimination). What is the trough "
                f"concentration at steady state, just before a dose, in mg/L?")

    def solve(self, p):
        x = math.exp(-p["CL"] / p["V"] * p["tau"])
        return p["D"] / p["V"] * x / (1 - x)

    def verify(self, p):
        k = p["CL"] / p["V"]
        n = int(math.ceil(40 / (k * p["tau"]))) + 2
        return float(predict(_one_cmt(p["CL"], p["V"]), regimen(p["D"], p["tau"], n),
                             [(n - 1) * p["tau"]])[0])  # pre-dose sample at the last dose

    def distractors(self, p):
        x = math.exp(-p["CL"] / p["V"] * p["tau"])
        return {"forgot_accumulation": p["D"] / p["V"] * x,
                "peak_instead_of_trough": p["D"] / p["V"] / (1 - x),
                "average_instead_of_trough": p["D"] / (p["CL"] * p["tau"])}


class AccumulationRatio(Family):
    name, tier, unit = "accumulation_ratio", 2, ""
    summary = "accumulation ratio of repeated dosing"

    def sample(self, rng):
        return {"t12": sig(rng.uniform(2, 48)), "tau": float(rng.choice([6, 8, 12, 24]))}

    def prompt(self, p):
        return (f"A drug with an elimination half-life of {fmt(p['t12'])} h is given as an IV "
                f"bolus every {fmt(p['tau'])} h. What is the accumulation ratio, the peak "
                f"concentration at steady state divided by the peak after the first dose? "
                f"(A ratio, no units.)")

    def solve(self, p):
        return 1 / (1 - 2 ** (-p["tau"] / p["t12"]))

    def verify(self, p):
        params = _one_cmt(LN2 * 10 / p["t12"], 10.0)
        n = int(math.ceil(40 * p["t12"] / p["tau"])) + 1
        eps = 1e-9
        first = predict(params, [Dose(0, 100.0)], [eps])[0]
        last = predict(params, regimen(100.0, p["tau"], n), [(n - 1) * p["tau"] + eps])[0]
        return last / first

    def distractors(self, p):
        return {"forgot_ln2": 1 / (1 - math.exp(-p["tau"] / p["t12"])),
                "inverted": 1 - 2 ** (-p["tau"] / p["t12"])}


class AUCTrapezoid(Family):
    name, tier, unit = "auc_trapezoid", 2, "mg*h/L"
    summary = "AUC from a concentration table by the linear trapezoidal rule"
    verify_tolerance = 1e-9

    def sample(self, rng):
        params = {"CL": sig(rng.uniform(2, 15)), "V": sig(rng.uniform(20, 150)),
                  "KA": sig(rng.uniform(0.5, 3))}
        # sparse sampling, as in real studies; it also keeps the wrong rules
        # (rectangles, skipped intervals) far enough from the right one to tell apart
        times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
        conc = predict(params, [Dose(0, float(rng.choice([100, 200, 500])), "oral")], times)
        return {"t": times, "C": [0.0] + [sig(c, 3) for c in conc[1:]]}

    def prompt(self, p):
        rows = "\n".join(f"  t = {fmt(t)} h: {fmt(c)} mg/L" for t, c in zip(p["t"], p["C"],
                                                                           strict=True))
        return ("Plasma concentrations after a single oral dose were:\n" + rows + "\nUsing the "
                "linear trapezoidal rule, what is the area under the concentration-time curve "
                f"from 0 to {fmt(p['t'][-1])} h, in mg*h/L?")

    def solve(self, p):
        t, c = p["t"], p["C"]
        return sum((c[i] + c[i + 1]) / 2 * (t[i + 1] - t[i]) for i in range(len(t) - 1))

    def verify(self, p):
        return nca.auc(p["t"], p["C"], method="linear")

    def distractors(self, p):
        t, c = p["t"], p["C"]
        dt = np.diff(t)
        return {"left_rectangles": float(np.dot(c[:-1], dt)),
                "right_rectangles": float(np.dot(c[1:], dt)),
                "skipped_first_interval": self.solve({"t": t[1:], "C": c[1:]}),
                "used_log_trapezoid": nca.auc(t, c, method="linear-up/log-down")}


class TerminalHalfLife(Family):
    name, tier, unit = "terminal_half_life", 2, "h"
    summary = "terminal half-life from two concentrations"
    verify_tolerance = 1e-9

    def sample(self, rng):
        t12 = rng.uniform(2, 30)
        t1 = float(rng.choice([4, 6, 8, 12]))
        t2 = t1 + float(rng.choice([6, 8, 12, 24]))
        c1 = sig(rng.uniform(2, 40), 3)
        return {"t1": t1, "C1": c1, "t2": t2, "C2": sig(c1 * 2 ** (-(t2 - t1) / t12), 3)}

    def prompt(self, p):
        return (f"In the terminal phase, the plasma concentration of a drug was "
                f"{fmt(p['C1'])} mg/L at {fmt(p['t1'])} h and {fmt(p['C2'])} mg/L at "
                f"{fmt(p['t2'])} h. Assuming log-linear decline, what is the terminal half-life "
                f"in hours?")

    def solve(self, p):
        return LN2 * (p["t2"] - p["t1"]) / math.log(p["C1"] / p["C2"])

    def verify(self, p):
        slope = np.polyfit([p["t1"], p["t2"]], np.log([p["C1"], p["C2"]]), 1)[0]
        return LN2 / -slope

    def distractors(self, p):
        dt, r = p["t2"] - p["t1"], p["C1"] / p["C2"]
        return {"forgot_ln2": dt / math.log(r),
                "linear_halving": dt * (p["C1"] / 2) / (p["C1"] - p["C2"]),
                "mixed_log_bases": LN2 * dt / math.log10(r)}


class InfusionTimeToFraction(Family):
    name, tier, unit = "infusion_time_to_fraction", 2, "h"
    summary = "time to reach a fraction of steady state during an infusion"

    def sample(self, rng):
        return {"t12": sig(rng.uniform(1, 30)), "pct": float(rng.choice([50, 75, 80, 90, 95]))}

    def prompt(self, p):
        return (f"A constant-rate IV infusion of a drug with an elimination half-life of "
                f"{fmt(p['t12'])} h is started (one-compartment, first-order elimination). How "
                f"many hours after the start does the concentration reach {fmt(p['pct'])}% of "
                f"its steady-state value?")

    def solve(self, p):
        return p["t12"] * math.log2(1 / (1 - p["pct"] / 100))

    def verify(self, p):
        params, rate = _one_cmt(LN2 * 10 / p["t12"], 10.0), 10.0
        css = rate / params["CL"]
        horizon = 100 * p["t12"]

        def gap(t):
            dose = Dose(0, rate * horizon, "infusion", horizon)
            return predict(params, [dose], [t])[0] - p["pct"] / 100 * css

        return brentq(gap, 1e-9, horizon, xtol=1e-12)

    def distractors(self, p):
        x = 1 / (1 - p["pct"] / 100)
        return {"forgot_ln2": p["t12"] * math.log(x), "used_log10": p["t12"] * math.log10(x) / LN2,
                "fraction_times_half_life": p["pct"] / 100 * p["t12"]}


# ---------------------------------------------------------------------- tier 3
class OralTmax(Family):
    name, tier, unit = "oral_tmax", 3, "h"
    summary = "time of peak concentration after an oral dose"
    verify_tolerance = 1e-5

    def sample(self, rng):
        while True:
            p = {"KA": sig(rng.uniform(0.3, 4)), "CL": sig(rng.uniform(1, 20)),
                 "V": sig(rng.uniform(10, 300))}
            if p["KA"] > 1.5 * p["CL"] / p["V"]:  # no flip-flop kinetics
                return p

    def prompt(self, p):
        return (f"After a single oral dose, a drug is absorbed with first-order rate constant "
                f"{fmt(p['KA'])} 1/h. Its clearance is {fmt(p['CL'])} L/h and its volume of "
                f"distribution is {fmt(p['V'])} L (one-compartment). At what time after the dose, "
                f"in hours, is the plasma concentration highest?")

    def solve(self, p):
        k = p["CL"] / p["V"]
        return math.log(p["KA"] / k) / (p["KA"] - k)

    def verify(self, p):
        params = {**_one_cmt(p["CL"], p["V"]), "KA": p["KA"]}
        upper = 20 / (p["CL"] / p["V"])
        res = minimize_scalar(lambda t: -predict(params, [Dose(0, 100.0, "oral")], [t])[0],
                              bounds=(1e-6, upper), method="bounded",
                              options={"xatol": 1e-10})
        return float(res.x)

    def distractors(self, p):
        k = p["CL"] / p["V"]
        return {"sum_in_denominator": math.log(p["KA"] / k) / (p["KA"] + k),
                "reciprocal_ka": 1 / p["KA"], "absorption_half_life": LN2 / p["KA"]}


class OralCmax(Family):
    name, tier, unit = "oral_cmax", 3, "mg/L"
    summary = "peak concentration after an oral dose"
    verify_tolerance = 1e-8

    def sample(self, rng):
        while True:
            p = {"D": float(rng.choice([50, 100, 200, 250, 500])), "F": sig(rng.uniform(0.3, 0.95)),
                 "KA": sig(rng.uniform(0.3, 4)), "CL": sig(rng.uniform(1, 20)),
                 "V": sig(rng.uniform(10, 300))}
            if p["KA"] > 1.5 * p["CL"] / p["V"]:
                return p

    def prompt(self, p):
        return (f"A single {fmt(p['D'])} mg oral dose is given. Bioavailability is {fmt(p['F'])}, "
                f"the absorption rate constant is {fmt(p['KA'])} 1/h, clearance is "
                f"{fmt(p['CL'])} L/h and the volume of distribution is {fmt(p['V'])} L "
                f"(one-compartment, first-order absorption and elimination). What is the peak "
                f"plasma concentration in mg/L?")

    def _tmax(self, p):
        k = p["CL"] / p["V"]
        return math.log(p["KA"] / k) / (p["KA"] - k)

    def solve(self, p):
        k, ka, tm = p["CL"] / p["V"], p["KA"], self._tmax(p)
        return p["F"] * p["D"] * ka / (p["V"] * (ka - k)) * (math.exp(-k * tm) - math.exp(-ka * tm))

    def verify(self, p):
        params = {**_one_cmt(p["CL"], p["V"]), "KA": p["KA"], "F": p["F"]}
        res = minimize_scalar(lambda t: -predict(params, [Dose(0, p["D"], "oral")], [t])[0],
                              bounds=(1e-6, 20 * p["V"] / p["CL"]), method="bounded",
                              options={"xatol": 1e-10})
        return float(-res.fun)

    def distractors(self, p):
        cmax = self.solve(p)
        return {"iv_bolus_peak": p["F"] * p["D"] / p["V"], "ignored_bioavailability": cmax / p["F"],
                "iv_bolus_peak_without_f": p["D"] / p["V"]}


class MichaelisMentenDose(Family):
    name, tier, unit = "michaelis_menten_dose", 3, "mg/day"
    summary = "daily dose for a target concentration under saturable elimination"
    verify_tolerance = 1e-4

    def sample(self, rng):
        km = sig(rng.uniform(2, 10))
        return {"VMAX": float(rng.choice([300, 400, 500, 600, 700])), "KM": km,
                "C": sig(km * rng.uniform(0.5, 2.5))}

    def prompt(self, p):
        return (f"A drug is eliminated by a saturable (Michaelis-Menten) process with Vmax = "
                f"{fmt(p['VMAX'])} mg/day and Km = {fmt(p['KM'])} mg/L. What daily dose, in "
                f"mg/day, gives an average steady-state plasma concentration of {fmt(p['C'])} "
                f"mg/L? (Assume complete bioavailability.)")

    def solve(self, p):
        return p["VMAX"] * p["C"] / (p["KM"] + p["C"])

    def verify(self, p):
        # infuse the computed dose and integrate the nonlinear ODE to steady state;
        # the returned value equals solve(p) exactly when the target is reached
        dose = self.solve(p)
        rate, vol, vmax = dose / 24, 50.0, p["VMAX"] / 24  # per hour
        # local time constant at the target: V (Km + C)^2 / (Vmax Km); run 60 of them
        horizon = 60 * vol * (p["KM"] + p["C"]) ** 2 / (vmax * p["KM"])
        conc = simulate_ode({"VMAX": vmax, "KM": p["KM"], "V": vol},
                            [Dose(0, rate * horizon, "infusion", horizon)], [horizon],
                            elimination="michaelis-menten", rtol=1e-10, atol=1e-12)[0]
        return dose * p["C"] / conc

    def distractors(self, p):
        return {"linear_clearance": p["VMAX"] / p["KM"] * p["C"],
                "swapped_km_and_c": p["VMAX"] * p["KM"] / (p["KM"] + p["C"]),
                "vmax_only": p["VMAX"]}


class TwoCompartmentTerminal(Family):
    name, tier, unit = "two_compartment_terminal", 3, "h"
    summary = "terminal half-life of a two-compartment model"
    verify_tolerance = 1e-8

    def sample(self, rng):
        return {"CL": sig(rng.uniform(1, 20)), "V1": sig(rng.uniform(5, 50)),
                "Q": sig(rng.uniform(2, 40)), "V2": sig(rng.uniform(20, 300))}

    def prompt(self, p):
        return (f"A drug follows two-compartment kinetics with clearance {fmt(p['CL'])} L/h, "
                f"central volume {fmt(p['V1'])} L, intercompartmental clearance {fmt(p['Q'])} "
                f"L/h and peripheral volume {fmt(p['V2'])} L. What is the terminal (beta-phase) "
                f"elimination half-life in hours?")

    def _rates(self, p):
        k10, k12, k21 = p["CL"] / p["V1"], p["Q"] / p["V1"], p["Q"] / p["V2"]
        a = k10 + k12 + k21
        alpha = (a + math.sqrt(a * a - 4 * k10 * k21)) / 2
        return alpha, k10 * k21 / alpha

    def solve(self, p):
        return LN2 / self._rates(p)[1]

    def verify(self, p):
        _, lambdas = disposition_eigen(p, 2, 1)
        return LN2 / float(lambdas.min())

    def distractors(self, p):
        alpha, _ = self._rates(p)
        return {"central_volume_only": LN2 * p["V1"] / p["CL"],
                "total_volume_shortcut": LN2 * (p["V1"] + p["V2"]) / p["CL"],
                "alpha_phase": LN2 / alpha}


FAMILIES: dict[str, Family] = {f.name: f for f in (
    HalfLife(), BolusConcentration(), InfusionSteadyState(), LoadingDose(),
    TimeToThreshold(), SteadyStateAverage(), SteadyStateTrough(), AccumulationRatio(),
    AUCTrapezoid(), TerminalHalfLife(), InfusionTimeToFraction(),
    OralTmax(), OralCmax(), MichaelisMentenDose(), TwoCompartmentTerminal(),
)}
