"""Who answers the benchmark: simulated students, or real models behind an API.

Every adapter has ``name`` and ``answer(problem) -> Reply``.

Simulated students make the *documented* mistakes at known rates, so the
pipeline can be shown to recover them: a student who forgets ln 2 half the time
should come back labelled ``forgot_ln2`` about half the time.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pkforge.bench.problems import Problem

SYSTEM_PROMPT = ("You are solving a pharmacokinetics exam question. Work carefully, keep "
                 "units consistent, and follow the answer-format instruction exactly.")


class AdapterError(RuntimeError):
    """A model call failed after retries; the run records it and continues."""


class AdapterConfigError(AdapterError):
    """Misconfiguration (bad key, unknown model); the run stops."""


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class StudentProfile:
    p_misconception: float  # chance of using a known wrong method (tier 1; grows with tier)
    p_unit_slip: float  # chance of an off-by-1000 conversion
    p_random_error: float  # chance of an arithmetic slip
    p_no_answer_line: float  # chance of ignoring the answer format
    digits: int  # significant figures reported
    p_convert_units: float  # chance of answering in an equivalent unit (ug/mL, min...)


STUDENTS = {
    "expert": StudentProfile(0.02, 0.0, 0.01, 0.0, 4, 0.3),
    "novice": StudentProfile(0.30, 0.04, 0.05, 0.05, 3, 0.1),
    "sloppy": StudentProfile(0.05, 0.15, 0.10, 0.15, 3, 0.2),
}
_EQUIVALENTS = {  # expected unit -> (alternative unit, factor to multiply the value by)
    "mg/L": [("ug/mL", 1.0), ("ng/mL", 1000.0), ("mcg/mL", 1.0)],
    "h": [("hours", 1.0), ("min", 60.0)],
    "mg": [("g", 1e-3)],
    "mg*h/L": [("mg·h/L", 1.0), ("ug·h/mL", 1.0)],
    "mg/day": [("mg per day", 1.0), ("g/day", 1e-3)],
}


class SimulatedStudent:
    def __init__(self, profile: str = "expert", seed: int = 0):
        if profile not in STUDENTS:
            raise ValueError(f"unknown student {profile!r}; choose from {sorted(STUDENTS)}")
        self.profile, self.p, self.seed = profile, STUDENTS[profile], seed
        self.name = f"student:{profile}"

    def _rng(self, problem: Problem) -> random.Random:
        key = f"{self.seed}|{self.profile}|{problem.id}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def answer(self, problem: Problem) -> Reply:
        rng, p = self._rng(problem), self.p
        value, note = problem.answer, "Applying the standard relationship"
        p_mis = min(0.9, p.p_misconception * (1 + 0.5 * (problem.tier - 1)))
        roll = rng.random()
        if problem.distractors and roll < p_mis:
            name = rng.choice(sorted(problem.distractors))
            value, note = problem.distractors[name], "Using the relationship I remember"
        elif roll < p_mis + p.p_unit_slip:
            value = problem.answer * 1000
        elif roll < p_mis + p.p_unit_slip + p.p_random_error:
            value = problem.answer * rng.choice([0.7, 0.8, 1.25, 1.4])
        unit = problem.unit
        if unit in _EQUIVALENTS and rng.random() < p.p_convert_units:
            unit, factor = rng.choice(_EQUIVALENTS[unit])
            value *= factor
        number = f"{value:.{p.digits}g}"
        body = f"{note} for this {problem.family.replace('_', ' ')} question."
        if rng.random() < p.p_no_answer_line:
            text = f"{body} The result is approximately {number} {unit}."
        else:
            text = f"{body}\nANSWER: {number} {unit}".rstrip()
        return Reply(text=text, model=self.name)


class OpenAICompatible:
    """Any server implementing the OpenAI chat-completions API (urllib only)."""

    RETRY = frozenset({408, 409, 429, 500, 502, 503, 504})
    FATAL = frozenset({400, 401, 403, 404, 422})

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, timeout: float = 120.0, max_retries: int = 3):
        self.model, self.name = model, f"openai:{model}"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.temperature, self.timeout, self.max_retries = temperature, timeout, max_retries

    def answer(self, problem: Problem) -> Reply:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": problem.full_prompt()}]})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body.encode(),
                                     headers, method="POST")
        start, last = time.perf_counter(), None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code in self.FATAL:
                    raise AdapterConfigError(f"HTTP {exc.code} from {self.base_url}") from exc
                if exc.code not in self.RETRY:
                    raise AdapterError(f"HTTP {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            if attempt < self.max_retries:
                time.sleep(2**attempt)
        else:
            raise AdapterError(f"gave up after {self.max_retries + 1} attempts: {last}")
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("unexpected response shape") from exc
        usage = data.get("usage") or {}
        return Reply(text, data.get("model", self.model),
                     round((time.perf_counter() - start) * 1000, 1),
                     usage.get("prompt_tokens"), usage.get("completion_tokens"),
                     choice.get("finish_reason"))


class Claude:
    """Claude through the official ``anthropic`` SDK (``pip install "pkforge[anthropic]"``).

    No server-side fallback to another model: an answer must be graded under
    the name of the model that wrote it, and a refusal is graded as no answer.
    Current Claude models take ``effort`` rather than sampling parameters.
    """

    def __init__(self, model: str, effort: str | None = None, max_tokens: int = 16000,
                 max_retries: int = 3, client: Any = None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = f"anthropic:{model}" + (f"@{effort}" if effort else "")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise AdapterConfigError('install the SDK: pip install "pkforge[anthropic]"') \
                    from exc
            client = anthropic.Anthropic(max_retries=max_retries)
        self._client = client

    def answer(self, problem: Problem) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": problem.full_prompt()}]}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        start = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            try:
                import anthropic
            except ImportError:  # pragma: no cover
                raise
            if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                                anthropic.NotFoundError, anthropic.BadRequestError)):
                raise AdapterConfigError(f"{type(exc).__name__}: {exc}") from exc
            if isinstance(exc, anthropic.APIError):
                raise AdapterError(f"{type(exc).__name__}: {exc}") from exc
            raise
        text = "" if response.stop_reason == "refusal" else "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        return Reply(text, getattr(response, "model", self.model),
                     round((time.perf_counter() - start) * 1000, 1),
                     getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
                     response.stop_reason)


def make_adapter(spec: str, **kwargs):
    """``student:<profile>``, ``openai:<model>`` or ``anthropic:<model>[@effort]``."""
    kind, _, arg = spec.partition(":")
    if kind == "student":
        return SimulatedStudent(arg or "expert", **kwargs)
    if kind == "openai" and arg:
        return OpenAICompatible(arg, **kwargs)
    if kind == "anthropic" and arg:
        model, _, effort = arg.partition("@")
        return Claude(model, effort or None, **kwargs)
    raise ValueError(f"bad model spec {spec!r}: use student:<profile>, openai:<model> or "
                     f"anthropic:<model>[@effort]")
