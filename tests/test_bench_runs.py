import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest

from pkforge.bench import analysis, generate, save
from pkforge.bench.adapters import (
    AdapterConfigError,
    AdapterError,
    Claude,
    OpenAICompatible,
    Reply,
    SimulatedStudent,
    make_adapter,
)
from pkforge.bench.grading import grade
from pkforge.bench.runner import load_responses, run

QUIET = {"progress": lambda m: None}


@pytest.fixture(scope="module")
def problems():
    return generate(150, seed=21)


@pytest.fixture(scope="module")
def results(problems, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    sha = save(problems, tmp / "p.jsonl")
    db = tmp / "b.db"
    for spec in ("student:expert", "student:novice", "student:sloppy"):
        run(db, problems, sha, make_adapter(spec), workers=2, **QUIET)
    return load_responses(db)


def test_every_distractor_is_diagnosed_by_name(problems):
    for p in problems:
        for name, value in p.distractors.items():
            g = grade(p, f"ANSWER: {value:.6g} {p.unit}")
            assert (g.label, g.misconception) == ("misconception", name), (p.id, name)


def test_students_are_ranked_and_diagnosed(results):
    board = analysis.leaderboard(results).set_index("model")
    assert board.loc["student:expert", "accuracy"] > 0.9
    assert board.loc["student:expert", "accuracy"] > board.loc["student:sloppy", "accuracy"]
    assert board.loc["student:sloppy", "accuracy"] > board.loc["student:novice", "accuracy"]
    assert board.loc["student:novice", "misconception_rate"] > 0.25
    assert board.loc["student:sloppy", "unit_error_rate"] > 0.08
    assert board.loc["student:sloppy", "format_ok"] < 1.0  # sometimes ignores the format
    assert board["ci_low"].lt(board["accuracy"]).all()


def test_analysis_tables(results):
    tests = analysis.paired_tests(results)
    assert len(tests) == 2 and tests["p_value"].lt(0.05).all()
    assert not analysis.misconceptions(results).empty
    assert analysis.outcome_table(results).sum(axis=1).round(6).eq(1).all()
    assert set(analysis.by_family(results).columns) == {"student:expert", "student:novice",
                                                       "student:sloppy"}


def test_wilson_interval():
    lo, hi = analysis.wilson_interval(8, 10)
    assert (lo, hi) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert analysis.wilson_interval(0, 10)[0] == 0.0


def test_mcnemar_uses_only_discordant_pairs():
    rows = [("a", f"q{i}", "correct") for i in range(10)]
    rows += [("b", f"q{i}", "correct" if i < 9 else "incorrect") for i in range(10)]
    df = pd.DataFrame(rows, columns=["adapter", "problem_id", "label"])
    t = analysis.paired_tests(df).iloc[0]
    assert (t.only_first_right, t.only_second_right) == (1, 0)
    assert t.p_value == pytest.approx(1.0)  # one discordant pair proves nothing


def test_resume_skips_answered(problems, tmp_path):
    sha = save(problems, tmp_path / "p.jsonl")
    first = run(tmp_path / "b.db", problems[:40], sha, SimulatedStudent("expert"), **QUIET)
    again = run(tmp_path / "b.db", problems, sha, SimulatedStudent("expert"),
                resume_run_id=first.run_id, **QUIET)
    assert (again.answered, again.skipped) == (len(problems) - 40, 40)


class Flaky:
    name = "flaky"

    def __init__(self, fatal=False):
        self.fatal = fatal

    def answer(self, problem):
        if self.fatal:
            raise AdapterConfigError("bad key")
        if problem.id.endswith(("0", "1")):
            raise AdapterError("HTTP 503")
        return Reply(f"ANSWER: {problem.answer} {problem.unit}", "flaky-1")


def test_failed_calls_are_recorded_and_config_errors_abort(problems, tmp_path):
    sha = save(problems, tmp_path / "p.jsonl")
    summary = run(tmp_path / "b.db", problems, sha, Flaky(), **QUIET)
    assert summary.failed > 0 and summary.accuracy == 1.0
    with pytest.raises(AdapterConfigError):
        run(tmp_path / "c.db", problems, sha, Flaky(fatal=True), workers=1, **QUIET)


class TestClaudeAdapter:
    def client(self, stop="end_turn"):
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text="ANSWER: 1 h")],
            stop_reason=stop, model="claude-test",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        calls = []
        return SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kw: calls.append(kw) or resp)), calls

    def test_request_and_reply(self, problems):
        client, calls = self.client()
        reply = Claude("claude-test", effort="low", client=client).answer(problems[0])
        assert reply.text == "ANSWER: 1 h" and reply.output_tokens == 5
        sent = calls[0]
        assert sent["output_config"] == {"effort": "low"}
        assert "temperature" not in sent and "fallbacks" not in sent
        assert "ANSWER: <number> <unit>" in sent["messages"][0]["content"]

    def test_refusal_is_no_answer(self, problems):
        client, _ = self.client(stop="refusal")
        assert Claude("claude-test", client=client).answer(problems[0]).text == ""


def test_openai_compatible_adapter_against_a_local_server(problems):
    replies = [(503, {}), (200, {"model": "m-1", "choices": [
        {"message": {"content": "ANSWER: 2 h"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3}}), (401, {})]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, body = replies.pop(0)
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        adapter = OpenAICompatible("m", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                   api_key="sk-test", max_retries=1)
        adapter_reply = adapter.answer(problems[0])  # 503, then success
        assert adapter_reply.text == "ANSWER: 2 h" and adapter_reply.output_tokens == 3
        with pytest.raises(AdapterConfigError, match="401"):
            adapter.answer(problems[0])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("spec", ["student:genius", "openai:", "nope:x"])
def test_bad_model_specs(spec):
    with pytest.raises(ValueError):
        make_adapter(spec)
