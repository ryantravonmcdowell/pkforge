"""Run a model over a problem set and store every graded reply in SQLite."""

from __future__ import annotations

import sqlite3
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pkforge.bench.adapters import AdapterConfigError, AdapterError
from pkforge.bench.grading import grade
from pkforge.bench.problems import Problem

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, adapter TEXT NOT NULL, problems_sha256 TEXT NOT NULL,
    n_problems INTEGER NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running');
CREATE TABLE IF NOT EXISTS problems (
    problems_sha256 TEXT NOT NULL, problem_id TEXT NOT NULL, family TEXT NOT NULL,
    tier INTEGER NOT NULL, answer REAL NOT NULL, unit TEXT NOT NULL,
    PRIMARY KEY (problems_sha256, problem_id));
CREATE TABLE IF NOT EXISTS responses (
    run_id TEXT NOT NULL REFERENCES runs(run_id), problem_id TEXT NOT NULL, text TEXT,
    model TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
    stop_reason TEXT, label TEXT NOT NULL, misconception TEXT, value REAL, rel_error REAL,
    unit_text TEXT, unit_assumed INTEGER, source TEXT, adapter_error TEXT,
    created_at TEXT NOT NULL, PRIMARY KEY (run_id, problem_id));
CREATE VIEW IF NOT EXISTS v_responses AS
SELECT r.*, runs.adapter, p.family, p.tier, p.answer, p.unit
FROM responses r
JOIN runs ON runs.run_id = r.run_id
JOIN problems p ON p.problem_id = r.problem_id AND p.problems_sha256 = runs.problems_sha256;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    # WAL keeps each committed reply durable (so a crashed run can resume) without
    # forcing a full disk sync per row, which costs ~50 ms each on Windows
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


@dataclass
class RunSummary:
    run_id: str
    adapter: str
    answered: int
    skipped: int
    failed: int
    accuracy: float | None


def run(db: str | Path, problems: list[Problem], problems_sha256: str, adapter, *,
        workers: int = 4, resume_run_id: str | None = None,
        progress: Callable[[str], None] | None = None) -> RunSummary:
    say = progress or (lambda m: print(m, file=sys.stderr))
    with closing(connect(db)) as conn:
        conn.executemany("INSERT OR IGNORE INTO problems VALUES (?, ?, ?, ?, ?, ?)",
                         [(problems_sha256, p.id, p.family, p.tier, p.answer, p.unit)
                          for p in problems])
        if resume_run_id:
            row = conn.execute("SELECT adapter, problems_sha256 FROM runs WHERE run_id = ?",
                               (resume_run_id,)).fetchone()
            if row is None:
                raise ValueError(f"no run {resume_run_id!r} in {db}")
            if row != (adapter.name, problems_sha256):
                raise ValueError("cannot resume: different model or problem set")
            run_id = resume_run_id
        else:
            run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
            conn.execute("INSERT INTO runs (run_id, adapter, problems_sha256, n_problems, "
                         "started_at) VALUES (?, ?, ?, ?, ?)",
                         (run_id, adapter.name, problems_sha256, len(problems), _now()))
        conn.commit()
        done = {r[0] for r in conn.execute("SELECT problem_id FROM responses WHERE run_id = ?",
                                           (run_id,))}
        todo = [p for p in problems if p.id not in done]
        say(f"[{adapter.name}] run {run_id}: {len(todo)} to answer, {len(done)} already done")

        failed = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(adapter.answer, p): p for p in todo}
            try:
                for i, fut in enumerate(as_completed(futures), start=1):
                    problem = futures[fut]
                    try:
                        reply = fut.result()
                    except AdapterConfigError:
                        raise
                    except AdapterError as exc:
                        failed += 1
                        conn.execute("INSERT OR REPLACE INTO responses (run_id, problem_id, "
                                     "label, adapter_error, created_at) VALUES (?, ?, ?, ?, ?)",
                                     (run_id, problem.id, "adapter_error", str(exc), _now()))
                        conn.commit()
                        continue
                    g = grade(problem, reply.text)
                    conn.execute(
                        "INSERT OR REPLACE INTO responses VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (run_id, problem.id, reply.text, reply.model, reply.latency_ms,
                         reply.input_tokens, reply.output_tokens, reply.stop_reason, g.label,
                         g.misconception, g.value, g.rel_error, g.unit_text, int(g.unit_assumed),
                         g.source, None, _now()))
                    conn.commit()
                    if i % 50 == 0 or i == len(todo):
                        say(f"[{adapter.name}] {i}/{len(todo)} graded")
            except AdapterConfigError:
                for f in futures:
                    f.cancel()
                conn.execute("UPDATE runs SET status = 'aborted', finished_at = ? WHERE "
                             "run_id = ?", (_now(), run_id))
                conn.commit()
                raise
        status = "complete" if not failed else "partial"
        conn.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                     (status, _now(), run_id))
        conn.commit()
        acc = conn.execute("SELECT AVG(label = 'correct') FROM responses WHERE run_id = ? AND "
                           "label != 'adapter_error'", (run_id,)).fetchone()[0]
    return RunSummary(run_id, adapter.name, len(todo), len(done), failed,
                      None if acc is None else round(acc, 4))


def load_responses(db: str | Path, run_ids: list[str] | None = None) -> pd.DataFrame:
    with closing(connect(db)) as conn:
        runs = pd.read_sql_query("SELECT * FROM runs ORDER BY started_at", conn)
        if run_ids is None:
            usable = runs[runs["status"].isin(["complete", "partial"])]
            run_ids = usable.groupby("adapter").tail(1)["run_id"].tolist()
        if not run_ids:
            return pd.DataFrame()
        marks = ", ".join("?" * len(run_ids))
        return pd.read_sql_query(f"SELECT * FROM v_responses WHERE run_id IN ({marks})", conn,
                                 params=list(run_ids))
