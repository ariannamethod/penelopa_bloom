import sys
from pathlib import Path
import math
import sqlite3
import asyncio

import asyncio
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import molly  # noqa: E402


def test_compute_metrics():
    line = "Love and hate 123 123"
    entropy, perplexity, resonance = molly.compute_metrics(line)
    probs = [0.2, 0.2, 0.2, 0.4]
    expected_entropy = -sum(p * math.log2(p) for p in probs)
    expected_perplexity = 2 ** expected_entropy
    assert math.isclose(entropy, expected_entropy, rel_tol=1e-5)
    assert math.isclose(perplexity, expected_perplexity, rel_tol=1e-5)
    assert resonance == 2


def test_prepare_lines(monkeypatch):
    monkeypatch.setattr(molly.random, "randint", lambda a, b: 2)
    text = "Good day! Bad night? 123 456 789."
    lines = molly.prepare_lines(text)
    assert lines == ["123 456 789", "Good day"]


def test_store_line(tmp_path, monkeypatch):
    db_path = tmp_path / "lines.db"
    lines_file = tmp_path / "lines.txt"
    monkeypatch.setattr(molly, "DB_PATH", db_path)
    monkeypatch.setattr(molly, "LINES_FILE", lines_file)
    molly.user_lines.clear()
    molly.user_weights.clear()
    molly.db_conn = None
    molly.init_db()
    weight = asyncio.run(molly.store_line("Love 123"))
    entropy, perplexity, resonance = molly.compute_metrics("Love 123")
    assert weight == pytest.approx(perplexity + resonance)
    assert molly.user_lines == ["Love 123"]
    assert molly.user_weights == [pytest.approx(weight)]
    assert lines_file.read_text(encoding="utf-8") == "Love 123\n"
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT line, entropy, perplexity, resonance FROM lines")
        row = cur.fetchone()
        assert row[0] == "Love 123"
        assert row[1] == pytest.approx(entropy)
        assert row[2] == pytest.approx(perplexity)
        assert row[3] == pytest.approx(resonance)
    molly.db_conn.close()
    molly.db_conn = None


def test_trim_user_lines(tmp_path, monkeypatch):
    lines_file = tmp_path / "lines.txt"
    lines_file.write_text("a\nb\nc\n", encoding="utf-8")
    monkeypatch.setattr(molly, "LINES_FILE", lines_file)
    molly.user_lines[:] = ["a", "b", "c"]
    molly.user_weights[:] = [1.0, 2.0, 3.0]
    molly.trim_user_lines(max_lines=2)
    assert molly.user_lines == ["b", "c"]
    assert molly.user_weights == [2.0, 3.0]
    assert lines_file.read_text(encoding="utf-8") == "b\nc\n"
