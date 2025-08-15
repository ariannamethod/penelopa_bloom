import sys
from pathlib import Path
import math
import asyncio
from datetime import datetime, timedelta
import pytest
import aiosqlite

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
    async def runner():
        db_path = tmp_path / "lines.db"
        lines_file = tmp_path / "lines.txt"
        monkeypatch.setattr(molly, "DB_PATH", db_path)
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines.clear()
        molly.user_weights.clear()
        molly.db_conn = None
        await molly.init_db()
        weight = await molly.store_line("Love 123")
        entropy, perplexity, resonance = molly.compute_metrics("Love 123")
        assert weight == pytest.approx(perplexity + resonance)
        assert molly.user_lines == ["Love 123"]
        assert molly.user_weights == [pytest.approx(weight)]
        assert lines_file.read_text(encoding="utf-8") == "Love 123\n"
        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(
                "SELECT line, entropy, perplexity, resonance FROM lines"
            )
            row = await cursor.fetchone()
            assert row[0] == "Love 123"
            assert row[1] == pytest.approx(entropy)
            assert row[2] == pytest.approx(perplexity)
            assert row[3] == pytest.approx(resonance)
        await molly.db_conn.close()
        molly.db_conn = None
    asyncio.run(runner())


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


def test_automatic_messages_preserve_state(monkeypatch):
    async def runner():
        chat_id = 123
        state = molly.ChatState()
        state.generator = iter(["hi"])
        state.last_activity = datetime.now(molly.UTC) - timedelta(
            seconds=molly.STALE_AFTER + 1
        )
        molly.chat_states[chat_id] = state

        class DummyBot:
            async def send_message(self, chat_id, text):
                pass

        class DummyApp:
            bot = DummyBot()

        async def no_typing(bot, chat_id, delay):
            pass

        async def no_store(line):
            pass

        monkeypatch.setattr(molly, "simulate_typing", no_typing)
        monkeypatch.setattr(molly, "_store_line", no_store)
        monkeypatch.setattr(
            molly, "schedule_next_message", lambda app, chat_id, state, delay=None: None
        )

        now = datetime.now(molly.UTC)
        assert (now - state.last_activity).total_seconds() > molly.STALE_AFTER

        await molly.send_chunk(DummyApp(), chat_id, state)

        now = datetime.now(molly.UTC)
        stale = [
            cid
            for cid, st in list(molly.chat_states.items())
            if (now - st.last_activity).total_seconds() > molly.STALE_AFTER
        ]
        for cid in stale:
            del molly.chat_states[cid]

        try:
            assert chat_id in molly.chat_states
        finally:
            molly.chat_states.clear()

    asyncio.run(runner())
