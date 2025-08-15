import sys
from pathlib import Path
import math
import asyncio
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


def test_split_and_select(monkeypatch):
    monkeypatch.setattr(molly.random, "randint", lambda a, b: 2)
    text = "Good day! Bad night? 123 456 789."
    fragments = molly.split_fragments(text)
    assert fragments == ["Good day", "Bad night", "123 456 789"]
    selected = molly.select_prefix_fragments(fragments)
    lines = [line for line, _ in selected]
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
    async def runner():
        lines_file = tmp_path / "lines.txt"
        lines_file.write_text("a\nb\nc\n", encoding="utf-8")
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines[:] = ["a", "b", "c"]
        molly.user_weights[:] = [1.0, 2.0, 3.0]
        await molly.trim_user_lines(max_lines=2)
        assert molly.user_lines == ["b", "c"]
        assert molly.user_weights == [2.0, 3.0]
        assert lines_file.read_text(encoding="utf-8") == "b\nc\n"

    asyncio.run(runner())


def test_concurrent_store_line(tmp_path, monkeypatch):
    async def runner():
        db_path = tmp_path / "lines.db"
        lines_file = tmp_path / "lines.txt"
        monkeypatch.setattr(molly, "DB_PATH", db_path)
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines.clear()
        molly.user_weights.clear()
        molly.db_conn = None
        await molly.init_db()
        await asyncio.gather(*(molly.store_line(f"line {i}") for i in range(5)))
        assert len(molly.user_lines) == 5
        assert set(molly.user_lines) == {f"line {i}" for i in range(5)}
        assert len(molly.user_weights) == 5
        await molly.db_conn.close()
        molly.db_conn = None

    asyncio.run(runner())


def test_send_chunk_respects_limit(monkeypatch):
    class DummyBot:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_message(self, chat_id: int, text: str) -> None:
            self.sent.append(text)

        async def send_chat_action(self, chat_id: int, action: object) -> None:  # pragma: no cover
            pass

    class DummyApp:
        def __init__(self, bot: DummyBot) -> None:
            self.bot = bot

    async def runner():
        async def no_store(_: str) -> float:
            return 0.0

        async def no_typing(*args, **kwargs) -> None:
            return None

        def no_schedule(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(molly, "_store_line", no_store)
        monkeypatch.setattr(molly, "simulate_typing", no_typing)
        monkeypatch.setattr(molly, "schedule_next_message", no_schedule)

        long_chunk = "word " * 1000
        state = molly.ChatState(generator=iter([long_chunk]))
        state.next_prefix = "prefix"

        bot = DummyBot()
        app = DummyApp(bot)

        await molly.send_chunk(app, 1, state)

        assert len(bot.sent) == 1
        assert len(bot.sent[0]) <= molly.MAX_MESSAGE_LENGTH

    asyncio.run(runner())


def test_handle_message_stores_all_fragments(tmp_path, monkeypatch):
    async def runner():
        db_path = tmp_path / "lines.db"
        lines_file = tmp_path / "lines.txt"
        monkeypatch.setattr(molly, "DB_PATH", db_path)
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines.clear()
        molly.user_weights.clear()
        molly.db_conn = None
        await molly.init_db()

        monkeypatch.setattr(molly, "schedule_next_message", lambda *_, **__: None)
        monkeypatch.setattr(molly.random, "choice", lambda seq: seq[0])
        monkeypatch.setattr(molly.random, "choices", lambda seq, weights, k=1: [seq[0]])

        class DummyMessage:
            def __init__(self, text: str) -> None:
                self.text = text

        class DummyUpdate:
            def __init__(self, text: str) -> None:
                self.message = DummyMessage(text)
                self.effective_chat = type("chat", (), {"id": 1})

        class DummyContext:
            def __init__(self) -> None:
                self.application = None

        update = DummyUpdate("One. Two three! Four?")
        context = DummyContext()

        await molly.handle_message(update, context)

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("SELECT line FROM lines")
            rows = [row[0] for row in await cursor.fetchall()]
            assert set(rows) == {"One", "Two three", "Four"}

        await molly.db_conn.close()
        molly.db_conn = None
        molly.chat_states.clear()

    asyncio.run(runner())
