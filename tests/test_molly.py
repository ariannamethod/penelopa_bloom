import sys
from pathlib import Path
import math
import asyncio
import importlib
import pytest
import aiosqlite

sys.path.append(str(Path(__file__).resolve().parents[1]))
import molly  # noqa: E402


def test_threshold_bytes_from_env(monkeypatch):
    import importlib

    monkeypatch.setenv("THRESHOLD_BYTES", "2048")
    importlib.reload(molly)
    assert molly.THRESHOLD_BYTES == 2048
    monkeypatch.delenv("THRESHOLD_BYTES", raising=False)
    importlib.reload(molly)


def test_threshold_bytes_from_config(tmp_path, monkeypatch):
    import importlib

    config_file = tmp_path / "config.ini"
    config_file.write_text("[DEFAULT]\nthreshold_bytes = 4096", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    importlib.reload(molly)
    assert molly.THRESHOLD_BYTES == 4096


def test_compute_metrics():
    line = "Love and hate 123 123"
    entropy, perplexity, resonance = molly.compute_metrics(line)
    assert math.isclose(perplexity, 2 ** entropy, rel_tol=1e-5)
    scores = molly.sentiment_analyzer.polarity_scores(line)
    expected_resonance = abs(scores["compound"]) + 2
    assert resonance == pytest.approx(expected_resonance)


def test_compute_delay_respects_daily_target(monkeypatch):
    monkeypatch.setattr(molly.random, "uniform", lambda a, b: 1)
    state = molly.ChatState()
    entropy = 5.0
    perplexity = 10.0
    state.daily_target = 8
    delay_low = molly.compute_delay(state, entropy, perplexity)
    state.daily_target = 10
    delay_high = molly.compute_delay(state, entropy, perplexity)
    assert delay_low > delay_high


def test_split_and_select(monkeypatch):
    monkeypatch.setattr(molly.random, "randint", lambda a, b: 2)
    text = "Good day! Bad night? 123 456 789."
    fragments = molly.split_fragments(text)
    assert fragments == ["Good day", "Bad night", "123 456 789"]
    selected = molly.select_prefix_fragments(fragments)
    lines = [line for line, _ in selected]
    assert lines == ["123 456 789", "Good day"]


def test_split_fragments_metrics():
    text = "one two three four five"
    fragments = molly.split_fragments(
        text, entropy_threshold=2.0, perplexity_threshold=4.0
    )
    assert fragments == ["one two three four five"]


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
        archive_file = lines_file.with_name("lines.archive.txt")
        assert molly.user_lines == ["b", "c"]
        assert molly.user_weights == [2.0, 3.0]
        assert lines_file.read_text(encoding="utf-8") == "b\nc\n"
        assert archive_file.read_text(encoding="utf-8") == "a\n"

    asyncio.run(runner())


def test_trim_user_lines_no_limit(tmp_path, monkeypatch):
    async def runner():
        lines_file = tmp_path / "lines.txt"
        lines_file.write_text("a\nb\nc\n", encoding="utf-8")
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines[:] = ["a", "b", "c"]
        molly.user_weights[:] = [1.0, 2.0, 3.0]
        await molly.trim_user_lines()
        assert molly.user_lines == ["a", "b", "c"]
        assert molly.user_weights == [1.0, 2.0, 3.0]
        assert lines_file.read_text(encoding="utf-8") == "a\nb\nc\n"

    asyncio.run(runner())


def test_trim_user_lines_archives(tmp_path, monkeypatch):
    async def runner():
        lines_file = tmp_path / "lines.txt"
        content = "\n".join(str(i) for i in range(100)) + "\n"
        lines_file.write_text(content, encoding="utf-8")
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines[:] = [str(i) for i in range(100)]
        molly.user_weights[:] = [float(i) for i in range(100)]
        await molly.trim_user_lines(max_lines=10)
        archive_file = lines_file.with_name("lines.archive.txt")
        assert molly.user_lines == [str(i) for i in range(90, 100)]
        assert archive_file.read_text(encoding="utf-8").splitlines() == [str(i) for i in range(90)]
        assert lines_file.read_text(encoding="utf-8").splitlines() == [str(i) for i in range(90, 100)]

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


def test_load_user_lines_large(tmp_path, monkeypatch):
    async def runner():
        db_path = tmp_path / "lines.db"
        monkeypatch.setattr(molly, "DB_PATH", db_path)
        molly.db_conn = None
        await molly.init_db()
        async with aiosqlite.connect(db_path) as conn:
            await conn.executemany(
                "INSERT INTO lines(line, perplexity, resonance) VALUES (?, ?, ?)",
                [(f"line {i}", 0.0, 0.0) for i in range(2000)],
            )
            await conn.commit()
        lines, weights = await molly.load_user_lines(max_lines=50)
        assert len(lines) == 50
        assert lines[0] == "line 1950"
        assert all(w == 0.0 for w in weights)
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


def test_send_chunk_inserts_at_position(monkeypatch):
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

        state = molly.ChatState(generator=iter(["abcdef"]))
        state.next_prefix = "XYZ"
        state.next_insert_position = 0.5

        bot = DummyBot()
        app = DummyApp(bot)

        await molly.send_chunk(app, 1, state)

        assert bot.sent == ["abc XYZ def"]

    asyncio.run(runner())


def test_send_chunk_does_not_store_unsent(tmp_path, monkeypatch):
    class FailingBot:
        async def send_message(self, chat_id: int, text: str) -> None:
            raise RuntimeError("boom")

        async def send_chat_action(self, chat_id: int, action: object) -> None:  # pragma: no cover
            pass

    class DummyApp:
        def __init__(self, bot: FailingBot) -> None:
            self.bot = bot

    async def runner():
        db_path = tmp_path / "lines.db"
        lines_file = tmp_path / "lines.txt"
        monkeypatch.setattr(molly, "DB_PATH", db_path)
        monkeypatch.setattr(molly, "LINES_FILE", lines_file)
        molly.user_lines.clear()
        molly.user_weights.clear()
        molly.db_conn = None
        await molly.init_db()

        async def no_typing(*args, **kwargs) -> None:
            return None

        def no_schedule(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(molly, "simulate_typing", no_typing)
        monkeypatch.setattr(molly, "schedule_next_message", no_schedule)

        state = molly.ChatState(generator=iter(["hello"]))
        bot = FailingBot()
        app = DummyApp(bot)

        await molly.send_chunk(app, 1, state)

        assert not lines_file.exists()
        assert molly.user_lines == []

        await molly.db_conn.close()
        molly.db_conn = None

    asyncio.run(runner())


def test_startup_no_side_loop(tmp_path, monkeypatch):
    async def runner() -> None:
        mod = importlib.reload(molly)
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "lines.db", raising=False)

        async def fake_load() -> tuple[list[str], list[float]]:
            return ["one"], [1.0]

        async def fake_cleanup() -> None:
            pass

        async def fake_monitor() -> None:
            pass

        monkeypatch.setattr(mod, "load_user_lines", fake_load)
        monkeypatch.setattr(mod, "cleanup_chat_states", fake_cleanup)
        monkeypatch.setattr(mod, "monitor_repo", fake_monitor)
        mod.user_lines.clear()
        mod.user_weights.clear()
        await mod.startup(None)
        assert mod.user_lines == ["one"]
        assert mod.user_weights == [1.0]
        await asyncio.gather(*mod.background_tasks)
        mod.background_tasks.clear()
        if mod.db_conn is not None:
            await mod.db_conn.close()
            mod.db_conn = None

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


def test_handle_message_sets_insert_position(monkeypatch):
    async def runner():
        monkeypatch.setattr(molly, "schedule_next_message", lambda *_, **__: None)
        monkeypatch.setattr(molly.random, "choice", lambda seq: seq[0])
        monkeypatch.setattr(molly.random, "choices", lambda seq, weights, k=1: [seq[0]])
        monkeypatch.setattr(molly.random, "random", lambda: 0.3)

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

        update = DummyUpdate("Hello")
        context = DummyContext()

        await molly.handle_message(update, context)

        state = molly.chat_states[1]
        assert state.next_prefix == "Hello"
        assert state.next_insert_position == 0.3

        molly.chat_states.clear()

    asyncio.run(runner())


def test_run_ullyses_nonblocking(monkeypatch):
    order: list[str] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        class Proc:
            returncode = 0

            async def communicate(self):
                order.append('proc_start')
                await asyncio.sleep(0.05)
                order.append('proc_end')
                return b'', b''

            def kill(self) -> None:  # pragma: no cover
                pass

        return Proc()

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)

    async def side_task():
        await asyncio.sleep(0.01)
        order.append('side')

    async def runner():
        await asyncio.gather(molly.run_ullyses('dummy'), side_task())

    asyncio.run(runner())
    assert order.index('side') < order.index('proc_end')
