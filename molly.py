import asyncio
import os
import random
import re
import sqlite3
import logging
import math
import aiosqlite
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Iterator
from functools import lru_cache
import hashlib
import subprocess
import tempfile
import shutil
import configparser

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
import numpy as np
import tiktoken
import string
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

ORIGIN_TEXT = Path('origin/molly.md')
LINES_FILE = Path('origin/logs/lines.txt')
DB_PATH = Path('origin/logs/lines.db')


def get_max_user_lines() -> int | None:
    """Return the configured maximum number of user lines."""
    _max_lines = os.getenv("MAX_USER_LINES")
    if _max_lines is None:
        return None
    if _max_lines == "" or _max_lines.lower() == "none":
        return None
    try:
        return int(_max_lines)
    except (TypeError, ValueError):  # pragma: no cover - invalid values treated as no limit
        return None


def _read_threshold_from_config() -> str | None:
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config.get("DEFAULT", "threshold_bytes", fallback=None)


CHANGELOG_DB = 'penelopa.db'
_threshold = os.getenv("THRESHOLD_BYTES") or _read_threshold_from_config()
try:
    THRESHOLD_BYTES = int(_threshold) if _threshold else 100 * 1024
except ValueError:  # pragma: no cover - invalid values treated as default
    THRESHOLD_BYTES = 100 * 1024  # 100 kilobytes
MAX_MESSAGE_LENGTH = 4096

# Sentiment analyzer and small language model for metrics
sentiment_analyzer = SentimentIntensityAnalyzer()
_char_vocab = string.ascii_lowercase + string.digits + " ?"
_char_to_idx = {c: i for i, c in enumerate(_char_vocab)}
_vocab_size = len(_char_vocab)
torch.manual_seed(0)
_bigram_weights = torch.randn(_vocab_size, _vocab_size)

# Global connection to be shared across coroutines
db_conn: aiosqlite.Connection | None = None
# Lock to serialize database access
db_lock = asyncio.Lock()
# Lock to protect in-memory line storage
lines_lock = asyncio.Lock()
# Stored user lines and their weights
user_lines: list[str] = []
user_weights: list[float] = []
# Rolling resonance stats for user lines
avg_user_resonance: float = 0.0
_resonance_samples: int = 0
# Background tasks to cancel on shutdown
background_tasks: list[asyncio.Task] = []


async def load_user_lines(max_lines: int | None = None) -> tuple[list[str], list[float]]:
    """Return previously stored user lines and weights."""
    if max_lines is None:
        max_lines = get_max_user_lines()
    if not DB_PATH.exists():
        return [], []
    lines: list[str] = []
    weights: list[float] = []
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            cursor = await conn.execute(
                'SELECT line, perplexity, resonance FROM lines ORDER BY id'
            )
            while True:
                rows = await cursor.fetchmany(1000)
                if not rows:
                    break
                for line, perp, res in rows:
                    lines.append(line)
                    weights.append((perp or 0.0) + (res or 0.0))
                if max_lines is not None and len(lines) > max_lines:
                    lines = lines[-max_lines:]
                    weights = weights[-max_lines:]
    except Exception:
        logging.exception("Failed to load user lines")
        return [], []
    return lines, weights


async def init_db() -> None:
    """Ensure the SQLite database exists and initialize global connection."""
    global db_conn
    try:
        db_conn = await aiosqlite.connect(DB_PATH)
        await db_conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line TEXT,
                entropy REAL,
                perplexity REAL,
                resonance REAL,
                created_at TEXT
            )
            '''
        )
        await db_conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS archived_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line TEXT,
                entropy REAL,
                perplexity REAL,
                resonance REAL,
                archived_at TEXT
            )
            '''
        )
        # Ensure new columns exist if database was created earlier
        cur = await db_conn.execute('PRAGMA table_info(lines)')
        cols = [c[1] for c in await cur.fetchall()]
        for col in ('entropy', 'perplexity', 'resonance'):
            if col not in cols:
                await db_conn.execute(f'ALTER TABLE lines ADD COLUMN {col} REAL')
        await db_conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_lines_created_at ON lines (created_at)'
        )
        await db_conn.commit()
    except Exception:
        logging.exception("Failed to initialize lines database")


@lru_cache(maxsize=1024)
def compute_metrics(line: str) -> tuple[float, float, float]:
    tokens = re.findall(r"\w+", line.lower())
    if not tokens:
        return 0.0, 0.0, 0.0

    ids = [_char_to_idx.get(c, _char_to_idx['?']) for c in line.lower()]
    if len(ids) < 2:
        loss = torch.tensor(0.0)
    else:
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        logits = _bigram_weights[x]
        loss = F.cross_entropy(logits, y)
    entropy = loss.item() / math.log(2) - math.log2(_vocab_size)
    perplexity = math.exp(loss.item()) / _vocab_size

    scores = sentiment_analyzer.polarity_scores(line)
    emotion_score = scores["compound"]
    num_count = sum(t.isdigit() for t in tokens)
    resonance = abs(emotion_score) + num_count
    return entropy, perplexity, resonance


async def _store_line(line: str) -> float:
    """Persist a line to the database, log file, and return its weight."""
    if db_conn is None:
        logging.error("Database not initialized")
        return 0.0
    entropy, perplexity, resonance = compute_metrics(line)
    try:
        async with db_lock:
            await db_conn.execute(
                '''
                INSERT INTO lines (
                    line, entropy, perplexity, resonance, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    line,
                    entropy,
                    perplexity,
                    resonance,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db_conn.commit()
    except Exception:
        logging.exception("Failed to store line")
        return 0.0

    def _append_line(text: str) -> None:
        LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LINES_FILE.open('a', encoding='utf-8') as f:
            f.write(text + '\n')

    await asyncio.to_thread(_append_line, line)

    weight = perplexity + resonance
    async with lines_lock:
        user_lines.append(line)
        user_weights.append(weight)
    return weight


async def store_line(line: str) -> float:
    """Asynchronous wrapper around _store_line."""
    return await _store_line(line)


async def archive_user_lines(max_lines: int | None = None) -> None:
    """Archive excess user lines to a separate file without permanent deletion."""
    if max_lines is None:
        max_lines = get_max_user_lines()
    if max_lines is None:
        return
    archive_file = LINES_FILE.with_name(f"{LINES_FILE.stem}.archive{LINES_FILE.suffix}")
    async with lines_lock:
        if len(user_lines) <= max_lines:
            return
        del user_lines[:-max_lines]
        del user_weights[:-max_lines]
    with LINES_FILE.open("r+", encoding="utf-8") as f:
        lines = f.readlines()
        old_lines = lines[:-max_lines]
        f.seek(0)
        f.writelines(lines[-max_lines:])
        f.truncate()
    if old_lines:
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        with archive_file.open("a", encoding="utf-8") as af:
            af.writelines(old_lines)
        if db_conn is not None:
            try:
                async with db_lock:
                    cursor = await db_conn.execute(
                        "SELECT id, line, entropy, perplexity, resonance FROM lines ORDER BY id LIMIT ?",
                        (len(old_lines),),
                    )
                    rows = await cursor.fetchall()
                    if rows:
                        await db_conn.executemany(
                            """
                            INSERT INTO archived_lines (
                                line, entropy, perplexity, resonance, archived_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            [
                                (
                                    line,
                                    ent,
                                    perp,
                                    res,
                                    datetime.now(UTC).isoformat(),
                                )
                                for _id, line, ent, perp, res in rows
                            ],
                        )
                        await db_conn.executemany(
                            "DELETE FROM lines WHERE id = ?",
                            [(_id,) for _id, *_ in rows],
                        )
                        await db_conn.commit()
            except Exception:
                logging.exception("Failed to archive lines to database")


async def get_random_archived_line() -> tuple[str, float] | None:
    """Return a random archived line and its resonance."""
    if db_conn is None:
        return None
    try:
        async with db_lock:
            cursor = await db_conn.execute(
                "SELECT line, resonance FROM archived_lines ORDER BY RANDOM() LIMIT 1"
            )
            row = await cursor.fetchone()
        if row:
            line, res = row
            return line, res or 0.0
    except Exception:
        logging.exception("Failed to fetch archived line")
    return None


def text_chunks() -> Iterator[str]:
    """Yield chunks up to Telegram's maximum size without splitting words."""
    buffer = ""
    with ORIGIN_TEXT.open("r", encoding="utf-8") as f:
        while True:
            data = f.read(MAX_MESSAGE_LENGTH - len(buffer))
            if not data:
                break
            buffer += data
            while len(buffer) >= MAX_MESSAGE_LENGTH:
                split_pos = buffer.rfind(" ", 0, MAX_MESSAGE_LENGTH)
                if split_pos == -1:
                    split_pos = buffer.find(" ", MAX_MESSAGE_LENGTH)
                    if split_pos == -1:
                        break
                chunk, buffer = buffer[:split_pos], buffer[split_pos + 1:]
                if chunk:
                    yield chunk
        remainder = buffer.strip()
        if remainder:
            yield remainder


def random_chunks() -> Iterator[str]:
    """Yield text chunks starting from a random byte offset.

    The file is read sequentially in blocks beginning at a random position.
    When the end of the file is reached the reader wraps to the start and
    continues until the starting offset is encountered again.
    """

    size = ORIGIN_TEXT.stat().st_size
    if size == 0:
        return

    start = random.randrange(size)
    block_size = MAX_MESSAGE_LENGTH

    def block_iter() -> Iterator[str]:
        with ORIGIN_TEXT.open("rb") as f:
            f.seek(start)
            remaining = size
            while remaining > 0:
                to_read = min(block_size, remaining)
                data = f.read(to_read)
                if not data:
                    f.seek(0)
                    continue
                remaining -= len(data)
                yield data.decode("utf-8", errors="ignore")

    buffer = ""
    for data in block_iter():
        buffer += data
        while len(buffer) >= MAX_MESSAGE_LENGTH:
            split_pos = buffer.rfind(" ", 0, MAX_MESSAGE_LENGTH)
            if split_pos == -1:
                split_pos = buffer.find(" ", MAX_MESSAGE_LENGTH)
                if split_pos == -1:
                    break
            chunk, buffer = buffer[:split_pos], buffer[split_pos + 1 :]
            if chunk:
                yield chunk
    remainder = buffer.strip()
    if remainder:
        yield remainder


async def simulate_typing(bot, chat_id: int, delay: int) -> None:
    """Show typing action for the specified delay."""
    elapsed = 0
    while elapsed < delay:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        step = min(4, delay - elapsed)
        await asyncio.sleep(step)
        elapsed += step


@dataclass
class ChatState:
    generator: Iterator[str] = field(default_factory=text_chunks)
    next_prefix: str | None = None
    next_insert_position: float | None = None
    last_activity: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    daily_target: int = field(default_factory=lambda: random.randint(8, 10))
    messages_today: int = 0
    avg_entropy: float = 0.0
    avg_perplexity: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(UTC))
    next_delay: float = 3600.0
    message_task: asyncio.Task | None = None
    awaiting_response: bool = False


chat_states: dict[int, ChatState] = {}
# Stored user lines are populated during asynchronous startup


CLEANUP_INTERVAL = 60
STALE_AFTER = 3600


def compute_delay(
    state: ChatState, entropy: float, perplexity: float, resonance: float
) -> float:
    """Return delay until next message.

    The delay grows with lower entropy, perplexity, and resonance, making Molly
    respond faster when incoming lines are more surprising or emotive.
    """
    target = max(1, state.daily_target)
    base_interval = 86400 / target
    entropy_factor = 1 + (10 - min(entropy, 10)) / 10
    perplexity_factor = 1 + 1 / (perplexity + 1)
    resonance_factor = 1 + 1 / (resonance + 1)
    return (
        base_interval
        * entropy_factor
        * perplexity_factor
        * resonance_factor
        * random.uniform(0.5, 1.5)
    )


def adjust_daily_target(state: ChatState, entropy: float, perplexity: float) -> None:
    count = state.messages_today
    state.avg_entropy = (state.avg_entropy * count + entropy) / (count + 1)
    state.avg_perplexity = (state.avg_perplexity * count + perplexity) / (count + 1)
    if state.avg_entropy < 5 and state.avg_perplexity < 30:
        state.daily_target = min(10, state.daily_target + 1)
    elif state.avg_entropy > 7 or state.avg_perplexity > 100:
        state.daily_target = max(8, state.daily_target - 1)


async def send_chunk(app: Application, chat_id: int, state: ChatState) -> None:
    try:
        try:
            chunk = next(state.generator)
        except StopIteration:
            logging.info("Generator exhausted for chat %s; restarting", chat_id)
            gen_name = getattr(state.generator, "gi_code", None)
            gen_name = gen_name.co_name if gen_name else ""
            state.generator = (
                random_chunks() if gen_name == "random_chunks" else text_chunks()
            )
            chunk = next(state.generator)
        prefix = None
        if state.next_prefix:
            prefix = state.next_prefix
            state.next_prefix = None
        elif user_lines or db_conn is not None:
            global avg_user_resonance, _resonance_samples
            if _resonance_samples == 0 and user_lines:
                res_vals = [compute_metrics(line)[2] for line in user_lines]
                if res_vals:
                    avg_user_resonance = sum(res_vals) / len(res_vals)
                    _resonance_samples = len(res_vals)
            insert_prob = 0.25 + min(avg_user_resonance / 4, 0.5)
            if random.random() < insert_prob:
                res_val = 0.0
                if db_conn is not None and (not user_lines or random.random() < 0.5):
                    archived = await get_random_archived_line()
                    if archived:
                        prefix, res_val = archived
                if prefix is None and user_lines:
                    total = sum(user_weights)
                    if total > 0:
                        prefix = random.choices(user_lines, weights=user_weights, k=1)[0]
                    else:
                        prefix = random.choice(user_lines)
                    _, _, res_val = compute_metrics(prefix)
                if prefix is not None:
                    logging.debug(
                        "Prefix resonance %.3f (avg %.3f)", res_val, avg_user_resonance
                    )
                    avg_user_resonance = (
                        avg_user_resonance * _resonance_samples + res_val
                    ) / (_resonance_samples + 1)
                    _resonance_samples += 1
        available = MAX_MESSAGE_LENGTH - (len(prefix) + 2 if prefix else 0)
        if len(chunk) > available:
            split_pos = chunk.rfind(" ", 0, available)
            if split_pos != -1:
                chunk = chunk[:split_pos]
            else:
                chunk = chunk[:available]
        if prefix:
            insert_ratio = (
                state.next_insert_position
                if state.next_insert_position is not None
                else random.random()
            )
            insert_pos = int(insert_ratio * len(chunk))
            parts = [chunk[:insert_pos].rstrip(), prefix, chunk[insert_pos:].lstrip()]
            chunk = " ".join(part for part in parts if part)
            state.next_insert_position = None
        entropy, perplexity, resonance = compute_metrics(chunk)
        delay = random.randint(3, 6) if state.awaiting_response else random.randint(1, 3)
        await simulate_typing(app.bot, chat_id, delay)
        state.awaiting_response = False
        try:
            await app.bot.send_message(chat_id=chat_id, text=chunk)
            await _store_line(chunk)
        except Exception:
            logging.exception("Failed to send chunk")
        finally:
            now = datetime.now(UTC)
            if state.last_reset.date() != now.date():
                state.last_reset = now
                state.messages_today = 0
                state.daily_target = random.randint(8, 10)
                state.avg_entropy = 0.0
                state.avg_perplexity = 0.0
            adjust_daily_target(state, entropy, perplexity)
            state.messages_today += 1
            if state.messages_today >= state.daily_target:
                tomorrow = datetime.combine((now + timedelta(days=1)).date(), time.min, tzinfo=UTC)
                state.next_delay = (tomorrow - now).total_seconds()
            else:
                state.next_delay = compute_delay(state, entropy, perplexity, resonance)
            schedule_next_message(app, chat_id, state)
    except Exception:
        logging.exception("Failed to send chunk")
        schedule_next_message(app, chat_id, state)


async def _delayed_message(app: Application, chat_id: int, state: ChatState, delay: float) -> None:
    try:
        await asyncio.sleep(delay)
        await send_chunk(app, chat_id, state)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logging.error("Error in delayed message: %s", exc)
        schedule_next_message(app, chat_id, state)


def schedule_next_message(app: Application, chat_id: int, state: ChatState, delay: float | None = None) -> None:
    if delay is None:
        delay = state.next_delay
    if state.message_task:
        state.message_task.cancel()
    state.message_task = asyncio.create_task(_delayed_message(app, chat_id, state, delay))


async def cleanup_chat_states() -> None:
    while True:
        now = datetime.now(UTC)
        stale = [
            chat_id
            for chat_id, state in list(chat_states.items())
            if (now - state.last_activity).total_seconds() > STALE_AFTER
        ]
        for chat_id in stale:
            state = chat_states[chat_id]
            if state.message_task:
                state.message_task.cancel()
                try:
                    await state.message_task
                except asyncio.CancelledError:
                    pass
            del chat_states[chat_id]
        await asyncio.sleep(CLEANUP_INTERVAL)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unexpected errors during update handling."""
    logging.exception("Unhandled exception", exc_info=context.error)

async def monologue(app: Application, chat_id: int) -> None:
    state = chat_states.setdefault(chat_id, ChatState())
    schedule_next_message(app, chat_id, state)


async def startup(app: Application) -> None:
    await init_db()
    global user_lines, user_weights
    user_lines, user_weights = await load_user_lines()
    background_tasks.append(asyncio.create_task(cleanup_chat_states()))
    background_tasks.append(asyncio.create_task(monitor_repo()))


async def shutdown(app: Application) -> None:
    for task in background_tasks:
        task.cancel()
    try:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    finally:
        if db_conn is not None:
            await db_conn.close()


def split_fragments(
    text: str,
    *,
    entropy_threshold: float = 2.0,
    perplexity_threshold: float = 4.0,
) -> list[str]:
    """Return all meaningful fragments from user text.

    The text is first split on punctuation. Each resulting piece is then
    further segmented so that the entropy or perplexity of a fragment never
    exceeds the supplied thresholds. Before returning, ``compute_metrics`` is
    executed for every produced fragment to ensure metrics are calculated.
    """

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    fragments: list[str] = []

    for line in raw_lines:
        parts = re.split(r"[.!?]+", line)
        for part in parts:
            cleaned = re.sub(r"[^\w\s]", "", part).strip()
            if not cleaned:
                continue
            words = cleaned.split()
            current: list[str] = []
            for word in words:
                candidate = " ".join(current + [word])
                entropy, perplexity, _ = compute_metrics(candidate)
                if (
                    entropy > entropy_threshold
                    or perplexity > perplexity_threshold
                ):
                    if current:
                        fragments.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                fragments.append(" ".join(current))

    if not fragments and raw_lines:
        for line in raw_lines:
            words = line.split()
            chunk = random.randint(6, 12)
            for i in range(0, len(words), chunk):
                frag = " ".join(words[i : i + chunk]).strip()
                if frag:
                    fragments.append(frag)

    for frag in fragments:
        compute_metrics(frag)

    return fragments


def select_prefix_fragments(fragments: list[str]) -> list[tuple[str, float]]:
    """Pick the most resonant fragments to use as a prefix."""
    if not fragments:
        return []
    scored = [(line, compute_metrics(line)) for line in fragments]
    scored.sort(key=lambda x: x[1][1] + x[1][2], reverse=True)
    lines_count = 2 if len(scored) <= 2 else random.randint(2, 3)
    return [
        (line, metrics[1] + metrics[2]) for line, metrics in scored[:lines_count]
    ]


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    text = update.message.text or ''
    fragments = split_fragments(text)
    if not fragments:
        return
    logging.debug("fragments: %s", fragments)
    selected = select_prefix_fragments(fragments)
    for frag in fragments:
        await store_line(frag)
    max_lines = get_max_user_lines()
    if max_lines is not None and len(user_lines) > max_lines:
        # Archive old lines without dropping them permanently
        await archive_user_lines(max_lines)
    chat_id = update.effective_chat.id
    state = chat_states.setdefault(chat_id, ChatState())
    if selected:
        lines, weights = zip(*selected)
        if sum(weights) > 0:
            state.next_prefix = random.choices(lines, weights=weights, k=1)[0]
        else:
            state.next_prefix = random.choice(lines)
        state.next_insert_position = random.random()
    state.last_activity = datetime.now(UTC)
    state.awaiting_response = True
    schedule_next_message(
        context.application, chat_id, state, delay=random.uniform(1, 2)
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = chat_states.get(chat_id)
    if state is None:
        state = ChatState()
        state.generator = random_chunks()
        state.awaiting_response = True
        state.last_activity = datetime.now(UTC)
        chat_states[chat_id] = state
        await send_chunk(context.application, chat_id, state)
    else:
        state.last_activity = datetime.now(UTC)


def main() -> None:
    load_dotenv()
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        raise RuntimeError(
            'TELEGRAM_TOKEN is not set. Provide your bot token via the '
            'TELEGRAM_TOKEN environment variable or a .env file.'
        )
    app = (
        Application.builder()
        .token(token)
        .post_init(startup)
        .post_shutdown(shutdown)
        .build()
    )
    app.add_handler(CommandHandler('start', start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    app.add_error_handler(error_handler)
    app.run_polling()

# Repository change tracking and fine-tuning utilities

def get_current_commit() -> str:
    """Return the current git commit hash.

    If the repository is not available (e.g., when running from a
    deployment where the ``.git`` directory is missing), gracefully
    return an empty string instead of raising an exception.
    """
    try:
        return (
            subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            .decode('utf-8')
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Git repository not found; commit hash unavailable")
        return ""


def repo_sha256(commit_hash: str) -> str:
    """Return a SHA-256 digest for the given commit hash."""
    return hashlib.sha256(commit_hash.encode('utf-8')).hexdigest()


def get_diff(prev_commit: str, current_commit: str) -> str:
    """Return git diff between two commits."""
    try:
        if prev_commit:
            diff_cmd = ['git', 'diff', prev_commit, current_commit]
            return subprocess.check_output(diff_cmd).decode('utf-8')
        show_cmd = ['git', 'show', current_commit]
        return subprocess.check_output(show_cmd).decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logging.warning("Unable to compute git diff: %s", exc)
        return ""


def init_change_db(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT,
                repo_hash TEXT,
                diff TEXT,
                size INTEGER,
                created_at TEXT
            )
            """
        )
        conn.commit()
    except Exception:
        logging.exception("Failed to initialize database")


async def monitor_repo() -> None:
    """Watch the repository for changes and log them asynchronously.

    Blocking operations are executed in a thread to avoid stalling the
    event loop. For heavier workloads consider dedicating a separate thread
    or process.
    """
    prev_commit = ""
    conn = await asyncio.to_thread(sqlite3.connect, CHANGELOG_DB)
    await asyncio.to_thread(init_change_db, conn)
    try:
        while True:
            try:
                current_commit = get_current_commit()
                if not current_commit:
                    await asyncio.sleep(300)
                    continue
                if prev_commit and current_commit != prev_commit:
                    diff = await asyncio.to_thread(get_diff, prev_commit, current_commit)
                    repo_hash = await asyncio.to_thread(repo_sha256, current_commit)
                    await asyncio.to_thread(
                        conn.execute,
                        "INSERT INTO changes (commit_hash, repo_hash, diff, size, created_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            current_commit,
                            repo_hash,
                            diff,
                            len(diff.encode('utf-8')),
                            datetime.now(UTC).isoformat(),
                        ),
                    )
                    await asyncio.to_thread(conn.commit)
                prev_commit = current_commit
            except Exception:
                logging.exception("Failed to monitor repository changes")
            await asyncio.sleep(300)
    except asyncio.CancelledError:
        logging.info("monitor_repo task cancelled")
        raise
    finally:
        await asyncio.to_thread(conn.close)


def get_last_commit(conn: sqlite3.Connection) -> str | None:
    try:
        cur = conn.cursor()
        cur.execute('SELECT commit_hash FROM changes ORDER BY id DESC LIMIT 1')
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        logging.exception("Failed to fetch last commit")
        return None


def log_change(
    conn: sqlite3.Connection,
    commit_hash: str,
    repo_hash: str,
    diff: str,
) -> None:
    size = len(diff.encode('utf-8'))
    try:
        conn.execute(
            '''
            INSERT INTO changes (commit_hash, repo_hash, diff, size, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (commit_hash, repo_hash, diff, size, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    except Exception:
        logging.exception("Failed to log repository change")


def total_logged_size(conn: sqlite3.Connection) -> int:
    try:
        cur = conn.cursor()
        cur.execute('SELECT COALESCE(SUM(size), 0) FROM changes')
        return cur.fetchone()[0]
    except Exception:
        logging.exception("Failed to compute logged size")
        return 0


async def run_ullyses(dataset_name: str, timeout: float = 60.0) -> None:
    """Run the fine-tuning script asynchronously.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to pass to the script.
    timeout: float, optional
        Maximum number of seconds to allow the process to run.
    """

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            'ullyses.py',
            f'--dataset={dataset_name}',
        )
        try:
            await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:  # pragma: no cover - kill handled below
            process.kill()
            await process.communicate()
            raise RuntimeError('Fine-tuning timed out') from exc
        if process.returncode != 0:
            raise RuntimeError(
                f'Fine-tuning failed with code {process.returncode}'
            )
    except Exception:
        logging.exception('Fine-tune process failed')
        raise


def fine_tune() -> None:
    """Fine-tune model on the original text and accumulated diffs."""

    print("Fine-tuning triggered on Molly's monologue...")

    conn = sqlite3.connect(CHANGELOG_DB)
    cur = conn.cursor()
    cur.execute('SELECT diff FROM changes ORDER BY id')
    diffs = [row[0] for row in cur.fetchall()]
    if not diffs:
        print('No diffs available for fine-tuning.')
        conn.close()
        return

    with open(ORIGIN_TEXT, 'r', encoding='utf-8') as f:
        base_text = f.read()

    combined_text = base_text + '\n'.join(diffs)

    tmpdir = tempfile.mkdtemp(dir='data')
    try:
        input_path = os.path.join(tmpdir, 'input.txt')
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        enc = tiktoken.get_encoding('gpt2')
        ids = enc.encode_ordinary(combined_text)
        n = len(ids)
        split = int(n * 0.9)
        train_ids = np.array(ids[:split], dtype=np.uint16)
        val_ids = np.array(ids[split:], dtype=np.uint16)
        train_ids.tofile(os.path.join(tmpdir, 'train.bin'))
        val_ids.tofile(os.path.join(tmpdir, 'val.bin'))

        dataset_name = os.path.basename(tmpdir)
        asyncio.run(run_ullyses(dataset_name))

        os.makedirs(os.path.join('origin', 'logs'), exist_ok=True)
        archive = os.path.join(
            'origin', 'logs', f"{datetime.now(UTC).isoformat()}.diff"
        )
        with open(archive, 'w', encoding='utf-8') as f:
            f.write('\n'.join(diffs))

        conn.execute('DELETE FROM changes')
        conn.commit()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        conn.close()


def monitor_repo_once() -> None:
    commit = get_current_commit()
    if not commit:
        logging.warning("Skipping repo monitoring; git commit not found")
        return
    sha = repo_sha256(commit)
    try:
        with sqlite3.connect(CHANGELOG_DB) as conn:
            init_change_db(conn)
            last_commit = get_last_commit(conn)
            if commit != last_commit:
                diff = get_diff(last_commit, commit)
                log_change(conn, commit, sha, diff)
                print('Repository change detected and logged.')
            else:
                print('No repository changes detected.')

            if total_logged_size(conn) > THRESHOLD_BYTES:
                fine_tune()
    except Exception:
        logging.exception("Database operation failed")


# Transformer model used for fine-tuning

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            if self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        # block_size is always 1024 for GPT model checkpoints
        # if one wants a lower block_size it has to be done through model surgery
        # later, by calling crop_block_size()

        # create a from-scratch initialized minGPT model
        config = GPTConfig(block_size=1024, **config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == '__main__':
    main()
