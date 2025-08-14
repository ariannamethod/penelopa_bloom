import asyncio
import os
import random
import re
import sqlite3
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

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

ORIGIN_TEXT = Path('origin/molly.md')
LINES_FILE = Path('origin/logs/lines.txt')
DB_PATH = Path('origin/logs/lines.db')
MAX_USER_LINES = 1000

# Global connection to be shared across threads
db_conn: sqlite3.Connection | None = None
# Stored user lines and their weights
user_lines: list[str] = []
user_weights: list[float] = []


def load_user_lines() -> tuple[list[str], list[float]]:
    """Return previously stored user lines and weights."""
    if not DB_PATH.exists():
        return [], []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                'SELECT line, perplexity, resonance FROM lines ORDER BY id'
            )
            rows = cur.fetchall()
    except Exception:
        logging.exception("Failed to load user lines")
        return [], []
    lines = [r[0] for r in rows]
    weights = [(r[1] or 0.0) + (r[2] or 0.0) for r in rows]
    if len(lines) > MAX_USER_LINES:
        lines = lines[-MAX_USER_LINES:]
        weights = weights[-MAX_USER_LINES:]
    return lines, weights


def init_db() -> None:
    """Ensure the SQLite database exists and initialize global connection."""
    global db_conn
    try:
        db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db_conn.execute(
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
        # Ensure new columns exist if database was created earlier
        cur = db_conn.cursor()
        cur.execute('PRAGMA table_info(lines)')
        cols = [c[1] for c in cur.fetchall()]
        for col in ('entropy', 'perplexity', 'resonance'):
            if col not in cols:
                cur.execute(f'ALTER TABLE lines ADD COLUMN {col} REAL')
        db_conn.commit()
    except Exception:
        logging.exception("Failed to initialize lines database")


POSITIVE_WORDS = {
    'love',
    'happy',
    'joy',
    'yes',
    'good',
    'hope',
    'dream',
}
NEGATIVE_WORDS = {
    'no',
    'sad',
    'bad',
    'fear',
    'hate',
    'dark',
}


def compute_metrics(line: str) -> tuple[float, float, float]:
    tokens = re.findall(r"\w+", line.lower())
    if not tokens:
        return 0.0, 0.0, 0.0
    total = len(tokens)
    counts = Counter(tokens)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    perplexity = 2 ** entropy
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    emotion_score = pos - neg
    num_count = sum(t.isdigit() for t in tokens)
    resonance = abs(emotion_score) + num_count
    return entropy, perplexity, resonance


def store_line(line: str) -> float:
    """Persist a line to the database, log file, and return its weight."""
    if db_conn is None:
        logging.error("Database not initialized")
        return 0.0
    entropy, perplexity, resonance = compute_metrics(line)
    try:
        db_conn.execute(
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
        db_conn.commit()
    except Exception:
        logging.exception("Failed to store line")
        return 0.0
    with LINES_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    logging.info("Stored user line: %s", line)
    weight = perplexity + resonance
    user_lines.append(line)
    user_weights.append(weight)
    return weight


def trim_user_lines(max_lines: int = MAX_USER_LINES) -> None:
    """Trim user_lines, weights, and log file to the last max_lines entries."""
    if len(user_lines) <= max_lines:
        return
    del user_lines[:-max_lines]
    del user_weights[:-max_lines]
    with LINES_FILE.open('w', encoding='utf-8') as f:
        for line in user_lines:
            f.write(line + '\n')


def text_chunks() -> Iterator[str]:
    """Yield chunks from Molly's monologue without loading it entirely."""
    buffer = ""
    with ORIGIN_TEXT.open("r", encoding="utf-8") as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            buffer += data
            while len(buffer) > 1024:
                split_pos = max(
                    buffer.rfind(" ", 0, 1024),
                    buffer.rfind("\n", 0, 1024),
                )
                if split_pos == -1:
                    # No whitespace in the first chunk, search entire buffer
                    split_pos = max(
                        buffer.rfind(" "),
                        buffer.rfind("\n"),
                    )
                    if split_pos == -1:
                        break
                chunk, buffer = (
                    buffer[:split_pos].strip(),
                    buffer[split_pos + 1:],
                )
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
    messages_since_pause: int = 0
    pause_target: int = field(default_factory=lambda: random.randint(6, 8))
    last_activity: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


chat_states: dict[int, ChatState] = {}
user_lines, user_weights = load_user_lines()

CLEANUP_INTERVAL = 60
STALE_AFTER = 3600


async def cleanup_chat_states() -> None:
    while True:
        now = datetime.now(UTC)
        stale = [
            chat_id
            for chat_id, state in list(chat_states.items())
            if (now - state.last_activity).total_seconds() > STALE_AFTER
        ]
        for chat_id in stale:
            del chat_states[chat_id]
        await asyncio.sleep(CLEANUP_INTERVAL)


async def close_db(app: Application) -> None:
    """Close the global database connection."""
    if db_conn is not None:
        db_conn.close()


async def monologue(app: Application, chat_id: int) -> None:
    state = chat_states.setdefault(chat_id, ChatState())
    async for chunk in _chunk_stream(state):
        delay = random.randint(5, 50)
        if random.random() < 0.1:
            delay = random.randint(120, 180)
        await simulate_typing(app.bot, chat_id, delay)
        await app.bot.send_message(chat_id=chat_id, text=chunk)
        state.messages_since_pause += 1
        if (
            state.messages_since_pause >= state.pause_target
            and random.random() < 0.3
        ):
            await asyncio.sleep(random.randint(3600, 7200))
            state.messages_since_pause = 0
            state.pause_target = random.randint(6, 8)


async def _chunk_stream(state: ChatState):
    for chunk in state.generator:
        prefix = None
        if state.next_prefix:
            prefix = state.next_prefix
            state.next_prefix = None
        elif user_lines and random.random() < 0.5:
            prefix = random.choices(user_lines, weights=user_weights, k=1)[0]
        yield f"{prefix} {chunk}" if prefix else chunk
        await asyncio.sleep(0)


def prepare_lines(text: str) -> list[str]:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[str] = []
    for raw in raw_lines:
        parts = re.split(r'[.!?]+', raw)
        for part in parts:
            part = part.strip()
            if part:
                segments.append(part)
    if not segments:
        return []
    if len(segments) < 2:
        words = segments[0].split()
        if len(words) > 4:
            cut = random.randint(1, len(words) - 1)
            first = ' '.join(words[:cut])
            second = ' '.join(words[cut:])
            segments = [first]
            if second:
                segments.append(second)
    scored = [
        (line, compute_metrics(line))
        for line in segments
    ]
    scored.sort(key=lambda x: x[1][1] + x[1][2], reverse=True)
    base_count = 2 if len(scored) <= 2 else random.randint(2, 3)
    lines_count = min(len(scored), base_count)
    selected = [line for line, _ in scored[:lines_count]]
    return selected


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    text = update.message.text or ''
    lines = prepare_lines(text)
    if not lines:
        return
    weights = [store_line(line) for line in lines]
    trim_user_lines()
    chat_id = update.effective_chat.id
    state = chat_states.setdefault(chat_id, ChatState())
    if weights:
        state.next_prefix = random.choices(lines, weights=weights, k=1)[0]
    else:
        state.next_prefix = random.choice(lines)
    state.last_activity = datetime.now(UTC)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id not in chat_states:
        chat_states[chat_id] = ChatState()
        asyncio.create_task(monologue(context.application, chat_id))
    chat_states[chat_id].last_activity = datetime.now(UTC)
    await update.message.reply_text('Molly starts whispering...')


def main() -> None:
    load_dotenv()
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        raise RuntimeError(
            'TELEGRAM_TOKEN is not set. Provide your bot token via the '
            'TELEGRAM_TOKEN environment variable or a .env file.'
        )
    init_db()

    async def post_init(app: Application) -> None:
        asyncio.create_task(cleanup_chat_states())

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(close_db)
        .build()
    )
    app.add_handler(CommandHandler('start', start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    app.run_polling()


if __name__ == '__main__':
    main()
