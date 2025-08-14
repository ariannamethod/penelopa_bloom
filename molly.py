import asyncio
import os
import random
import re
import sqlite3
import logging
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

ORIGIN_TEXT = Path('origin/molly.md')
LINES_FILE = Path('origin/logs/lines.txt')
DB_PATH = Path('origin/logs/lines.db')
MAX_USER_LINES = 1000

# Global connection to be shared across threads
db_conn: sqlite3.Connection | None = None


def load_user_lines() -> list[str]:
    """Return previously stored user lines, trimmed to the last MAX_USER_LINES."""
    if not LINES_FILE.exists():
        return []
    with LINES_FILE.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) > MAX_USER_LINES:
        lines = lines[-MAX_USER_LINES:]
        with LINES_FILE.open('w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
    return lines


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
                created_at TEXT
            )
            '''
        )
        db_conn.commit()
    except Exception:
        logging.exception("Failed to initialize lines database")


def store_line(line: str) -> None:
    """Persist a line to the database and the log file."""
    if db_conn is None:
        logging.error("Database not initialized")
        return
    try:
        db_conn.execute(
            'INSERT INTO lines (line, created_at) VALUES (?, ?)',
            (line, datetime.now(UTC).isoformat()),
        )
        db_conn.commit()
    except Exception:
        logging.exception("Failed to store line")
        return
    with LINES_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def trim_user_lines(max_lines: int = MAX_USER_LINES) -> None:
    """Trim user_lines and the log file to the last max_lines entries."""
    if len(user_lines) <= max_lines:
        return
    del user_lines[:-max_lines]
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
                split_pos = max(buffer.rfind(" ", 0, 1024), buffer.rfind("\n", 0, 1024))
                if split_pos == -1:
                    # No whitespace in the first chunk, search entire buffer
                    split_pos = max(buffer.rfind(" "), buffer.rfind("\n"))
                    if split_pos == -1:
                        break
                chunk, buffer = buffer[:split_pos].strip(), buffer[split_pos + 1 :]
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
user_lines: list[str] = load_user_lines()

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
            prefix = random.choice(user_lines)
        yield f"{prefix} {chunk}" if prefix else chunk
        await asyncio.sleep(0)


def prepare_lines(text: str) -> list[str]:
    sentences = re.split(r'[.!?]+', text)
    cleaned = [re.sub(r'[^\w\s]', '', s).strip() for s in sentences]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return []
    lines_count = 2 if len(cleaned) <= 2 else random.randint(2, 3)
    group_size = max(1, len(cleaned) // lines_count)
    lines = []
    idx = 0
    for _ in range(lines_count):
        lines.append(' '.join(cleaned[idx:idx + group_size]))
        idx += group_size
    return lines


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    text = update.message.text or ''
    lines = prepare_lines(text)
    if not lines:
        return
    for line in lines:
        store_line(line)
        user_lines.append(line)
    trim_user_lines()
    chat_id = update.effective_chat.id
    state = chat_states.setdefault(chat_id, ChatState())
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
    token = os.environ['TELEGRAM_TOKEN']
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
