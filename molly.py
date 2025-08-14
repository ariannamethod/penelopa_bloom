import asyncio
import os
import random
import re
import sqlite3
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
MUTED_LOG = Path('origin/logs/muted.txt')


def load_user_lines() -> list[str]:
    """Return previously stored user lines."""
    if not LINES_FILE.exists():
        return []
    with LINES_FILE.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def init_db() -> None:
    """Ensure the SQLite database exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line TEXT,
            created_at TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


def store_line(line: str) -> None:
    """Persist a line to the database and the log file."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO lines (line, created_at) VALUES (?, ?)',
        (line, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    conn.close()
    with LINES_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def text_chunks() -> Iterator[str]:
    """Yield chunks from Molly's monologue without cutting words."""
    text = ORIGIN_TEXT.read_text(encoding='utf-8')
    pos = 0
    n = len(text)
    while pos < n:
        length = random.randint(200, 800)
        end = min(pos + length, n)
        while end < n and text[end] not in {' ', '\n'}:
            end += 1
        chunk = text[pos:end].strip()
        pos = end + 1
        if chunk:
            yield chunk


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
    voice_enabled: bool = True


chat_states: dict[int, ChatState] = {}
user_lines: list[str] = load_user_lines()


async def monologue(app: Application, chat_id: int) -> None:
    state = chat_states.setdefault(chat_id, ChatState())
    async for chunk in _chunk_stream(state):
        if state.voice_enabled:
            delay = random.randint(5, 50)
            if random.random() < 0.1:
                delay = random.randint(120, 180)
            await simulate_typing(app.bot, chat_id, delay)
            await app.bot.send_message(chat_id=chat_id, text=chunk)
        else:
            with MUTED_LOG.open('a', encoding='utf-8') as f:
                f.write(chunk + '\n')
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
    chat_id = update.effective_chat.id
    state = chat_states.setdefault(chat_id, ChatState())
    state.next_prefix = random.choice(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id not in chat_states:
        chat_states[chat_id] = ChatState()
        asyncio.create_task(monologue(context.application, chat_id))
    await update.message.reply_text('Molly starts whispering...')


async def toggle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = chat_states.setdefault(chat_id, ChatState())
    state.voice_enabled = not state.voice_enabled
    status = 'enabled' if state.voice_enabled else 'disabled'
    await update.message.reply_text(f"Voice {status}")


def main() -> None:
    token = os.environ['TELEGRAM_TOKEN']
    init_db()
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('voice', toggle_voice))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    app.run_polling()


if __name__ == '__main__':
    main()
