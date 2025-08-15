import ast
import random
import re
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "molly.py").read_text(encoding="utf-8")

# Extract constants
ORIGIN_TEXT = ROOT / re.search(r"^ORIGIN_TEXT = Path\('([^']+)'\)", SOURCE, re.MULTILINE).group(1)
MAX_MESSAGE_LENGTH = int(re.search(r"^MAX_MESSAGE_LENGTH = (\d+)", SOURCE, re.MULTILINE).group(1))

# Extract random_chunks function without importing heavy dependencies
module = {}
for node in ast.parse(SOURCE).body:
    if isinstance(node, ast.FunctionDef) and node.name == "random_chunks":
        func_code = ast.Module(body=[node], type_ignores=[])
        exec(
            compile(func_code, filename="molly.py", mode="exec"),
            {
                "Path": Path,
                "random": random,
                "Iterator": Iterator,
                "ORIGIN_TEXT": ORIGIN_TEXT,
                "MAX_MESSAGE_LENGTH": MAX_MESSAGE_LENGTH,
            },
            module,
        )
        break
random_chunks = module["random_chunks"]


def chunk_string(text: str) -> list[str]:
    buffer = text
    chunks = []
    while len(buffer) >= MAX_MESSAGE_LENGTH:
        split_pos = buffer.rfind(" ", 0, MAX_MESSAGE_LENGTH)
        if split_pos == -1:
            split_pos = buffer.find(" ", MAX_MESSAGE_LENGTH)
            if split_pos == -1:
                break
        chunk, buffer = buffer[:split_pos], buffer[split_pos + 1 :]
        if chunk:
            chunks.append(chunk)
    remainder = buffer.strip()
    if remainder:
        chunks.append(remainder)
    return chunks


def rotated_content(offset: int) -> str:
    with ORIGIN_TEXT.open("rb") as f:
        f.seek(offset)
        part1 = f.read()
        f.seek(0)
        part2 = f.read(offset)
    return (part1 + part2).decode("utf-8", errors="ignore")


def test_random_chunks_rotation(monkeypatch):
    offset = 123
    monkeypatch.setattr(random, "randrange", lambda _: offset)
    expected_text = rotated_content(offset)
    expected_chunks = chunk_string(expected_text)
    assert list(random_chunks()) == expected_chunks
