import molly


def test_text_chunks_no_word_cut(tmp_path, monkeypatch):
    text = "one two three four five six seven eight nine ten"
    file_path = tmp_path / "molly.md"
    file_path.write_text(text)
    monkeypatch.setattr(molly, "ORIGIN_TEXT", file_path)
    monkeypatch.setattr(molly.random, "randint", lambda a, b: 10)
    chunks = list(molly.text_chunks())
    assert " ".join(chunks) == text
    max_word = max(len(w) for w in text.split())
    for chunk in chunks:
        assert len(chunk) <= 10 + max_word


def test_prepare_lines(monkeypatch):
    monkeypatch.setattr(molly.random, "randint", lambda a, b: 2)
    text = "Hello world! How are you? I'm fine."
    lines = molly.prepare_lines(text)
    assert len(lines) == 2
    assert all(lines)


def test_prepare_lines_empty():
    assert molly.prepare_lines("!!!") == []
