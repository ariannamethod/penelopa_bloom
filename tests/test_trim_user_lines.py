import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import molly  # noqa: E402


def test_trim_user_lines(tmp_path):
    molly.LINES_FILE = tmp_path / "lines.txt"
    molly.user_lines = [f"line {i}" for i in range(10)]
    molly.user_weights = [float(i) for i in range(10)]
    molly.LINES_FILE.write_text("\n".join(molly.user_lines) + "\n", encoding="utf-8")
    molly.trim_user_lines(max_lines=5)
    assert molly.user_lines == [f"line {i}" for i in range(5, 10)]
    assert molly.user_weights == [float(i) for i in range(5, 10)]
    content = molly.LINES_FILE.read_text(encoding="utf-8").splitlines()
    assert content == [f"line {i}" for i in range(5, 10)]
