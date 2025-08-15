import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from frm import read_frm, write_frm  # noqa: E402


def test_round_trip(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    file_path = tmp_path / 'sample.frm'
    write_frm(df, file_path)
    loaded = read_frm(file_path)
    assert loaded.equals(df)
