import pandas as pd
from molly.frm import read_frm, write_frm


def test_round_trip(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    file_path = tmp_path / 'sample.frm'
    write_frm(df, file_path)
    loaded = read_frm(file_path)
    assert loaded.equals(df)
