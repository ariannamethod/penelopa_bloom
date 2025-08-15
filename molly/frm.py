"""Utilities for reading and writing FRM files using pandas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_frm(path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a FRM file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the FRM file.
    encoding:
        Text encoding to use when reading the file. Defaults to ``"utf-8"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame constructed from the FRM contents.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    return pd.read_csv(file_path, encoding=encoding)


def write_frm(
    df: pd.DataFrame, path: str | Path, *, encoding: str = "utf-8"
) -> None:
    """Write a :class:`pandas.DataFrame` to a FRM file.

    Parameters
    ----------
    df:
        DataFrame to serialize.
    path:
        Destination file path.
    encoding:
        Text encoding to use when writing the file. Defaults to ``"utf-8"``.
    """
    file_path = Path(path)
    df.to_csv(file_path, index=False, encoding=encoding)


__all__ = ["read_frm", "write_frm"]
