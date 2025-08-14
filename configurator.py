"""Configuration utilities using argparse.

This module provides a function :func:`update_config` that updates a dictionary
of configuration values based on command line arguments and an optional config
file. The config file can be in `.py` or `.yaml` format and is parsed without
executing arbitrary code.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict

import yaml


def _str2bool(v: str) -> bool:
    if v.lower() in {"1", "true", "yes", "y"}:
        return True
    if v.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_node(n) for n in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(n) for n in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_node(k): _eval_node(v) for k, v in zip(node.keys, node.values)}
    raise ValueError("Unsupported expression in config file")


def _read_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if path.suffix == ".py":
        with open(path, "r", encoding="utf-8") as f:
            node = ast.parse(f.read(), filename=str(path))
        cfg: Dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
                stmt.targets[0], ast.Name
            ):
                cfg[stmt.targets[0].id] = _eval_node(stmt.value)
        return cfg
    raise ValueError(f"Unsupported config file: {path}")


def update_config(cfg: Dict[str, Any]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to .py or .yaml config")
    for key, value in list(cfg.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=_str2bool)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int)
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float)
        elif isinstance(value, str):
            parser.add_argument(f"--{key}", type=str)
        elif isinstance(value, tuple) and all(isinstance(x, float) for x in value):
            parser.add_argument(f"--{key}", type=lambda s: tuple(float(x) for x in s.split(',')))
    args = parser.parse_args()

    file_cfg: Dict[str, Any] = {}
    if args.config:
        file_cfg = _read_config_file(Path(args.config))

    for k, v in file_cfg.items():
        if k not in cfg:
            raise ValueError(f"Unknown config key: {k}")
        cfg[k] = v

    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        cfg[k] = v
