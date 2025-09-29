#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Final, Sequence

SCRIPT_MAP: Final[dict[str, str]] = {
    "transfer": "inference_transfer_impl.py",
    "transfer_multiview": "inference_transfer_multiview_impl.py",
}


def parsing_argument(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(SCRIPT_MAP))
    ns, forwarded_args = parser.parse_known_args(argv)
    return ns.mode, forwarded_args


def import_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def script_path_for_mode(mode: str) -> Path:
    path = Path(__file__).resolve().parent / SCRIPT_MAP[mode]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return path


@contextmanager
def override_argv(new_argv: list[str]):
    original = list(sys.argv)
    try:
        sys.argv = new_argv
        yield
    finally:
        sys.argv = original


def invoke_child_main(script_path: Path, forwarded_args: list[str]) -> int:
    module = import_module_from_path(script_path)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise AttributeError(f"{script_path.name} has no callable main()")
    with override_argv([str(script_path), *forwarded_args]):
        try:
            rv = main_fn()
            return int(rv) if isinstance(rv, int) else 0
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    mode, forwarded_args = parsing_argument(tuple(argv or sys.argv[1:]))
    script_path = script_path_for_mode(mode)
    return invoke_child_main(script_path, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
