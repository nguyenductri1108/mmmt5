"""Config loading: config.yaml for behaviour, .env for secrets."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Thin wrapper over the YAML tree with dotted-path lookups."""

    def __init__(self, data: dict[str, Any], root: Path = ROOT) -> None:
        self._data = data
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, path: str) -> Any:
        value = self.get(path, _MISSING)
        if value is _MISSING:
            raise KeyError(f"missing config key: {path}")
        return value

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def enabled_symbols(self) -> list[dict[str, Any]]:
        return [s for s in self.get("symbols", []) if s.get("enabled", True)]

    def symbol_conf(self, name: str) -> dict[str, Any]:
        for s in self.get("symbols", []):
            if s.get("name") == name:
                return s
        return {"name": name}


_MISSING = object()


def load_config(path: str | Path | None = None) -> Config:
    """Read .env then config.yaml. Returns a Config; env vars stay in os.environ."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default) or default


def env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except ValueError:
        return default


def setup_logging(cfg: Config) -> logging.Logger:
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    log_file = cfg.get("logging.file", "logs/bot.log")
    path = ROOT / log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Third-party libs are chatty at DEBUG/INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("bot")
