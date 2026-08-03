"""Strategies. Add your own by subclassing Strategy and registering it here."""

from __future__ import annotations

from core.strategy.base import Strategy
from core.strategy.ema_atr import EmaAtrStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "ema_atr": EmaAtrStrategy,
}

__all__ = ["Strategy", "EmaAtrStrategy", "make_strategy"]


def make_strategy(cfg) -> Strategy:
    name = str(cfg.get("strategy.name", "ema_atr"))
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown strategy {name!r}; available: {', '.join(_REGISTRY)}")
    return cls(cfg.get("strategy.params", {}) or {})
