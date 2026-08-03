"""Strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import Candle, Signal, SymbolInfo


class Strategy(ABC):
    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @abstractmethod
    def generate(
        self, symbol: str, timeframe: str, candles: list[Candle], info: SymbolInfo
    ) -> Signal | None:
        """Return a Signal for the just-closed bar, or None.

        The engine only calls this once per closed bar, so implementations
        should read from `candles[-2]` (last closed) rather than `candles[-1]`
        (still forming).
        """
