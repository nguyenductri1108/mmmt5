"""The interface every broker adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import (
    AccountInfo,
    Candle,
    OrderResult,
    Position,
    Side,
    SymbolInfo,
    Tick,
)

# Minutes per timeframe — used by the paper broker and for bar-close maths.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


class Broker(ABC):
    """Minimal surface the engine needs. Keep it small — it has two implementations."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection. Raise on failure — the engine will not start."""

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    def candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        """Most recent `count` bars, oldest first. The last bar may be unfinished."""

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    def market_order(
        self,
        symbol: str,
        side: Side,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult: ...

    @abstractmethod
    def modify_position(self, ticket: int, sl: float, tp: float) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: int) -> OrderResult: ...

    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        """Flatten everything (optionally one symbol). Default impl loops close_position."""
        results = []
        for pos in self.positions(symbol):
            results.append(self.close_position(pos.ticket))
        return results

    # Hook the paper broker uses to advance simulated time; live is a no-op.
    def pump(self) -> None:  # noqa: B027 - intentional no-op default
        pass

    def closed_trade_info(self, ticket: int) -> tuple[float, float, str] | None:
        """(close_price, realised_pnl, reason) for a position that just vanished.

        Optional: adapters that can query trade history override this so the
        engine reports exact fills. Returning None makes the engine fall back to
        the last state it observed before the position disappeared.
        """
        return None
