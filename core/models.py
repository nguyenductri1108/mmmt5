"""Broker-agnostic data types.

Every adapter (MT5, paper) speaks in these, so the strategy, risk and AI layers
never touch a vendor-specific structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short — handy in P&L / distance math."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    digits: int
    point: float           # smallest price increment (e.g. 0.01 for XAUUSD)
    tick_size: float
    tick_value: float      # account-currency P&L for one tick on one lot
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    stops_level_points: int = 0   # broker's minimum SL/TP distance, in points

    def normalize_volume(self, volume: float) -> float:
        """Round a lot size down onto the broker's volume grid."""
        if self.volume_step <= 0:
            return round(volume, 2)
        steps = int(volume / self.volume_step + 1e-9)
        vol = steps * self.volume_step
        vol = max(self.volume_min, min(self.volume_max, vol))
        # Volume steps are decimal (0.01, 0.1, 1.0) — kill float noise.
        decimals = max(0, len(f"{self.volume_step:.8f}".rstrip("0").split(".")[-1]))
        return round(vol, decimals)

    def normalize_price(self, price: float) -> float:
        return round(price, self.digits)


@dataclass(frozen=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    def spread_points(self, info: SymbolInfo) -> float:
        if info.point <= 0:
            return 0.0
        return (self.ask - self.bid) / info.point

    def price_for(self, side: Side) -> float:
        """Price you actually get filled at when opening `side`."""
        return self.ask if side is Side.BUY else self.bid


@dataclass(frozen=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    leverage: int

    @property
    def free_margin_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return 100.0 * self.free_margin / self.equity


@dataclass
class Position:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float           # floating P&L in account currency
    price_current: float
    time: datetime
    magic: int = 0
    comment: str = ""

    def r_multiple(self) -> float:
        """How far price has travelled, expressed in initial-risk units."""
        risk = abs(self.price_open - self.sl)
        if risk <= 0:
            return 0.0
        moved = (self.price_current - self.price_open) * self.side.sign
        return moved / risk


@dataclass
class OrderResult:
    ok: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    error: str = ""
    retcode: int = 0


@dataclass
class Signal:
    """A trade candidate produced by a strategy, before risk/AI processing."""

    symbol: str
    side: Side
    entry: float
    sl: float
    tp: float
    reason: str                       # short human-readable rationale
    timeframe: str = ""
    atr: float = 0.0
    bar_time: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)  # indicator snapshot for the AI

    # Filled in later by the risk / AI stages
    volume: float = 0.0
    risk_amount: float = 0.0

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def reward_distance(self) -> float:
        return abs(self.tp - self.entry)

    @property
    def rr(self) -> float:
        return self.reward_distance / self.risk_distance if self.risk_distance else 0.0


@dataclass
class AIVerdict:
    decision: str                     # approve | reject
    confidence: float                 # 0.0 - 1.0
    size_multiplier: float = 1.0      # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    error: str = ""

    @property
    def approved(self) -> bool:
        return self.decision == "approve"
