"""Position sizing and the hard limits that sit in front of every order.

Nothing here is advisory — if `check()` returns a rejection, the engine does
not trade, regardless of what the strategy or the AI said.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

from core.models import AccountInfo, Position, Signal, SymbolInfo, Tick

log = logging.getLogger("risk")


@dataclass
class RiskDecision:
    ok: bool
    volume: float = 0.0
    risk_amount: float = 0.0
    reason: str = ""


class RiskManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.default_risk_pct = float(cfg.get("risk.default_risk_pct", 0.5))
        self.max_open = int(cfg.get("risk.max_open_positions", 3))
        self.max_per_symbol = int(cfg.get("risk.max_positions_per_symbol", 1))
        self.daily_max_loss_pct = float(cfg.get("risk.daily_max_loss_pct", 3.0))
        self.daily_max_trades = int(cfg.get("risk.daily_max_trades", 8))
        self.max_lot = float(cfg.get("risk.max_lot", 1.0))
        self.min_free_margin_pct = float(cfg.get("risk.min_free_margin_pct", 30.0))

        self._day: date | None = None
        self._day_start_equity: float = 0.0
        self._trades_today: int = 0
        self.halted_reason: str = ""

    # ---------------------------------------------------------------- day roll
    def roll_day(self, account: AccountInfo, now: datetime | None = None) -> bool:
        """Reset daily counters when the UTC date changes. Returns True on roll."""
        today = (now or datetime.now(timezone.utc)).date()
        if self._day == today:
            return False
        self._day = today
        self._day_start_equity = account.equity
        self._trades_today = 0
        self.halted_reason = ""
        log.info("New trading day %s — start equity %.2f", today, account.equity)
        return True

    def record_trade(self) -> None:
        self._trades_today += 1

    @property
    def trades_today(self) -> int:
        return self._trades_today

    @property
    def day_start_equity(self) -> float:
        return self._day_start_equity

    def daily_pnl(self, account: AccountInfo) -> float:
        if not self._day_start_equity:
            return 0.0
        return account.equity - self._day_start_equity

    def daily_pnl_pct(self, account: AccountInfo) -> float:
        if not self._day_start_equity:
            return 0.0
        return 100.0 * self.daily_pnl(account) / self._day_start_equity

    # ---------------------------------------------------------------- gates
    def daily_limit_breached(self, account: AccountInfo) -> str:
        """Non-empty string = trading should stop for the rest of the day."""
        if self.daily_max_loss_pct > 0:
            pnl_pct = self.daily_pnl_pct(account)
            if pnl_pct <= -abs(self.daily_max_loss_pct):
                return (
                    f"daily loss limit hit: {pnl_pct:.2f}% "
                    f"(limit -{self.daily_max_loss_pct:.2f}%)"
                )
        if self.daily_max_trades > 0 and self._trades_today >= self.daily_max_trades:
            return f"daily trade cap reached ({self._trades_today}/{self.daily_max_trades})"
        return ""

    def check(
        self,
        signal: Signal,
        account: AccountInfo,
        info: SymbolInfo,
        tick: Tick,
        open_positions: list[Position],
    ) -> RiskDecision:
        """Full pre-trade gate. Returns sized volume or the reason we said no."""
        breach = self.daily_limit_breached(account)
        if breach:
            return RiskDecision(False, reason=breach)

        if len(open_positions) >= self.max_open:
            return RiskDecision(
                False, reason=f"max open positions ({len(open_positions)}/{self.max_open})"
            )

        same_symbol = [p for p in open_positions if p.symbol == signal.symbol]
        if len(same_symbol) >= self.max_per_symbol:
            return RiskDecision(
                False,
                reason=f"already {len(same_symbol)} position(s) on {signal.symbol}",
            )

        if account.free_margin_pct < self.min_free_margin_pct:
            return RiskDecision(
                False,
                reason=f"free margin {account.free_margin_pct:.1f}% below "
                f"{self.min_free_margin_pct:.1f}% floor",
            )

        sym_conf = self.cfg.symbol_conf(signal.symbol)
        max_spread = float(sym_conf.get("max_spread_points", 0) or 0)
        spread = tick.spread_points(info)
        if max_spread > 0 and spread > max_spread:
            return RiskDecision(
                False, reason=f"spread {spread:.0f}pt > max {max_spread:.0f}pt"
            )

        risk_pct = float(sym_conf.get("risk_pct", self.default_risk_pct))
        risk_amount = account.equity * risk_pct / 100.0
        if risk_amount <= 0:
            return RiskDecision(False, reason="computed risk amount is zero")

        volume = self.size_position(signal, info, risk_amount)
        if volume <= 0:
            return RiskDecision(
                False,
                reason=f"stop distance too wide to size at {risk_pct}% "
                f"(min lot {info.volume_min} would risk more than "
                f"{risk_amount:.2f} {account.currency})",
            )

        actual_risk = self.risk_for_volume(signal, info, volume)
        # Rounding up to volume_min can overshoot the budget — 25% tolerance.
        if actual_risk > risk_amount * 1.25:
            return RiskDecision(
                False,
                reason=f"min lot risks {actual_risk:.2f} vs budget {risk_amount:.2f}",
            )

        return RiskDecision(True, volume=volume, risk_amount=actual_risk)

    # ---------------------------------------------------------------- sizing
    def size_position(self, signal: Signal, info: SymbolInfo, risk_amount: float) -> float:
        """Lots such that a stop-out costs approximately `risk_amount`."""
        distance = signal.risk_distance
        if distance <= 0 or info.tick_size <= 0 or info.tick_value <= 0:
            return 0.0

        ticks = distance / info.tick_size
        loss_per_lot = ticks * info.tick_value
        if loss_per_lot <= 0:
            return 0.0

        raw = risk_amount / loss_per_lot
        volume = info.normalize_volume(min(raw, self.max_lot))

        # normalize_volume clamps up to volume_min; make sure that didn't
        # silently create a position we never intended to allow.
        if raw < info.volume_min * 0.5:
            return 0.0
        return volume

    def risk_for_volume(self, signal: Signal, info: SymbolInfo, volume: float) -> float:
        ticks = signal.risk_distance / info.tick_size if info.tick_size else 0
        return ticks * info.tick_value * volume
