"""Programmatic backtest runner.

Used two ways:
  * scripts/backtest.py wraps it for the CLI
  * core/learn/optimizer.py calls it to evaluate parameter candidates

Returns per-trade R-multiples with timestamps so the optimizer can split the
result into in-sample / out-of-sample windows by time.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.broker.paper_adapter import PaperBroker
from core.config import Config
from core.risk import RiskManager
from core.strategy import make_strategy

log = logging.getLogger("backtest")


@dataclass
class BTResult:
    signals: int = 0
    trades: int = 0
    wins: int = 0
    net: float = 0.0
    end_balance: float = 0.0
    start_balance: float = 0.0
    max_dd_pct: float = 0.0
    blocked: dict[str, int] = field(default_factory=dict)
    exits: dict[str, int] = field(default_factory=dict)
    # (close_time, r_multiple) per closed trade, in close order
    r_series: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def r_list(self) -> list[float]:
        return [r for _, r in self.r_series]

    @property
    def expectancy_r(self) -> float:
        rs = self.r_list
        return sum(rs) / len(rs) if rs else 0.0

    @property
    def win_rate(self) -> float:
        return 100.0 * self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float | None:
        gross_win = sum(r for r in self.r_list if r > 0)
        gross_loss = -sum(r for r in self.r_list if r < 0)
        return round(gross_win / gross_loss, 3) if gross_loss > 0 else None


def run_quiet(
    base_cfg: Config,
    bars: int,
    params: dict[str, Any] | None = None,
    allow_synthetic: bool = False,
) -> BTResult:
    """Replay the strategy over the paper feed with optional param overrides.

    Runs on a deep copy of the config — the caller's Config is never mutated.
    Refuses the synthetic feed unless explicitly allowed: parameter selection
    on a random walk is exactly the overfitting trap this system exists to avoid.
    """
    cfg = Config(copy.deepcopy(base_cfg.raw), root=base_cfg.root)
    cfg.set("telegram.enabled", False)
    cfg.set("ai.enabled", False)
    cfg.set("session.enabled", False)
    cfg.set("paper.speed", 1)
    if params:
        merged = dict(cfg.get("strategy.params", {}) or {})
        merged.update(params)
        cfg.set("strategy.params", merged)

    feed = str(cfg.get("paper.feed", "synthetic")).lower()
    if feed != "csv" and not allow_synthetic:
        raise RuntimeError(
            "backtest on the synthetic feed is disabled for optimization — "
            "provide CSV history (paper.feed: csv) or pass allow_synthetic=True "
            "for plumbing tests only"
        )

    timeframe = str(cfg.get("engine.timeframe", "M15")).upper()
    candle_count = int(cfg.get("engine.candles", 400))

    broker = PaperBroker(cfg)
    broker.connect()
    strategy = make_strategy(cfg)
    risk = RiskManager(cfg)

    symbols = [s["name"] for s in cfg.enabled_symbols()]
    result = BTResult(start_balance=broker.start_balance)

    last_bar: dict[str, object] = {}
    risk_by_ticket: dict[int, float] = {}
    equity_curve: list[float] = []
    settled = 0

    for _ in range(bars):
        broker.pump()
        if broker.data_exhausted():
            break

        account = broker.account()
        equity_curve.append(account.equity)
        risk.roll_day(account, now=broker.tick(symbols[0]).time)

        # Collect any fills settled by this pump before evaluating new entries.
        settled = _collect_closes(broker, risk_by_ticket, result, settled)

        if risk.daily_limit_breached(account):
            continue

        for symbol in symbols:
            candles = broker.candles(symbol, timeframe, candle_count)
            if len(candles) < 60 or last_bar.get(symbol) == candles[-2].time:
                continue
            last_bar[symbol] = candles[-2].time

            info = broker.symbol_info(symbol)
            sig = strategy.generate(symbol, timeframe, candles, info)
            if sig is None:
                continue
            result.signals += 1

            decision = risk.check(sig, account, info, broker.tick(symbol), broker.positions())
            if not decision.ok:
                key = decision.reason.split(":")[0].split("(")[0].strip()
                result.blocked[key] = result.blocked.get(key, 0) + 1
                continue

            order = broker.market_order(symbol, sig.side, decision.volume, sig.sl, sig.tp, "bt")
            if order.ok:
                risk_by_ticket[order.ticket] = decision.risk_amount
                risk.record_trade()

    broker.close_all()
    broker.pump()
    _collect_closes(broker, risk_by_ticket, result, settled)

    result.end_balance = round(broker.balance, 2)
    result.net = round(broker.balance - broker.start_balance, 2)
    result.trades = len(result.r_series)
    result.wins = sum(1 for _, r in result.r_series if r > 0)

    peak = equity_curve[0] if equity_curve else broker.start_balance
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            result.max_dd_pct = max(result.max_dd_pct, 100.0 * (peak - value) / peak)

    return result


def _collect_closes(
    broker: PaperBroker, risk_by_ticket: dict[int, float], result: BTResult, seen: int
) -> int:
    """Convert newly-settled trades into (time, R) entries."""
    history = broker.closed_trades
    for trade in history[seen:]:
        risk_amount = risk_by_ticket.get(trade["ticket"], 0.0)
        r = trade["pnl"] / risk_amount if risk_amount > 0 else 0.0
        close_time = _trade_close_time(broker, trade)
        result.r_series.append((close_time, r))
        result.exits[trade["reason"]] = result.exits.get(trade["reason"], 0) + 1
    return len(history)


def _trade_close_time(broker: PaperBroker, trade: dict) -> datetime:
    # The paper broker doesn't stamp close times on history rows; the symbol's
    # current bar time is the settlement time by construction.
    try:
        return broker.tick(trade["symbol"]).time
    except Exception:  # pragma: no cover
        return datetime.utcnow()
