"""Simulated broker — lets the whole bot run on macOS/Linux with no MT5 install.

Two price sources:
  * feed: synthetic  -> seeded random walk (deterministic, always available)
  * feed: csv        -> replays data/<SYMBOL>_<TF>.csv (time,open,high,low,close,volume)

SL/TP are evaluated bar-by-bar. When a bar's range touches both, the stop is
assumed to hit first — the pessimistic assumption, so backtests don't flatter.
"""

from __future__ import annotations

import csv
import logging
import random
from datetime import datetime, timedelta, timezone
from itertools import count as _counter
from pathlib import Path

from core.broker.base import TIMEFRAME_MINUTES, Broker
from core.models import (
    AccountInfo,
    Candle,
    OrderResult,
    Position,
    Side,
    SymbolInfo,
    Tick,
)

log = logging.getLogger("broker.paper")

# Reasonable stand-ins for real contract specs (USD account).
_SPECS: dict[str, dict] = {
    "XAUUSD": dict(digits=2, point=0.01, tick_value=1.0, contract_size=100, price=2350.0, vol=0.0016),
    "EURUSD": dict(digits=5, point=0.00001, tick_value=1.0, contract_size=100_000, price=1.0850, vol=0.0006),
    "GBPUSD": dict(digits=5, point=0.00001, tick_value=1.0, contract_size=100_000, price=1.2700, vol=0.0007),
    "USDJPY": dict(digits=3, point=0.001, tick_value=0.9, contract_size=100_000, price=155.00, vol=0.0006),
    "BTCUSD": dict(digits=2, point=0.01, tick_value=1.0, contract_size=1, price=64000.0, vol=0.004),
}
_DEFAULT_SPEC = dict(digits=5, point=0.00001, tick_value=1.0, contract_size=100_000, price=1.0000, vol=0.0006)


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.magic = int(cfg.get("engine.magic", 570123))
        self.timeframe = str(cfg.get("engine.timeframe", "M15")).upper()
        self.tf_minutes = TIMEFRAME_MINUTES.get(self.timeframe, 15)

        self.balance = float(cfg.get("paper.balance", 10_000))
        self.start_balance = self.balance
        self.leverage = int(cfg.get("paper.leverage", 500))
        self.spread_points = float(cfg.get("paper.spread_points", 15))
        self.commission_per_lot = float(cfg.get("paper.commission_per_lot", 7.0))
        self.slippage_points = float(cfg.get("paper.slippage_points", 3))
        self.feed = str(cfg.get("paper.feed", "synthetic")).lower()
        self.speed = max(1, int(cfg.get("paper.speed", 1)))

        self._rng = random.Random(int(cfg.get("paper.synthetic_seed", 42)))
        self._tickets = _counter(1000001)
        self._positions: dict[int, Position] = {}
        self._history: list[dict] = []           # closed trades, for the summary
        self._series: dict[str, list[Candle]] = {}
        self._csv_cursor: dict[str, int] = {}
        self._csv_all: dict[str, list[Candle]] = {}

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        for sym in self.cfg.enabled_symbols():
            name = sym["name"]
            if self.feed == "csv":
                self._load_csv(name)
            else:
                self._seed_synthetic(name)
        log.info(
            "Paper broker ready | feed=%s | balance=%.2f | symbols=%s",
            self.feed, self.balance, ", ".join(self._series),
        )

    def shutdown(self) -> None:
        wins = [t for t in self._history if t["pnl"] > 0]
        log.info(
            "Paper session: %d trades | %d wins (%.0f%%) | net %.2f | balance %.2f",
            len(self._history),
            len(wins),
            100.0 * len(wins) / len(self._history) if self._history else 0.0,
            self.balance - self.start_balance,
            self.balance,
        )

    # ---------------------------------------------------------------- price feed
    def _spec(self, symbol: str) -> dict:
        return _SPECS.get(symbol.upper().split(".")[0], _DEFAULT_SPEC)

    def _seed_synthetic(self, symbol: str, bars: int = 500) -> None:
        spec = self._spec(symbol)
        price = spec["price"]
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = now - timedelta(minutes=self.tf_minutes * bars)

        series: list[Candle] = []
        for i in range(bars):
            series.append(self._next_synthetic_bar(symbol, price, start + timedelta(minutes=self.tf_minutes * i)))
            price = series[-1].close
        self._series[symbol] = series

    def _next_synthetic_bar(self, symbol: str, prev_close: float, when: datetime) -> Candle:
        spec = self._spec(symbol)
        vol = spec["vol"]
        # Mild trend persistence so the trend strategy has something to find.
        drift = self._rng.gauss(0, vol * 0.35)
        ret = self._rng.gauss(drift, vol)
        close = max(prev_close * (1 + ret), spec["point"] * 10)

        wick = abs(self._rng.gauss(0, vol * 0.6)) * prev_close
        o = prev_close
        h = max(o, close) + wick
        low = min(o, close) - wick
        d = spec["digits"]
        return Candle(
            time=when,
            open=round(o, d),
            high=round(h, d),
            low=round(low, d),
            close=round(close, d),
            volume=float(self._rng.randint(200, 2000)),
        )

    def _load_csv(self, symbol: str) -> None:
        path = Path(self.cfg.root) / str(self.cfg.get("paper.csv_dir", "data")) / f"{symbol}_{self.timeframe}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"CSV feed enabled but {path} is missing. "
                "Export it from MT5 (Tools > History Center) with columns "
                "time,open,high,low,close,volume — or set paper.feed: synthetic."
            )
        rows: list[Candle] = []
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                raw_time = row.get("time") or row.get("date") or row.get("datetime")
                rows.append(
                    Candle(
                        time=_parse_time(raw_time),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or row.get("tick_volume") or 0),
                    )
                )
        rows.sort(key=lambda c: c.time)
        if len(rows) < 250:
            raise ValueError(f"{path} has only {len(rows)} bars — need at least 250 for warmup.")

        self._csv_all[symbol] = rows
        warmup = 250
        self._series[symbol] = rows[:warmup]
        self._csv_cursor[symbol] = warmup
        log.info("Loaded %d bars for %s from %s", len(rows), symbol, path.name)

    def pump(self) -> None:
        """Advance simulated time by `speed` bars and settle any SL/TP hits."""
        for _ in range(self.speed):
            for symbol, series in self._series.items():
                bar = self._advance_one(symbol, series)
                if bar is None:
                    continue
                self._settle_bar(symbol, bar)

    def _advance_one(self, symbol: str, series: list[Candle]) -> Candle | None:
        if self.feed == "csv":
            cursor = self._csv_cursor.get(symbol, 0)
            all_bars = self._csv_all.get(symbol, [])
            if cursor >= len(all_bars):
                return None
            bar = all_bars[cursor]
            self._csv_cursor[symbol] = cursor + 1
        else:
            last = series[-1]
            bar = self._next_synthetic_bar(
                symbol, last.close, last.time + timedelta(minutes=self.tf_minutes)
            )
        series.append(bar)
        if len(series) > 2000:
            del series[:-1500]
        return bar

    def data_exhausted(self) -> bool:
        """True once a CSV replay has run off the end of every file."""
        if self.feed != "csv":
            return False
        return all(
            self._csv_cursor.get(sym, 0) >= len(bars) for sym, bars in self._csv_all.items()
        )

    # ---------------------------------------------------------------- interface
    def symbol_info(self, symbol: str) -> SymbolInfo:
        spec = self._spec(symbol)
        return SymbolInfo(
            name=symbol,
            digits=spec["digits"],
            point=spec["point"],
            tick_size=spec["point"],
            tick_value=spec["tick_value"],
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            contract_size=spec["contract_size"],
            stops_level_points=0,
        )

    def tick(self, symbol: str) -> Tick:
        series = self._series.get(symbol)
        if not series:
            raise RuntimeError(f"no simulated series for {symbol}")
        info = self.symbol_info(symbol)
        mid = series[-1].close
        half = self.spread_points * info.point / 2.0
        return Tick(
            symbol=symbol,
            bid=info.normalize_price(mid - half),
            ask=info.normalize_price(mid + half),
            time=series[-1].time,
        )

    def candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        series = self._series.get(symbol, [])
        return list(series[-count:])

    def account(self) -> AccountInfo:
        floating = sum(p.profit for p in self._refresh_positions())
        equity = self.balance + floating
        used = sum(
            p.volume * self.symbol_info(p.symbol).contract_size * p.price_open / self.leverage
            for p in self._positions.values()
        )
        return AccountInfo(
            login=0,
            balance=round(self.balance, 2),
            equity=round(equity, 2),
            margin=round(used, 2),
            free_margin=round(equity - used, 2),
            currency="USD",
            leverage=self.leverage,
        )

    def positions(self, symbol: str | None = None) -> list[Position]:
        positions = self._refresh_positions()
        if symbol:
            return [p for p in positions if p.symbol == symbol]
        return positions

    def market_order(
        self,
        symbol: str,
        side: Side,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult:
        info = self.symbol_info(symbol)
        volume = info.normalize_volume(volume)
        if volume <= 0:
            return OrderResult(ok=False, error="volume rounds to zero")

        tick = self.tick(symbol)
        slip = self.slippage_points * info.point * side.sign
        price = info.normalize_price(tick.price_for(side) + slip)

        ticket = next(self._tickets)
        self._positions[ticket] = Position(
            ticket=ticket,
            symbol=symbol,
            side=side,
            volume=volume,
            price_open=price,
            sl=info.normalize_price(sl),
            tp=info.normalize_price(tp),
            profit=0.0,
            price_current=price,
            time=tick.time,
            magic=self.magic,
            comment=comment,
        )
        log.info("[paper] OPEN #%d %s %s %.2f @ %s", ticket, side.value, symbol, volume, price)
        return OrderResult(ok=True, ticket=ticket, price=price, volume=volume)

    def modify_position(self, ticket: int, sl: float, tp: float) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(ok=False, error=f"position {ticket} not found")
        info = self.symbol_info(pos.symbol)
        pos.sl = info.normalize_price(sl)
        pos.tp = info.normalize_price(tp)
        return OrderResult(ok=True, ticket=ticket)

    def close_position(self, ticket: int) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult(ok=False, error=f"position {ticket} not found")
        tick = self.tick(pos.symbol)
        price = tick.price_for(pos.side.opposite)
        return self._settle(ticket, price, "manual")

    # ---------------------------------------------------------------- internals
    def _pnl(self, pos: Position, price: float) -> float:
        info = self.symbol_info(pos.symbol)
        ticks = (price - pos.price_open) * pos.side.sign / info.tick_size
        return ticks * info.tick_value * pos.volume

    def _refresh_positions(self) -> list[Position]:
        out = []
        for pos in self._positions.values():
            tick = self.tick(pos.symbol)
            pos.price_current = tick.price_for(pos.side.opposite)
            pos.profit = round(self._pnl(pos, pos.price_current), 2)
            out.append(pos)
        return out

    def _settle_bar(self, symbol: str, bar: Candle) -> None:
        """Check every open position on this symbol against the bar's range."""
        for ticket, pos in list(self._positions.items()):
            if pos.symbol != symbol:
                continue
            hit_sl = (pos.side is Side.BUY and bar.low <= pos.sl) or (
                pos.side is Side.SELL and bar.high >= pos.sl
            )
            hit_tp = (pos.side is Side.BUY and bar.high >= pos.tp) or (
                pos.side is Side.SELL and bar.low <= pos.tp
            )
            # Pessimistic: if both are inside one bar, assume the stop went first.
            if hit_sl:
                self._settle(ticket, pos.sl, "sl")
            elif hit_tp:
                self._settle(ticket, pos.tp, "tp")

    def _settle(self, ticket: int, price: float, why: str) -> OrderResult:
        pos = self._positions.pop(ticket)
        # Round-turn commission is charged here, inside the P&L, so the journal
        # and the balance agree on what the trade actually made.
        pnl = round(self._pnl(pos, price) - self.commission_per_lot * pos.volume, 2)
        self.balance += pnl
        self._history.append(
            dict(ticket=ticket, symbol=pos.symbol, side=pos.side.value, volume=pos.volume,
                 open=pos.price_open, close=price, pnl=pnl, reason=why)
        )
        log.info("[paper] CLOSE #%d %s @ %s (%s) pnl=%.2f", ticket, pos.symbol, price, why, pnl)
        return OrderResult(ok=True, ticket=ticket, price=price, volume=pos.volume)

    def closed_trade_info(self, ticket: int) -> tuple[float, float, str] | None:
        for trade in reversed(self._history):
            if trade["ticket"] == ticket:
                reason = {"sl": "stop loss", "tp": "take profit"}.get(
                    trade["reason"], trade["reason"]
                )
                return trade["close"], trade["pnl"], reason
        return None

    @property
    def closed_trades(self) -> list[dict]:
        return list(self._history)


def _parse_time(value: str | None) -> datetime:
    if not value:
        raise ValueError("CSV row has no time column")
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Unix epoch fallback
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError):
        raise ValueError(f"unrecognised time format: {value!r}") from None
