"""MetaTrader 5 adapter — the live broker.

The `MetaTrader5` package is Windows-only, so this module is imported lazily
(see core/broker/__init__.py). On macOS/Linux, use mode=paper instead.

Requirements on the Windows box:
  * MT5 terminal installed and running, logged into the account
  * "Algo Trading" enabled (toolbar button) and the symbol visible in Market Watch
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.broker.base import Broker
from core.config import env, env_int
from core.models import (
    AccountInfo,
    Candle,
    OrderResult,
    Position,
    Side,
    SymbolInfo,
    Tick,
)

log = logging.getLogger("broker.mt5")

try:  # pragma: no cover - import guarded so the module can be inspected anywhere
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


_TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

_RETCODE_DONE = 10009
_RETCODE_DONE_PARTIAL = 10008          # request accepted (placed) — treat as success
_RETCODE_INVALID_FILL = 10030          # "Unsupported filling mode"


class MT5Broker(Broker):
    name = "mt5"

    def __init__(self, cfg) -> None:
        if mt5 is None:
            raise RuntimeError(
                "The MetaTrader5 package is not installed. It only works on Windows — "
                "run the bot with --mode paper here, and deploy live on a Windows machine."
            )
        self.cfg = cfg
        self.magic = int(cfg.get("engine.magic", 570123))
        self._info_cache: dict[str, SymbolInfo] = {}
        self._filling: dict[str, int] = {}

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        path = env("MT5_PATH")
        login = env_int("MT5_LOGIN")
        password = env("MT5_PASSWORD")
        server = env("MT5_SERVER")

        kwargs: dict = {}
        if path:
            kwargs["path"] = path
        if login and password and server:
            kwargs.update(login=login, password=password, server=server)

        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

        acct = mt5.account_info()
        if acct is None:
            mt5.shutdown()
            raise RuntimeError(f"mt5.account_info failed: {mt5.last_error()}")

        term = mt5.terminal_info()
        if term is not None and not term.trade_allowed:
            log.warning(
                "Algo Trading is DISABLED in the MT5 terminal — orders will be rejected. "
                "Enable the 'Algo Trading' toolbar button."
            )

        log.info(
            "Connected to MT5: %s #%s on %s | balance=%.2f %s | leverage=1:%d",
            acct.name, acct.login, acct.server, acct.balance, acct.currency, acct.leverage,
        )

        # Make sure every configured symbol is selected in Market Watch.
        for sym in self.cfg.enabled_symbols():
            name = sym["name"]
            if not mt5.symbol_select(name, True):
                raise RuntimeError(
                    f"symbol {name!r} is not available on this account. "
                    "Check the exact name in MT5 Market Watch (brokers add suffixes "
                    "like .m / _raw / .pro)."
                )

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:  # pragma: no cover - best effort on teardown
            pass

    # ---------------------------------------------------------------- market data
    def symbol_info(self, symbol: str) -> SymbolInfo:
        cached = self._info_cache.get(symbol)
        if cached is not None:
            return cached

        raw = mt5.symbol_info(symbol)
        if raw is None:
            raise RuntimeError(f"symbol_info({symbol}) returned None: {mt5.last_error()}")

        info = SymbolInfo(
            name=raw.name,
            digits=raw.digits,
            point=raw.point,
            tick_size=raw.trade_tick_size or raw.point,
            tick_value=raw.trade_tick_value,
            volume_min=raw.volume_min,
            volume_max=raw.volume_max,
            volume_step=raw.volume_step,
            contract_size=raw.trade_contract_size,
            stops_level_points=raw.trade_stops_level,
        )
        self._info_cache[symbol] = info
        self._filling[symbol] = self._pick_filling(raw)
        return info

    def _pick_filling(self, raw) -> int:
        """Choose a fill policy the broker actually supports for this symbol."""
        mode = getattr(raw, "filling_mode", 0)
        if mode & 1:                       # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if mode & 2:                       # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def tick(self, symbol: str) -> Tick:
        raw = mt5.symbol_info_tick(symbol)
        if raw is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
        return Tick(
            symbol=symbol,
            bid=raw.bid,
            ask=raw.ask,
            time=datetime.fromtimestamp(raw.time, tz=timezone.utc),
        )

    def candles(self, symbol: str, timeframe: str, count: int) -> list[Candle]:
        tf_name = _TIMEFRAMES.get(timeframe.upper())
        if tf_name is None:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        tf = getattr(mt5, tf_name)

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos({symbol}) failed: {mt5.last_error()}")

        return [
            Candle(
                time=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]),
            )
            for r in rates
        ]

    # ---------------------------------------------------------------- account
    def account(self) -> AccountInfo:
        raw = mt5.account_info()
        if raw is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")
        return AccountInfo(
            login=raw.login,
            balance=raw.balance,
            equity=raw.equity,
            margin=raw.margin,
            free_margin=raw.margin_free,
            currency=raw.currency,
            leverage=raw.leverage,
        )

    def positions(self, symbol: str | None = None) -> list[Position]:
        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if raw is None:
            return []
        out: list[Position] = []
        for p in raw:
            if p.magic and p.magic != self.magic:
                continue  # not ours — leave manual/other-EA trades alone
            out.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    side=Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL,
                    volume=p.volume,
                    price_open=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    profit=p.profit,
                    price_current=p.price_current,
                    time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                    magic=p.magic,
                    comment=p.comment,
                )
            )
        return out

    # ---------------------------------------------------------------- trading
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
        tick = self.tick(symbol)
        price = tick.price_for(side)

        sl, tp = self._respect_stops_level(info, side, price, sl, tp)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": info.normalize_volume(volume),
            "type": mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": info.normalize_price(sl),
            "tp": info.normalize_price(tp),
            "deviation": 20,
            "magic": self.magic,
            "comment": (comment or "BOT_MT5")[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling.get(symbol, mt5.ORDER_FILLING_IOC),
        }

        result = mt5.order_send(request)

        # Some brokers reject the filling mode we guessed — retry once with IOC/FOK.
        if result is not None and result.retcode == _RETCODE_INVALID_FILL:
            for fallback in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                if fallback == request["type_filling"]:
                    continue
                request["type_filling"] = fallback
                result = mt5.order_send(request)
                if result is not None and result.retcode in (_RETCODE_DONE, _RETCODE_DONE_PARTIAL):
                    self._filling[symbol] = fallback
                    break

        return self._to_order_result(result, "market_order")

    def modify_position(self, ticket: int, sl: float, tp: float) -> OrderResult:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(ok=False, error=f"position {ticket} not found")
        pos = positions[0]
        info = self.symbol_info(pos.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": info.normalize_price(sl),
            "tp": info.normalize_price(tp),
            "magic": self.magic,
        }
        return self._to_order_result(mt5.order_send(request), "modify_position")

    def close_position(self, ticket: int) -> OrderResult:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(ok=False, error=f"position {ticket} not found")
        pos = positions[0]

        side = Side.BUY if pos.type == mt5.POSITION_TYPE_BUY else Side.SELL
        tick = self.tick(pos.symbol)
        close_side = side.opposite

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if side is Side.BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.price_for(close_side),
            "deviation": 20,
            "magic": self.magic,
            "comment": "BOT_MT5 close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling.get(pos.symbol, mt5.ORDER_FILLING_IOC),
        }
        return self._to_order_result(mt5.order_send(request), "close_position")

    def closed_trade_info(self, ticket: int) -> tuple[float, float, str] | None:
        """Look the closing deal up in history so alerts show the real fill."""
        try:
            deals = mt5.history_deals_get(position=ticket)
        except Exception as exc:  # pragma: no cover - terminal quirks
            log.debug("history_deals_get(%s) failed: %s", ticket, exc)
            return None
        if not deals:
            return None

        exits = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if not exits:
            return None

        last = exits[-1]
        pnl = sum(d.profit + d.swap + d.commission for d in exits)
        reason = {
            mt5.DEAL_REASON_SL: "stop loss",
            mt5.DEAL_REASON_TP: "take profit",
            mt5.DEAL_REASON_SO: "stop out",
            mt5.DEAL_REASON_EXPERT: "bot",
            mt5.DEAL_REASON_CLIENT: "manual",
        }.get(last.reason, "closed")
        return float(last.price), float(pnl), reason

    # ---------------------------------------------------------------- helpers
    def _respect_stops_level(
        self, info: SymbolInfo, side: Side, price: float, sl: float, tp: float
    ) -> tuple[float, float]:
        """Push SL/TP out if they sit inside the broker's minimum stop distance."""
        min_dist = info.stops_level_points * info.point
        if min_dist <= 0:
            return sl, tp

        if side is Side.BUY:
            sl = min(sl, price - min_dist)
            tp = max(tp, price + min_dist)
        else:
            sl = max(sl, price + min_dist)
            tp = min(tp, price - min_dist)
        return sl, tp

    @staticmethod
    def _to_order_result(result, what: str) -> OrderResult:
        if result is None:
            err = mt5.last_error()
            log.error("%s: order_send returned None (%s)", what, err)
            return OrderResult(ok=False, error=f"order_send returned None: {err}")

        ok = result.retcode in (_RETCODE_DONE, _RETCODE_DONE_PARTIAL)
        if not ok:
            log.error("%s failed: retcode=%s %s", what, result.retcode, result.comment)

        return OrderResult(
            ok=ok,
            ticket=getattr(result, "order", 0) or getattr(result, "deal", 0),
            price=getattr(result, "price", 0.0),
            volume=getattr(result, "volume", 0.0),
            error="" if ok else f"retcode={result.retcode}: {result.comment}",
            retcode=result.retcode,
        )
