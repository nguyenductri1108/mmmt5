"""EMA trend + pullback entry, ATR-based stop, fixed R:R target.

Rules (long; short is the mirror):
  1. Regime   — close > EMA200 and EMA_fast > EMA_slow
  2. Strength — ADX >= adx_min (skip chop)
  3. Trigger  — the bar pulled back to within `pullback_atr` * ATR of EMA_fast
                and closed back above EMA_fast (a bullish rejection)
  4. Stop     — entry - sl_atr_mult * ATR, but never inside the recent swing low
  5. Target   — entry + tp_r_multiple * risk

It's deliberately plain: the point is a signal source that is auditable and
easy to reason about, with the AI layer sitting on top as the judgement call.
"""

from __future__ import annotations

import logging

from core.indicators import adx, atr, ema, rsi, swing_high, swing_low, to_frame
from core.models import Candle, Side, Signal, SymbolInfo
from core.strategy.base import Strategy

log = logging.getLogger("strategy.ema_atr")


class EmaAtrStrategy(Strategy):
    name = "ema_atr"

    def generate(
        self, symbol: str, timeframe: str, candles: list[Candle], info: SymbolInfo
    ) -> Signal | None:
        fast_p = int(self.p("ema_fast", 20))
        slow_p = int(self.p("ema_slow", 50))
        trend_p = int(self.p("ema_trend", 200))
        atr_p = int(self.p("atr_period", 14))
        adx_p = int(self.p("adx_period", 14))
        adx_min = float(self.p("adx_min", 18))
        pullback_atr = float(self.p("pullback_atr", 0.8))
        sl_mult = float(self.p("sl_atr_mult", 1.5))
        tp_r = float(self.p("tp_r_multiple", 2.0))
        min_atr_points = float(self.p("min_atr_points", 5))

        need = max(trend_p, slow_p, atr_p, adx_p) + 10
        if len(candles) < need:
            return None

        df = to_frame(candles)
        if df.empty:
            return None

        df["ema_fast"] = ema(df["close"], fast_p)
        df["ema_slow"] = ema(df["close"], slow_p)
        df["ema_trend"] = ema(df["close"], trend_p)
        df["atr"] = atr(df, atr_p)
        df["adx"] = adx(df, adx_p)
        df["rsi"] = rsi(df["close"], 14)

        # The engine calls us on a bar close, so the forming bar is the last row.
        bar = df.iloc[-2]
        prev = df.iloc[-3]
        bar_time = df.index[-2]

        atr_now = float(bar["atr"])
        if atr_now <= 0 or atr_now / info.point < min_atr_points:
            return None
        if float(bar["adx"]) < adx_min:
            return None

        close = float(bar["close"])
        ema_f = float(bar["ema_fast"])
        ema_s = float(bar["ema_slow"])
        ema_t = float(bar["ema_trend"])

        uptrend = close > ema_t and ema_f > ema_s
        downtrend = close < ema_t and ema_f < ema_s
        if not (uptrend or downtrend):
            return None

        side: Side | None = None
        if uptrend:
            touched = float(bar["low"]) <= ema_f + pullback_atr * atr_now
            rejected = close > ema_f and close > float(bar["open"])
            momentum = close > float(prev["close"])
            if touched and rejected and momentum:
                side = Side.BUY
        else:
            touched = float(bar["high"]) >= ema_f - pullback_atr * atr_now
            rejected = close < ema_f and close < float(bar["open"])
            momentum = close < float(prev["close"])
            if touched and rejected and momentum:
                side = Side.SELL

        if side is None:
            return None

        entry = close
        raw_stop = sl_mult * atr_now

        if side is Side.BUY:
            structural = swing_low(df.iloc[:-1], 20) - 0.2 * atr_now
            sl = min(entry - raw_stop, structural)
            risk = entry - sl
            tp = entry + tp_r * risk
        else:
            structural = swing_high(df.iloc[:-1], 20) + 0.2 * atr_now
            sl = max(entry + raw_stop, structural)
            risk = sl - entry
            tp = entry - tp_r * risk

        if risk <= 0:
            return None

        # A stop wider than 3 ATR means structure is far away — the trade is
        # no longer the setup we screened for.
        if risk > 3.0 * atr_now:
            return None

        reason = (
            f"{'Up' if side is Side.BUY else 'Down'}trend "
            f"(EMA{fast_p}{'>' if side is Side.BUY else '<'}EMA{slow_p}, price "
            f"{'above' if side is Side.BUY else 'below'} EMA{trend_p}), "
            f"pullback to EMA{fast_p} rejected, ADX {float(bar['adx']):.0f}"
        )

        return Signal(
            symbol=symbol,
            side=side,
            entry=info.normalize_price(entry),
            sl=info.normalize_price(sl),
            tp=info.normalize_price(tp),
            reason=reason,
            timeframe=timeframe,
            atr=atr_now,
            bar_time=bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time,
            context={
                "close": round(close, info.digits),
                "ema_fast": round(ema_f, info.digits),
                "ema_slow": round(ema_s, info.digits),
                "ema_trend": round(ema_t, info.digits),
                "atr": round(atr_now, info.digits),
                "atr_points": round(atr_now / info.point, 1),
                "adx": round(float(bar["adx"]), 1),
                "rsi": round(float(bar["rsi"]), 1),
                "dist_to_ema_fast_atr": round((close - ema_f) / atr_now, 2),
                "swing_high_20": round(swing_high(df.iloc[:-1], 20), info.digits),
                "swing_low_20": round(swing_low(df.iloc[:-1], 20), info.digits),
                "last_5_closes": [round(float(x), info.digits) for x in df["close"].iloc[-6:-1]],
                "bar_range_atr": round(
                    (float(bar["high"]) - float(bar["low"])) / atr_now, 2
                ),
            },
        )
