"""Indicator helpers. Pure pandas, no TA library dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.models import Candle


def to_frame(candles: list[Candle]) -> pd.DataFrame:
    """Candles -> DataFrame indexed by bar time."""
    df = pd.DataFrame(
        [
            dict(time=c.time, open=c.open, high=c.high, low=c.low, close=c.close, volume=c.volume)
            for c in candles
        ]
    )
    if not df.empty:
        df = df.set_index("time")
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (the same smoothing MT5 uses)."""
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend-strength filter."""
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)

    # np.nan (not pd.NA) — pd.NA turns the Series into object dtype, which
    # .ewm() then refuses to aggregate.
    tr_smooth = true_range(df).ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def swing_high(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["high"].tail(lookback).max())


def swing_low(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["low"].tail(lookback).min())
