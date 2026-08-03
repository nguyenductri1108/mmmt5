"""Turn a signal's indicator context into a fixed numeric vector.

Direction-normalised: signed features are multiplied by the trade's sign, so a
with-trend long pullback and a with-trend short pullback map to the *same*
region of feature space. That doubles the effective sample per pattern.

All scales are fixed constants (not fitted), so a vector computed today is
comparable with one computed a year ago — no silent drift in the memory.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

# Order matters: vectors are only comparable if built by the same code version.
FEATURE_VERSION = 1

_REQUIRED = ("adx", "rsi", "dist_to_ema_fast_atr", "bar_range_atr",
             "ema_fast", "ema_slow", "ema_trend", "atr", "close")


def vector_from_context(
    side: str, context: dict[str, Any], ts: datetime | None
) -> list[float] | None:
    """Context dict (as stored in the journal) -> feature vector, or None."""
    if not all(k in context for k in _REQUIRED):
        return None
    try:
        sign = 1.0 if side == "BUY" else -1.0
        atr = float(context["atr"])
        if atr <= 0:
            return None

        adx = _clamp(float(context["adx"]) / 50.0, 0.0, 1.5)
        rsi_aligned = _clamp((float(context["rsi"]) - 50.0) / 50.0 * sign, -1.0, 1.0)
        dist_fast = _clamp(float(context["dist_to_ema_fast_atr"]) * sign / 2.0, -1.5, 1.5)
        trend_gap = _clamp(
            (float(context["ema_fast"]) - float(context["ema_slow"])) / atr * sign / 2.0,
            -2.0, 2.0,
        )
        trend_dist = _clamp(
            (float(context["close"]) - float(context["ema_trend"])) / atr * sign / 4.0,
            -2.0, 2.0,
        )
        bar_range = _clamp(float(context["bar_range_atr"]) / 2.0, 0.0, 2.0)

        if ts is not None:
            hour_angle = 2.0 * math.pi * (ts.hour + ts.minute / 60.0) / 24.0
            hour_sin, hour_cos = math.sin(hour_angle), math.cos(hour_angle)
        else:
            hour_sin = hour_cos = 0.0

        return [adx, rsi_aligned, dist_fast, trend_gap, trend_dist, bar_range,
                hour_sin, hour_cos]
    except (TypeError, ValueError):
        return None


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
