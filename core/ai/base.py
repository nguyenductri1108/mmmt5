"""Shared contract for AI advisors: one prompt, one JSON schema, two providers.

Design notes
------------
* The model is a **veto layer**, not a signal generator. The deterministic
  strategy decides *what* to trade; the model only gets to say "not this one",
  and optionally shrink the size. It can never invent a trade, move a stop, or
  widen risk — that keeps the failure mode bounded.
* Output is schema-constrained JSON so the engine never parses prose.
* Everything is clamped after parsing: a model that returns confidence 7.5 or
  size_multiplier 3.0 cannot enlarge a position.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from core.models import AccountInfo, AIVerdict, Position, Signal

log = logging.getLogger("ai")

# Kept byte-stable so Claude's prompt cache keeps hitting across calls.
SYSTEM_PROMPT = """\
You are the risk-review stage of an automated futures/FX/metals trading system.

A deterministic, rule-based strategy has already produced a trade candidate and
sized it. Your ONLY job is to decide whether this specific candidate should be
allowed through, and optionally to reduce its size. You are a veto, not a
signal generator.

You cannot: propose a different trade, change direction, move the stop or
target, or increase size. Those are fixed by the strategy and the risk engine.

Approve by default. The strategy has a documented edge and second-guessing it
on every trade destroys that edge. Reject only when you can point to a concrete,
specific problem in the data you were given. Examples of a real problem:
  - The stop sits on the wrong side of obvious structure (e.g. a long whose stop
    is above a nearby swing low it will almost certainly tag first).
  - The reward:risk stated is materially worse than the strategy's own target.
  - The trade is directly against a much stronger higher-timeframe signal that
    is visible in the context provided.
  - Volatility context makes the stop implausible (e.g. stop distance is a small
    fraction of the average bar range).
  - Risk state is already stretched: heavy drawdown today, correlated exposure
    already open in the same direction.

Not a reason to reject: general market uncertainty, "could go either way",
absence of news/fundamental data, a preference for a different entry, or the
fact that the trade might lose. Every trade might lose — that is priced in by
the position size.

Use size_multiplier between 0.5 and 1.0 when the setup is valid but something
in the context argues for caution. Use 1.0 when you see nothing to flag. Never
return a multiplier above 1.0.

Be specific and terse. Each reason and risk must be one short sentence that
refers to an actual number from the payload. No hedging, no boilerplate.
"""

# Structured-output schemas reject numeric bounds (minimum/maximum), so ranges
# live in the prompt and are enforced by clamping in `parse_verdict`.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "reject"],
            "description": "Whether this trade candidate may be executed.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the decision, 0.0 to 1.0.",
        },
        "size_multiplier": {
            "type": "number",
            "description": "Scale factor for the position size, 0.0 to 1.0. Use 1.0 unless reducing.",
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 short, specific justifications citing numbers from the payload.",
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "0-3 concrete risks to this specific trade. Empty list if none stand out.",
        },
    },
    "required": ["decision", "confidence", "size_multiplier", "reasons", "risks"],
    "additionalProperties": False,
}


class Advisor(ABC):
    """One AI provider."""

    provider: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """False when the SDK is missing or no API key is configured."""

    @abstractmethod
    def review(self, payload: dict[str, Any], lessons: str = "") -> AIVerdict:
        """Send the payload, return a parsed verdict. Raise on transport failure.

        `lessons` is an optional extra system block of journal-derived rules
        (see core/learn/lessons.py) — kept separate from SYSTEM_PROMPT so the
        stable prefix stays byte-identical for prompt caching.
        """

    @abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 8000,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """One-shot schema-constrained JSON call (used by the weekly analyst)."""


def build_review_payload(
    signal: Signal,
    account: AccountInfo,
    open_positions: list[Position],
    risk_amount: float,
    daily_pnl_pct: float,
    trades_today: int,
    recent_trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything the model is allowed to see, as a flat JSON-able dict."""
    return {
        "candidate": {
            "symbol": signal.symbol,
            "direction": signal.side.value,
            "timeframe": signal.timeframe,
            "entry": signal.entry,
            "stop_loss": signal.sl,
            "take_profit": signal.tp,
            "stop_distance": round(signal.risk_distance, 6),
            "target_distance": round(signal.reward_distance, 6),
            "reward_to_risk": round(signal.rr, 2),
            "planned_volume_lots": signal.volume,
            "money_at_risk": round(risk_amount, 2),
            "risk_pct_of_equity": round(
                100.0 * risk_amount / account.equity, 3
            ) if account.equity else 0.0,
            "strategy_rationale": signal.reason,
        },
        "market_context": signal.context,
        "account": {
            "equity": round(account.equity, 2),
            "balance": round(account.balance, 2),
            "currency": account.currency,
            "free_margin_pct": round(account.free_margin_pct, 1),
        },
        "risk_state": {
            "day_pnl_pct": round(daily_pnl_pct, 2),
            "trades_today": trades_today,
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.side.value,
                    "volume": p.volume,
                    "open_r_multiple": round(p.r_multiple(), 2),
                    "floating_pnl": round(p.profit, 2),
                }
                for p in open_positions
            ],
        },
        "recent_closed_trades": (recent_trades or [])[-10:],
    }


def render_user_message(payload: dict[str, Any]) -> str:
    return (
        "Review this trade candidate and return your verdict as JSON.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )


def parse_verdict(raw: str, provider: str, model: str, latency_ms: int) -> AIVerdict:
    """Turn the model's JSON into a clamped, trustworthy verdict."""
    data = json.loads(raw)

    decision = str(data.get("decision", "reject")).strip().lower()
    if decision not in ("approve", "reject"):
        decision = "reject"

    confidence = _clamp(data.get("confidence", 0.0), 0.0, 1.0)
    # Hard ceiling of 1.0: the model may only ever shrink a position.
    size_multiplier = _clamp(data.get("size_multiplier", 1.0), 0.0, 1.0)

    return AIVerdict(
        decision=decision,
        confidence=confidence,
        size_multiplier=size_multiplier,
        reasons=_string_list(data.get("reasons")),
        risks=_string_list(data.get("risks")),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
    )


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _string_list(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value[:limit] if str(v).strip()]
