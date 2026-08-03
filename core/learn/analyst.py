"""The weekly AI analyst — the reasoning half of the learning loop.

Runs off the trading thread. Reads the journal (closed trades with full
indicator context, gate performance, current lessons and parameters), and
returns structured output:

  * lessons to keep / drop / add  -> injected into the gate prompt
  * parameter proposals           -> candidates for the walk-forward optimizer

The analyst never changes anything itself. Its parameter ideas still have to
survive the same out-of-sample validation as random candidates, and its
lessons are re-validated (or expire) every subsequent week. If no AI provider
is available the whole step is skipped and the loop continues without it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core.journal import Journal
from core.learn.calibrator import GateAssessment
from core.learn.lessons import Lesson, LessonsUpdate

log = logging.getLogger("learn.analyst")

ANALYST_SYSTEM = """\
You are the weekly performance analyst for an automated trading system. You are
given the system's own journal: closed trades with the exact indicator context
each was taken in, how the AI trade-gate voted, the currently active "lessons"
injected into that gate's prompt, and the current strategy parameters.

Your job:
1. LESSONS — maintain a small set of short, evidence-cited rules for the trade
   gate. Keep a lesson only if the fresh data still supports it. Drop lessons
   the data no longer supports. Add a new lesson only when you can cite a
   pattern backed by at least 10 trades in the data provided. Each lesson must
   reference concrete numbers (a threshold, a win rate, an average R).
2. PARAMETER PROPOSALS — optionally propose up to N alternative strategy
   parameter sets that a walk-forward backtest will evaluate against the
   incumbent. Only propose a change when the trade data suggests a specific
   hypothesis (e.g. "losses cluster at ADX 18-22, raising adx_min may help").
   Every value must stay inside the provided bounds. Do not propose cosmetic
   changes — each proposal needs a stated hypothesis.

Be statistical, not narrative. Small samples prove nothing: with fewer than 10
trades supporting a pattern, say nothing. It is perfectly acceptable to return
zero new lessons and zero proposals — most weeks that is the right answer.
"""

ANALYST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_lesson_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "IDs of existing lessons the fresh data still supports.",
        },
        "drop_lesson_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "IDs of existing lessons to retire.",
        },
        "new_lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The rule, one sentence, citing concrete numbers.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Sample size and effect, e.g. '14 trades, avg -0.42R'.",
                    },
                },
                "required": ["text", "evidence"],
                "additionalProperties": False,
            },
        },
        "param_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "What pattern in the data motivates this change.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Only the parameters that differ from the incumbent.",
                        "properties": {
                            "ema_fast": {"type": "number"},
                            "ema_slow": {"type": "number"},
                            "adx_min": {"type": "number"},
                            "pullback_atr": {"type": "number"},
                            "sl_atr_mult": {"type": "number"},
                            "tp_r_multiple": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["hypothesis", "params"],
                "additionalProperties": False,
            },
        },
        "summary": {
            "type": "string",
            "description": "2-4 sentences on the week: what is working, what is not.",
        },
    },
    "required": ["keep_lesson_ids", "drop_lesson_ids", "new_lessons",
                 "param_proposals", "summary"],
    "additionalProperties": False,
}

_INT_PARAMS = {"ema_fast", "ema_slow", "adx_min"}


@dataclass
class AnalystResult:
    lessons_update: LessonsUpdate
    param_proposals: list[dict[str, Any]] = field(default_factory=list)  # merged full sets
    summary: str = ""
    ran: bool = False


def gather_dataset(
    journal: Journal,
    gate: GateAssessment,
    lessons: list[Lesson],
    current_params: dict[str, Any],
    bounds: dict[str, list[float]],
    days: int = 90,
    max_trades: int = 150,
) -> dict[str, Any]:
    """Everything the analyst sees, as one JSON-able dict."""
    rows = journal.conn.execute(
        """
        SELECT s.symbol, s.side, s.ts, s.rr, s.ai_decision, s.ai_confidence,
               s.risk_amount, s.context_json, t.pnl, t.close_reason
          FROM signals s
          JOIN trades  t ON t.ticket = s.ticket
         WHERE s.outcome = 'executed'
           AND t.close_time IS NOT NULL
           AND s.risk_amount > 0
           AND s.ts >= datetime('now', ?)
         ORDER BY s.ts DESC
         LIMIT ?
        """,
        (f"-{days} days", max_trades),
    ).fetchall()

    trades = []
    for r in rows:
        try:
            context = json.loads(r["context_json"] or "{}")
        except ValueError:
            context = {}
        trades.append({
            "symbol": r["symbol"],
            "side": r["side"],
            "ts": r["ts"],
            "outcome_r": round(float(r["pnl"]) / float(r["risk_amount"]), 3),
            "exit": r["close_reason"],
            "planned_rr": r["rr"],
            "ai": {"decision": r["ai_decision"], "confidence": r["ai_confidence"]},
            "context": {
                k: context.get(k)
                for k in ("adx", "rsi", "dist_to_ema_fast_atr", "bar_range_atr",
                          "atr_points")
                if k in context
            },
        })

    return {
        "closed_trades": trades,
        "gate_assessment": {
            "mode": gate.mode,
            "n_approved": gate.n_approved,
            "n_rejected_measured": gate.n_rejected,
            "approved_avg_r": round(gate.approved_avg_r, 3),
            "rejected_avg_r": round(gate.rejected_avg_r, 3),
        },
        "active_lessons": [
            {"id": lesson.id, "text": lesson.text, "evidence": lesson.evidence}
            for lesson in lessons
        ],
        "current_params": current_params,
        "param_bounds": bounds,
    }


def run_analyst(ai_router, dataset: dict[str, Any], cfg) -> AnalystResult:
    """Call the LLM. On any failure, return a no-op result — the loop continues."""
    empty = AnalystResult(lessons_update=LessonsUpdate(
        keep_ids=[l["id"] for l in dataset.get("active_lessons", [])],
        drop_ids=[], new=[],
    ))

    if not bool(cfg.get("learning.analyst.enabled", True)):
        return empty

    max_proposals = int(cfg.get("learning.analyst.max_param_proposals", 3))
    user = (
        "Here is this system's journal for review. Return your analysis as JSON.\n\n"
        f"```json\n{json.dumps(dataset, indent=1, default=str)}\n```"
    )

    raw = ai_router.complete_json(
        system=ANALYST_SYSTEM,
        user=user,
        schema=ANALYST_SCHEMA,
        max_tokens=16000,
        effort=str(cfg.get("learning.analyst.effort", "xhigh")),
    )
    if raw is None:
        log.warning("analyst skipped — no AI provider succeeded")
        return empty

    bounds = dict(cfg.get("learning.optimizer.bounds", {}) or {})
    incumbent = dict(dataset.get("current_params", {}))

    proposals: list[dict[str, Any]] = []
    for prop in (raw.get("param_proposals") or [])[:max_proposals]:
        clamped = clamp_proposal(prop.get("params") or {}, incumbent, bounds)
        if clamped is not None:
            proposals.append(clamped)

    update = LessonsUpdate(
        keep_ids=[int(i) for i in (raw.get("keep_lesson_ids") or []) if isinstance(i, (int, float))],
        drop_ids=[int(i) for i in (raw.get("drop_lesson_ids") or []) if isinstance(i, (int, float))],
        new=[n for n in (raw.get("new_lessons") or []) if isinstance(n, dict)],
    )

    return AnalystResult(
        lessons_update=update,
        param_proposals=proposals,
        summary=str(raw.get("summary", ""))[:800],
        ran=True,
    )


def clamp_proposal(
    partial: dict[str, Any],
    incumbent: dict[str, Any],
    bounds: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Merge a partial proposal over the incumbent and clamp every value.

    Keys outside the bounds whitelist are DISCARDED — the analyst cannot smuggle
    a change to anything not explicitly optimizable. Returns None if nothing
    survives (no-op proposal) or the result is structurally invalid.
    """
    merged = dict(incumbent)
    changed = False
    for key, value in partial.items():
        if key not in bounds:
            continue
        try:
            low, high = float(bounds[key][0]), float(bounds[key][1])
            v = max(low, min(high, float(value)))
        except (TypeError, ValueError, IndexError):
            continue
        if key in _INT_PARAMS:
            v = int(round(v))
        if merged.get(key) != v:
            merged[key] = v
            changed = True

    if not changed:
        return None
    # Structural sanity the bounds alone can't express.
    if float(merged.get("ema_fast", 1)) >= float(merged.get("ema_slow", 999)):
        return None
    return merged
