"""Gate calibration: is the AI veto actually earning its keep?

The measurement problem: in enforce mode, rejected signals are never traded, so
there is no outcome to learn from. The fix is built into the engine — a random
`audit_fraction` of AI-rejected signals is executed anyway, tagged in the
journal as executed-with-ai_decision='reject'. Those audit trades (plus every
trade in shadow mode) form the measured counterfactual this module runs on.

Decision policy, evidence-gated:
  rejected-bucket avg R < enforce_below_r  (they lose money) -> ENFORCE
  rejected-bucket avg R > shadow_above_r   (they make money) -> SHADOW
  in between, or not enough samples                          -> keep current mode

min_confidence is re-chosen weekly from the grid: the threshold whose
would-approve set has the best expectancy, subject to a minimum sample.
All pure functions on plain rows -> unit-testable without a database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.journal import Journal

log = logging.getLogger("learn.calibrator")


@dataclass
class GateOutcomeRow:
    ai_decision: str          # approve | reject
    confidence: float
    outcome_r: float


@dataclass
class GateAssessment:
    n_approved: int = 0
    n_rejected: int = 0
    approved_avg_r: float = 0.0
    rejected_avg_r: float = 0.0
    mode: str = "shadow"
    mode_reason: str = ""
    min_confidence: float = 0.55
    confidence_reason: str = ""
    rows: int = 0
    per_threshold: dict[float, tuple[int, float]] = field(default_factory=dict)


def fetch_gate_rows(journal: Journal, days: int = 90) -> list[GateOutcomeRow]:
    """Executed AND closed trades that carry an AI verdict, with outcome in R."""
    rows = journal.conn.execute(
        """
        SELECT s.ai_decision, s.ai_confidence, s.risk_amount, t.pnl
          FROM signals s
          JOIN trades  t ON t.ticket = s.ticket
         WHERE s.outcome = 'executed'
           AND t.close_time IS NOT NULL
           AND s.risk_amount > 0
           AND s.ai_decision IN ('approve', 'reject')
           AND s.ai_provider NOT IN ('disabled', 'fallback')
           AND s.ts >= datetime('now', ?)
        """,
        (f"-{days} days",),
    ).fetchall()
    return [
        GateOutcomeRow(
            ai_decision=r["ai_decision"],
            confidence=float(r["ai_confidence"] or 0.0),
            outcome_r=float(r["pnl"]) / float(r["risk_amount"]),
        )
        for r in rows
    ]


def assess(
    rows: list[GateOutcomeRow],
    current_mode: str,
    current_min_confidence: float,
    cfg,
) -> GateAssessment:
    min_rejects = int(cfg.get("learning.gate.min_measured_rejects", 15))
    enforce_below = float(cfg.get("learning.gate.enforce_below_r", -0.05))
    shadow_above = float(cfg.get("learning.gate.shadow_above_r", 0.05))
    grid = list(cfg.get("learning.gate.confidence_grid", [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]))
    min_sample = int(cfg.get("learning.gate.confidence_min_sample", 30))

    approved = [r.outcome_r for r in rows if r.ai_decision == "approve"]
    rejected = [r.outcome_r for r in rows if r.ai_decision == "reject"]

    out = GateAssessment(
        n_approved=len(approved),
        n_rejected=len(rejected),
        approved_avg_r=_mean(approved),
        rejected_avg_r=_mean(rejected),
        mode=current_mode,
        min_confidence=current_min_confidence,
        rows=len(rows),
    )

    # --- mode decision -----------------------------------------------------
    if len(rejected) < min_rejects:
        out.mode_reason = (
            f"only {len(rejected)}/{min_rejects} measured reject-outcomes — "
            f"keeping {current_mode}"
        )
    elif out.rejected_avg_r < enforce_below:
        out.mode = "enforce"
        out.mode_reason = (
            f"rejected trades average {out.rejected_avg_r:+.3f}R over "
            f"{len(rejected)} samples — the gate dodges losers, enforcing"
        )
    elif out.rejected_avg_r > shadow_above:
        out.mode = "shadow"
        out.mode_reason = (
            f"rejected trades average {out.rejected_avg_r:+.3f}R over "
            f"{len(rejected)} samples — the gate is blocking WINNERS, "
            "dropping to shadow"
        )
    else:
        out.mode_reason = (
            f"rejected bucket {out.rejected_avg_r:+.3f}R is inside the neutral "
            f"band [{enforce_below}, {shadow_above}] — keeping {current_mode}"
        )

    # --- min_confidence ----------------------------------------------------
    # For each threshold, the expectancy of trades the gate would have let
    # through: AI-approved AND confidence >= t.
    best_t, best_e, best_n = None, None, 0
    for t in grid:
        subset = [r.outcome_r for r in rows if r.ai_decision == "approve" and r.confidence >= t]
        out.per_threshold[t] = (len(subset), _mean(subset))
        if len(subset) >= min_sample:
            e = _mean(subset)
            if best_e is None or e > best_e + 1e-9:
                best_t, best_e, best_n = t, e, len(subset)

    if best_t is None:
        out.confidence_reason = (
            f"no threshold has >= {min_sample} approved outcomes — keeping "
            f"{current_min_confidence:.2f}"
        )
    elif abs(best_t - current_min_confidence) < 1e-9:
        out.confidence_reason = f"current {current_min_confidence:.2f} is already optimal"
    else:
        out.min_confidence = best_t
        out.confidence_reason = (
            f"threshold {best_t:.2f} gives {best_e:+.3f}R over {best_n} trades "
            f"(was {current_min_confidence:.2f})"
        )

    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
