"""Episodic memory: the bot's own closed trades as searchable experience.

Backed entirely by the existing journal (signals ⋈ trades), so it backfills
from history automatically and there is no second write path to drift out of
sync. An "episode" is an executed signal whose trade has closed, expressed as
(feature vector, outcome in R).

Two consumers:
  * the AI gate — gets a summary of the k most similar past setups
  * the sizer — gets a bounded multiplier in [floor, 1.0] via shrunk expectancy

Shrinkage: E = (Σ r_i + m·μ) / (k + m), pulling thin evidence toward the global
mean μ, so a handful of unlucky neighbours can't halve your size. Only
*negative* shrunk expectancy reduces size; good evidence never inflates it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from core.journal import Journal
from core.learn.features import distance, vector_from_context
from core.models import Signal

log = logging.getLogger("learn.memory")


@dataclass
class Evidence:
    n: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    shrunk_r: float = 0.0
    multiplier: float = 1.0
    note: str = "insufficient history"
    neighbors: list[dict] = field(default_factory=list)   # compact rows for the AI

    def payload(self) -> dict:
        """What the AI gate sees."""
        return {
            "similar_setups_found": self.n,
            "their_win_rate_pct": round(self.win_rate, 1),
            "their_avg_r": round(self.avg_r, 3),
            "shrunk_expectancy_r": round(self.shrunk_r, 3),
            "note": self.note,
            "closest_examples": self.neighbors[:5],
        }


class EpisodeMemory:
    def __init__(self, journal: Journal, cfg) -> None:
        self.journal = journal
        self.k = int(cfg.get("learning.evidence.k", 25))
        self.min_neighbors = int(cfg.get("learning.evidence.min_neighbors", 10))
        self.shrink_m = float(cfg.get("learning.evidence.shrink_m", 15))
        self.floor = float(cfg.get("learning.evidence.floor", 0.5))
        self.enabled = bool(cfg.get("learning.evidence.enabled", True))

    # ------------------------------------------------------------------ episodes
    def episodes(self, symbol: str) -> list[tuple[list[float], float]]:
        """(vector, outcome_r) for every closed executed trade on this symbol."""
        rows = self.journal.conn.execute(
            """
            SELECT s.side, s.context_json, s.ts, s.risk_amount, t.pnl
              FROM signals s
              JOIN trades  t ON t.ticket = s.ticket
             WHERE s.outcome = 'executed'
               AND s.symbol = ?
               AND t.close_time IS NOT NULL
               AND s.risk_amount > 0
             ORDER BY s.ts DESC
             LIMIT 5000
            """,
            (symbol,),
        ).fetchall()

        out: list[tuple[list[float], float]] = []
        for row in rows:
            try:
                context = json.loads(row["context_json"] or "{}")
                ts = datetime.fromisoformat(row["ts"]) if row["ts"] else None
            except (ValueError, TypeError):
                continue
            vec = vector_from_context(row["side"], context, ts)
            if vec is None:
                continue
            out.append((vec, float(row["pnl"]) / float(row["risk_amount"])))
        return out

    def global_mean_r(self) -> float:
        row = self.journal.conn.execute(
            """
            SELECT AVG(t.pnl / s.risk_amount) AS mu
              FROM signals s JOIN trades t ON t.ticket = s.ticket
             WHERE s.outcome = 'executed' AND t.close_time IS NOT NULL
               AND s.risk_amount > 0
            """
        ).fetchone()
        return float(row["mu"]) if row and row["mu"] is not None else 0.0

    # ------------------------------------------------------------------ evidence
    def evidence_for(self, signal: Signal) -> Evidence:
        if not self.enabled:
            return Evidence(note="evidence disabled")

        query = vector_from_context(signal.side.value, signal.context, signal.bar_time)
        if query is None:
            return Evidence(note="signal context incomplete")

        episodes = self.episodes(signal.symbol)
        if len(episodes) < self.min_neighbors:
            return Evidence(
                n=len(episodes),
                note=f"only {len(episodes)} closed trades on {signal.symbol} — "
                     f"no adjustment until {self.min_neighbors}",
            )

        ranked = sorted(episodes, key=lambda ep: distance(query, ep[0]))[: self.k]
        outcomes = [r for _, r in ranked]
        n = len(outcomes)
        wins = sum(1 for r in outcomes if r > 0)
        avg_r = sum(outcomes) / n
        shrunk = shrunk_expectancy(outcomes, self.global_mean_r(), self.shrink_m)
        mult = multiplier_from_expectancy(shrunk, self.floor)

        note = "similar setups historically "
        note += "profitable" if shrunk >= 0 else f"unprofitable — size scaled to {mult:.0%}"

        neighbors = [
            {"outcome_r": round(r, 2)} for _, r in ranked[:5]
        ]
        return Evidence(
            n=n,
            win_rate=100.0 * wins / n,
            avg_r=avg_r,
            shrunk_r=shrunk,
            multiplier=mult,
            note=note,
            neighbors=neighbors,
        )


# ---------------------------------------------------------------- pure math
def shrunk_expectancy(outcomes: list[float], global_mean: float, m: float) -> float:
    """Bayesian-shrunk mean: thin evidence is pulled toward the global mean."""
    n = len(outcomes)
    if n == 0:
        return global_mean
    return (sum(outcomes) + m * global_mean) / (n + m)


def multiplier_from_expectancy(shrunk_r: float, floor: float) -> float:
    """Map shrunk expectancy to a size multiplier.

    >= 0        -> 1.0   (good evidence never INCREASES size — that would let
                          a lucky streak compound risk)
    0 .. -0.2R  -> linear decay 1.0 -> floor
    <= -0.2R    -> floor
    """
    if shrunk_r >= 0:
        return 1.0
    frac = min(1.0, -shrunk_r / 0.2)
    return round(max(floor, 1.0 - frac * (1.0 - floor)), 3)
