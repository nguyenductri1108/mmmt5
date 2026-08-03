"""Champion/challenger parameter evolution with walk-forward validation.

Weekly, off the trading thread:

  candidates = incumbent + random perturbations (within bounds)
             + the analyst's hypotheses (already bounds-clamped)
  each is backtested over the SAME real-history window
  trades are split by time: older 70% in-sample, newest 30% out-of-sample
  a candidate is promoted only if, versus the incumbent:
      - it has enough out-of-sample trades to mean anything
      - OOS expectancy beats the incumbent by a margin (in R)
      - a bootstrap over OOS trade outcomes prefers it with high probability
      - it also holds up in-sample (guards against fitting the recent regime)
      - its drawdown isn't materially worse

Promotion just stages the params; the scheduler applies them on the main
thread and the live rollback guard (`should_rollback`) can undo them later.

Every gate here exists to answer one question honestly: "is this difference
edge, or noise?" — because an auto-tuner without that question is a machine
for overfitting.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from core.backtest import BTResult, run_quiet
from core.config import Config

log = logging.getLogger("learn.optimizer")

_INT_PARAMS = {"ema_fast", "ema_slow", "adx_min"}


@dataclass
class Split:
    """Three windows, cut on a clock shared by every candidate.

      is_r    train    — where params are free to fit
      oos_r   select   — where challengers are RANKED and gated
      hold_r  confirm  — never used for ranking; the winner must not fall apart
                         here, which is what stops the winner's curse
    """

    is_r: list[float] = field(default_factory=list)
    oos_r: list[float] = field(default_factory=list)
    hold_r: list[float] = field(default_factory=list)

    @property
    def is_expectancy(self) -> float:
        return _mean(self.is_r)

    @property
    def oos_expectancy(self) -> float:
        return _mean(self.oos_r)

    @property
    def hold_expectancy(self) -> float:
        return _mean(self.hold_r)


@dataclass
class CandidateEval:
    params: dict[str, Any]
    source: str                  # incumbent | random | analyst
    split: Split
    max_dd_pct: float
    trades: int


@dataclass
class OptimizerOutcome:
    ran: bool = False
    skipped_reason: str = ""
    promoted: dict[str, Any] | None = None
    promoted_source: str = ""
    promoted_oos_expectancy: float = 0.0
    incumbent_oos_expectancy: float = 0.0
    evidence: str = ""
    evaluated: int = 0


# ---------------------------------------------------------------- candidates
def generate_candidates(
    incumbent: dict[str, Any],
    bounds: dict[str, list[float]],
    analyst_proposals: list[dict[str, Any]],
    rng: random.Random,
    n_random: int = 12,
) -> list[tuple[dict[str, Any], str]]:
    """(params, source) list. Incumbent is always index 0."""
    out: list[tuple[dict[str, Any], str]] = [(dict(incumbent), "incumbent")]
    seen = {_key(incumbent)}

    for proposal in analyst_proposals:
        key = _key(proposal)
        if key not in seen:
            seen.add(key)
            out.append((proposal, "analyst"))

    attempts = 0
    while len([1 for _, s in out if s == "random"]) < n_random and attempts < n_random * 10:
        attempts += 1
        candidate = dict(incumbent)
        for name, (low, high) in bounds.items():
            low, high = float(low), float(high)
            base = float(candidate.get(name, (low + high) / 2))
            if rng.random() < 0.5:
                continue  # keep the incumbent's value for this dim
            if rng.random() < 0.5:
                value = rng.gauss(base, 0.1 * (high - low))   # local step
            else:
                value = rng.uniform(low, high)                 # exploration
            value = max(low, min(high, value))
            if name in _INT_PARAMS:
                value = int(round(value))
            else:
                value = round(value, 2)
            candidate[name] = value

        if float(candidate.get("ema_fast", 1)) >= float(candidate.get("ema_slow", 999)):
            continue
        key = _key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append((candidate, "random"))

    return out


# ---------------------------------------------------------------- evaluation
def evaluate_candidate(
    cfg: Config, params: dict[str, Any], source: str, bars: int
) -> tuple[dict[str, Any], str, BTResult] | None:
    """Backtest one candidate. Splitting happens later, on a shared clock."""
    try:
        result: BTResult = run_quiet(cfg, bars=bars, params=params)
    except Exception as exc:  # noqa: BLE001 - a broken candidate just drops out
        log.warning("candidate %s failed to evaluate: %s", params, exc)
        return None
    return params, source, result


def shared_cutoffs(
    results: list[BTResult], is_fraction: float, holdout_fraction: float
) -> tuple[Any, Any] | None:
    """(select_start, holdout_start) on the BAR window, shared by all candidates.

    Anchoring to each candidate's own first/last TRADE would give every
    candidate a different calendar window — a stricter filter that starts
    trading later would silently get a different, easier OOS period. Anchoring
    to the replayed bar window (identical across candidates on one feed)
    removes that.
    """
    starts = [r.window_start for r in results if r.window_start is not None]
    ends = [r.window_end for r in results if r.window_end is not None]
    if not starts or not ends:
        return None
    t0, t1 = min(starts), max(ends)
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return None
    select_frac = max(0.0, min(1.0, is_fraction))
    hold_frac = max(0.0, min(1.0 - select_frac, holdout_fraction))
    select_start = t0 + timedelta(seconds=span * select_frac)
    holdout_start = t0 + timedelta(seconds=span * (1.0 - hold_frac))
    return select_start, holdout_start


def split_by_time(
    result: BTResult, is_fraction: float, cutoffs: tuple[Any, Any] | None = None
) -> Split:
    """Cut a candidate's trades into train / select / holdout on a shared clock."""
    split = Split()
    if not result.r_series:
        return split

    if cutoffs is None:  # standalone use (no cross-candidate comparison)
        times = [t for t, _ in result.r_series]
        t0, t1 = min(times), max(times)
        span = (t1 - t0).total_seconds()
        if span <= 0:
            split.is_r = [r for _, r in result.r_series]
            return split
        select_start = t0 + timedelta(seconds=span * is_fraction)
        holdout_start = t1 + timedelta(seconds=1)   # no holdout
    else:
        select_start, holdout_start = cutoffs

    for when, r in result.r_series:
        if when < select_start:
            split.is_r.append(r)
        elif when < holdout_start:
            split.oos_r.append(r)
        else:
            split.hold_r.append(r)
    return split


# ---------------------------------------------------------------- promotion
def required_bootstrap_prob(base: float, n_challengers: int, correct: bool) -> float:
    """Bonferroni-style tightening for testing many challengers at once.

    Ranking ~15 candidates on one window and promoting the best is a multiple-
    comparisons problem: with enough draws, noise clears any fixed bar. A
    bootstrap probability `p` corresponds to a one-sided error of (1-p); to hold
    the FAMILY-wise error at (1-base) across K challengers, each must clear
    1 - (1-base)/K.
    """
    if not correct or n_challengers <= 1:
        return base
    return min(0.999, 1.0 - (1.0 - base) / n_challengers)


def passes_promotion(
    candidate: CandidateEval,
    incumbent: CandidateEval,
    cfg,
    rng: random.Random,
    n_challengers: int = 1,
) -> tuple[bool, str]:
    """The full gauntlet. Returns (promote?, human-readable reason)."""
    min_oos = int(cfg.get("learning.optimizer.min_oos_trades", 20))
    min_hold = int(cfg.get("learning.optimizer.min_holdout_trades", 10))
    margin = float(cfg.get("learning.optimizer.margin_r", 0.03))
    base_prob = float(cfg.get("learning.optimizer.bootstrap_prob", 0.75))
    correct = bool(cfg.get("learning.optimizer.multiplicity_correction", True))
    max_dd_ratio = float(cfg.get("learning.optimizer.max_dd_ratio", 1.3))

    c, i = candidate.split, incumbent.split

    if len(c.oos_r) < min_oos:
        return False, f"only {len(c.oos_r)}/{min_oos} selection-window trades"
    if len(i.oos_r) < min_oos:
        return False, f"incumbent has only {len(i.oos_r)} selection trades — too thin to compare"

    edge = c.oos_expectancy - i.oos_expectancy
    if edge < margin:
        return False, f"selection edge {edge:+.3f}R below margin {margin}R"

    # Regime-fit guard: a candidate that only wins in the newest window is
    # suspicious. It must not be materially worse than the incumbent in-sample.
    if c.is_expectancy < i.is_expectancy - 2 * margin:
        return False, (
            f"in-sample regression ({c.is_expectancy:+.3f}R vs "
            f"{i.is_expectancy:+.3f}R) — looks regime-fit"
        )

    if incumbent.max_dd_pct > 0 and candidate.max_dd_pct > incumbent.max_dd_pct * max_dd_ratio + 1.0:
        return False, (
            f"drawdown {candidate.max_dd_pct:.1f}% vs incumbent "
            f"{incumbent.max_dd_pct:.1f}% exceeds ratio {max_dd_ratio}"
        )

    need_prob = required_bootstrap_prob(base_prob, n_challengers, correct)
    prob = bootstrap_prob(c.oos_r, i.oos_r, n=2000, rng=rng)
    if prob < need_prob:
        return False, (
            f"bootstrap P(candidate>incumbent)={prob:.3f} < {need_prob:.3f} "
            f"(family-wise over {n_challengers} challengers)"
        )

    # Holdout confirmation — the winner was RANKED on the selection window, so
    # that window is contaminated by selection. The holdout was never ranked on;
    # the winner must simply not be worse than the incumbent there.
    if min_hold > 0:
        if len(c.hold_r) < min_hold or len(i.hold_r) < min_hold:
            return False, (
                f"holdout too thin ({len(c.hold_r)} vs incumbent {len(i.hold_r)}, "
                f"need {min_hold}) — cannot confirm the selection"
            )
        if c.hold_expectancy < i.hold_expectancy:
            return False, (
                f"failed holdout confirmation: {c.hold_expectancy:+.3f}R vs "
                f"incumbent {i.hold_expectancy:+.3f}R on data it was not selected on"
            )

    return True, (
        f"selection {c.oos_expectancy:+.3f}R vs {i.oos_expectancy:+.3f}R "
        f"({len(c.oos_r)} trades), bootstrap {prob:.3f} >= {need_prob:.3f}, "
        f"holdout {c.hold_expectancy:+.3f}R vs {i.hold_expectancy:+.3f}R"
    )


def bootstrap_prob(
    candidate_r: list[float],
    incumbent_r: list[float],
    n: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """P(mean of a resampled candidate beats mean of a resampled incumbent).

    Non-parametric — no distribution assumptions about trade outcomes, which
    are famously fat-tailed.
    """
    if not candidate_r or not incumbent_r:
        return 0.0
    rng = rng or random.Random(7)
    wins = 0
    for _ in range(n):
        ca = _mean([rng.choice(candidate_r) for _ in range(len(candidate_r))])
        ia = _mean([rng.choice(incumbent_r) for _ in range(len(incumbent_r))])
        if ca > ia:
            wins += 1
    return wins / n


# ---------------------------------------------------------------- entry point
def run_optimizer(
    cfg: Config,
    incumbent_params: dict[str, Any],
    analyst_proposals: list[dict[str, Any]],
    seed: int,
) -> OptimizerOutcome:
    if not bool(cfg.get("learning.optimizer.enabled", True)):
        return OptimizerOutcome(skipped_reason="optimizer disabled")

    if str(cfg.get("paper.feed", "synthetic")).lower() != "csv":
        return OptimizerOutcome(
            skipped_reason=(
                "no real history available (paper.feed is not csv) — refusing to "
                "optimize on synthetic data"
            )
        )

    bounds = dict(cfg.get("learning.optimizer.bounds", {}) or {})
    if not bounds:
        return OptimizerOutcome(skipped_reason="no learning.optimizer.bounds configured")

    bars = int(cfg.get("learning.optimizer.bars", 20000))
    is_fraction = float(cfg.get("learning.optimizer.is_fraction", 0.6))
    holdout_fraction = float(cfg.get("learning.optimizer.holdout_fraction", 0.15))
    n_random = int(cfg.get("learning.optimizer.n_random", 12))
    rng = random.Random(seed)

    candidates = generate_candidates(incumbent_params, bounds, analyst_proposals, rng, n_random)

    raw: list[tuple[dict[str, Any], str, BTResult]] = []
    for params, source in candidates:
        ev = evaluate_candidate(cfg, params, source, bars)
        if ev is not None:
            raw.append(ev)

    if not raw:
        return OptimizerOutcome(skipped_reason="no candidate evaluated successfully")

    # ONE clock for every candidate — see shared_cutoffs().
    cutoffs = shared_cutoffs([r for _, _, r in raw], is_fraction, holdout_fraction)
    evals = [
        CandidateEval(
            params=params, source=source,
            split=split_by_time(result, is_fraction, cutoffs),
            max_dd_pct=result.max_dd_pct, trades=result.trades,
        )
        for params, source, result in raw
    ]

    incumbent_eval = next((e for e in evals if e.source == "incumbent"), None)
    if incumbent_eval is None:
        return OptimizerOutcome(skipped_reason="incumbent failed to evaluate", evaluated=len(evals))

    outcome = OptimizerOutcome(
        ran=True,
        evaluated=len(evals),
        incumbent_oos_expectancy=incumbent_eval.split.oos_expectancy,
    )

    # Rank challengers on the SELECTION window; the winner still has to survive
    # the multiplicity-corrected bootstrap and the untouched holdout.
    challengers = sorted(
        (e for e in evals if e.source != "incumbent"),
        key=lambda e: e.split.oos_expectancy,
        reverse=True,
    )
    n_challengers = len(challengers)
    for challenger in challengers:
        ok, reason = passes_promotion(challenger, incumbent_eval, cfg, rng, n_challengers)
        if ok:
            outcome.promoted = challenger.params
            outcome.promoted_source = challenger.source
            # Report the HOLDOUT expectancy as the promise the live rollback
            # guard is measured against. The selection-window number is
            # upward-biased by having been the max of many draws.
            outcome.promoted_oos_expectancy = min(
                challenger.split.oos_expectancy, challenger.split.hold_expectancy
            )
            outcome.evidence = reason
            log.info("promoting %s candidate: %s", challenger.source, reason)
            return outcome

    best = challengers[0].split.oos_expectancy if challengers else 0.0
    outcome.evidence = (
        f"no challenger cleared the gauntlet (best selection {best:+.3f}R vs "
        f"incumbent {incumbent_eval.split.oos_expectancy:+.3f}R)"
    )
    return outcome


# ---------------------------------------------------------------- rollback
def should_rollback(
    live_r: list[float],
    promised_oos_expectancy: float,
    min_trades: int,
    margin_r: float,
) -> tuple[bool, str]:
    """Live guard: champion must live up to its own out-of-sample estimate.

    Reverts when, with enough live trades, realised expectancy is BOTH below
    (promise − margin) AND negative. The second condition stops us reverting a
    param set that's making money merely because the backtest oversold it.
    """
    if len(live_r) < min_trades:
        return False, f"{len(live_r)}/{min_trades} live trades since promotion"
    realised = _mean(live_r)
    threshold = promised_oos_expectancy - margin_r
    if realised < threshold and realised < 0:
        return True, (
            f"live {realised:+.3f}R over {len(live_r)} trades vs promised "
            f"{promised_oos_expectancy:+.3f}R (threshold {threshold:+.3f}R)"
        )
    return False, f"live {realised:+.3f}R holding up vs promise {promised_oos_expectancy:+.3f}R"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _key(params: dict[str, Any]) -> tuple:
    return tuple(sorted((k, round(float(v), 4)) for k, v in params.items()
                        if isinstance(v, (int, float))))
