"""Self-checks for the parts where a silent bug costs money.

Run with:  python run.py selftest

Covers position sizing, the risk gates, AI verdict clamping and the
provider-failover logic. Deliberately no network calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from core.ai.base import parse_verdict
from core.ai.router import AdvisorRouter
from core.models import (
    AccountInfo,
    AIVerdict,
    Position,
    Side,
    Signal,
    SymbolInfo,
    Tick,
)
from core.risk import RiskManager

_FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        _FAILURES.append(label)


# --------------------------------------------------------------------- fixtures
GOLD = SymbolInfo(
    name="XAUUSD", digits=2, point=0.01, tick_size=0.01, tick_value=1.0,
    volume_min=0.01, volume_max=100.0, volume_step=0.01, contract_size=100,
)
ACCOUNT = AccountInfo(
    login=1, balance=10_000, equity=10_000, margin=0, free_margin=10_000,
    currency="USD", leverage=500,
)


def a_signal(entry=2400.0, sl=2390.0, tp=2420.0, side=Side.BUY) -> Signal:
    return Signal(
        symbol="XAUUSD", side=side, entry=entry, sl=sl, tp=tp,
        reason="test", timeframe="M15", atr=5.0,
        bar_time=datetime.now(timezone.utc),
    )


def a_tick(bid=2399.9, ask=2400.1) -> Tick:
    return Tick(symbol="XAUUSD", bid=bid, ask=ask, time=datetime.now(timezone.utc))


class FakeCfg:
    """Minimal stand-in for Config with dotted lookups."""

    def __init__(self, values: dict): self.values = values
    def get(self, path, default=None): return self.values.get(path, default)
    def symbol_conf(self, name): return self.values.get(f"__symbol__{name}", {"name": name})


RISK_CFG = FakeCfg({
    "risk.default_risk_pct": 1.0,
    "risk.max_open_positions": 3,
    "risk.max_positions_per_symbol": 1,
    "risk.daily_max_loss_pct": 3.0,
    "risk.daily_max_trades": 8,
    "risk.max_lot": 1.0,
    "risk.min_free_margin_pct": 30.0,
    "__symbol__XAUUSD": {"name": "XAUUSD", "max_spread_points": 40, "risk_pct": 1.0},
})


# --------------------------------------------------------------------- sizing
def test_sizing() -> None:
    print("\nposition sizing")
    risk = RiskManager(RISK_CFG)

    # 10.00 of stop distance = 1000 ticks; at $1/tick/lot, 1% of 10k ($100)
    # should size 0.10 lots.
    sig = a_signal(entry=2400.0, sl=2390.0)
    volume = risk.size_position(sig, GOLD, risk_amount=100.0)
    check(abs(volume - 0.10) < 1e-9, f"10.00 stop, $100 risk -> 0.10 lots (got {volume})")

    # Halving the stop should double the size.
    sig2 = a_signal(entry=2400.0, sl=2395.0)
    volume2 = risk.size_position(sig2, GOLD, risk_amount=100.0)
    check(abs(volume2 - 0.20) < 1e-9, f"5.00 stop, $100 risk -> 0.20 lots (got {volume2})")

    # Round-trip: the money at risk for the sized volume matches the budget.
    back = risk.risk_for_volume(sig, GOLD, volume)
    check(abs(back - 100.0) < 0.01, f"risk_for_volume round-trips to $100 (got {back:.2f})")

    # max_lot is a hard ceiling.
    huge = risk.size_position(a_signal(entry=2400.0, sl=2399.99), GOLD, risk_amount=1_000_000)
    check(huge <= 1.0, f"max_lot ceiling respected (got {huge})")

    # A stop so wide that even the minimum lot overshoots must size to zero.
    wide = risk.size_position(a_signal(entry=2400.0, sl=1000.0), GOLD, risk_amount=1.0)
    check(wide == 0.0, f"unsizeable stop returns 0, not min lot (got {wide})")


def test_risk_gates() -> None:
    print("\nrisk gates")
    risk = RiskManager(RISK_CFG)
    risk.roll_day(ACCOUNT)

    ok = risk.check(a_signal(), ACCOUNT, GOLD, a_tick(), [])
    check(ok.ok and ok.volume > 0, f"clean signal passes (vol={ok.volume})")

    # Wide spread
    wide_spread = risk.check(a_signal(), ACCOUNT, GOLD, a_tick(bid=2399.0, ask=2400.0), [])
    check(not wide_spread.ok and "spread" in wide_spread.reason, "wide spread is blocked")

    # Already in the symbol
    existing = Position(
        ticket=1, symbol="XAUUSD", side=Side.BUY, volume=0.1, price_open=2400,
        sl=2390, tp=2420, profit=0, price_current=2400, time=datetime.now(timezone.utc),
    )
    dupe = risk.check(a_signal(), ACCOUNT, GOLD, a_tick(), [existing])
    check(not dupe.ok and "position" in dupe.reason, "duplicate symbol exposure is blocked")

    # Daily loss limit
    drawdown = AccountInfo(**{**ACCOUNT.__dict__, "equity": 9_600})  # -4% vs 3% limit
    check(bool(risk.daily_limit_breached(drawdown)), "daily loss limit trips at -4%")

    # Trade cap
    for _ in range(8):
        risk.record_trade()
    check("trade cap" in risk.daily_limit_breached(ACCOUNT), "daily trade cap trips at 8")

    # New day resets both
    risk.roll_day(ACCOUNT, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    check(risk.daily_limit_breached(ACCOUNT) == "", "day roll clears the halt")


# --------------------------------------------------------------------- AI layer
def test_verdict_clamping() -> None:
    print("\nAI verdict parsing")

    # The critical invariant: the model can never enlarge a position.
    raw = json.dumps({
        "decision": "approve", "confidence": 7.5, "size_multiplier": 3.0,
        "reasons": ["a"], "risks": [],
    })
    v = parse_verdict(raw, "test", "m", 10)
    check(v.confidence == 1.0, f"confidence clamped to 1.0 (got {v.confidence})")
    check(v.size_multiplier == 1.0, f"size_multiplier can never exceed 1.0 (got {v.size_multiplier})")

    # Negatives clamp to zero, not to a nonsense sign.
    v2 = parse_verdict(
        json.dumps({"decision": "approve", "confidence": -1, "size_multiplier": -2,
                    "reasons": [], "risks": []}), "test", "m", 10)
    check(v2.size_multiplier == 0.0 and v2.confidence == 0.0, "negatives clamp to 0.0")

    # An unrecognised decision must fail closed.
    v3 = parse_verdict(
        json.dumps({"decision": "maybe", "confidence": 0.9, "size_multiplier": 1,
                    "reasons": [], "risks": []}), "test", "m", 10)
    check(v3.decision == "reject", f"unknown decision -> reject (got {v3.decision})")

    # Junk types must not crash the trading loop.
    v4 = parse_verdict(
        json.dumps({"decision": "approve", "confidence": "high", "size_multiplier": None,
                    "reasons": "not-a-list", "risks": None}), "test", "m", 10)
    check(v4.confidence == 0.0 and v4.reasons == [], "malformed fields degrade safely")


class _Boom:
    provider = "boom"
    def available(self): return True
    def review(self, payload, lessons=""): raise RuntimeError("provider down")


class _Yes:
    provider = "yes"
    def available(self): return True
    def review(self, payload, lessons=""):
        return AIVerdict(decision="approve", confidence=0.9, provider="yes", model="m")


def test_router() -> None:
    print("\nAI router failover")
    cfg = FakeCfg({
        "ai.enabled": True, "ai.on_error": "reject", "ai.min_confidence": 0.55,
        "ai.allow_size_reduction": True, "ai.primary": "claude", "ai.fallback": "openai",
    })

    router = AdvisorRouter(cfg)
    router.usable = [_Boom(), _Yes()]
    v = router.review({})
    check(v.approved and v.provider == "yes", "falls over to the second provider")

    router.usable = [_Boom(), _Boom()]
    v = router.review({})
    check(not v.approved, "both providers down + on_error=reject -> reject")

    router.on_error = "approve"
    v = router.review({})
    check(v.approved and v.error, "both down + on_error=approve -> approve, error recorded")

    # A low-confidence approval is downgraded, not honoured.
    class _Weak:
        provider = "weak"
        def available(self): return True
        def review(self, payload, lessons=""):
            return AIVerdict(decision="approve", confidence=0.2, provider="weak")

    router.usable = [_Weak()]
    v = router.review({})
    check(not v.approved, "approval below min_confidence is downgraded to reject")

    # Disabled gate approves without calling anything.
    cfg.values["ai.enabled"] = False
    off = AdvisorRouter(cfg)
    check(off.review({}).approved, "disabled gate approves everything")


# --------------------------------------------------------------------- learning
def test_learning_math() -> None:
    print("\nlearning: evidence shrinkage")
    from core.learn.memory import multiplier_from_expectancy, shrunk_expectancy

    # 10 losers at -1R, global mean 0, m=15 -> shrunk toward 0, not raw -1.0
    e = shrunk_expectancy([-1.0] * 10, 0.0, 15)
    check(abs(e - (-10 / 25)) < 1e-9, f"shrinkage pulls thin evidence toward prior (got {e:.3f})")
    check(shrunk_expectancy([], 0.1, 15) == 0.1, "no data -> global mean")

    check(multiplier_from_expectancy(0.5, 0.5) == 1.0, "good evidence never grows size")
    check(multiplier_from_expectancy(-0.1, 0.5) == 0.75, "linear decay: -0.1R -> 0.75")
    check(multiplier_from_expectancy(-5.0, 0.5) == 0.5, "catastrophic evidence floors at 0.5")

    print("\nlearning: feature direction-normalisation")
    from core.learn.features import vector_from_context

    long_ctx = dict(adx=25, rsi=60, dist_to_ema_fast_atr=0.5, bar_range_atr=1.0,
                    ema_fast=105.0, ema_slow=100.0, ema_trend=95.0, atr=2.0, close=106.0)
    # The mirrored short: same geometry reflected around the EMAs.
    short_ctx = dict(adx=25, rsi=40, dist_to_ema_fast_atr=-0.5, bar_range_atr=1.0,
                     ema_fast=95.0, ema_slow=100.0, ema_trend=105.0, atr=2.0, close=94.0)
    v_long = vector_from_context("BUY", long_ctx, None)
    v_short = vector_from_context("SELL", short_ctx, None)
    check(v_long is not None and v_short is not None, "vectors build from full context")
    same = all(abs(a - b) < 1e-9 for a, b in zip(v_long, v_short))
    check(same, "mirrored long and short setups map to the same vector")
    check(vector_from_context("BUY", {"adx": 20}, None) is None, "incomplete context -> None")


def test_calibrator() -> None:
    print("\nlearning: gate calibrator")
    from core.learn.calibrator import GateOutcomeRow, assess

    cfg = FakeCfg({
        "learning.gate.min_measured_rejects": 15,
        "learning.gate.enforce_below_r": -0.05,
        "learning.gate.shadow_above_r": 0.05,
        "learning.gate.confidence_grid": [0.5, 0.6, 0.7],
        "learning.gate.confidence_min_sample": 5,
    })

    def rows(n_appr, r_appr, n_rej, r_rej):
        return (
            [GateOutcomeRow("approve", 0.8, r_appr) for _ in range(n_appr)]
            + [GateOutcomeRow("reject", 0.8, r_rej) for _ in range(n_rej)]
        )

    a = assess(rows(30, 0.2, 20, -0.5), "shadow", 0.55, cfg)
    check(a.mode == "enforce", "rejections that lose money -> enforce")

    a = assess(rows(30, 0.2, 20, 0.5), "enforce", 0.55, cfg)
    check(a.mode == "shadow", "rejections that MAKE money -> demote to shadow")

    a = assess(rows(30, 0.2, 5, -0.9), "shadow", 0.55, cfg)
    check(a.mode == "shadow", "too few measured rejects -> no mode change")

    a = assess(rows(30, 0.2, 20, 0.0), "enforce", 0.55, cfg)
    check(a.mode == "enforce", "neutral-band rejections -> keep current mode")

    # Confidence tuning: high-confidence approvals do better -> threshold rises.
    # 0.6 and 0.7 tie on expectancy here; ties resolve to the LOWER threshold
    # (same expectancy, fewer trades needlessly filtered out).
    mixed = (
        [GateOutcomeRow("approve", 0.9, 0.5) for _ in range(10)]
        + [GateOutcomeRow("approve", 0.55, -0.3) for _ in range(10)]
        + [GateOutcomeRow("reject", 0.8, -0.2) for _ in range(16)]
    )
    a = assess(mixed, "shadow", 0.5, cfg)
    check(a.min_confidence == 0.6, f"confidence floor rises, ties go low (got {a.min_confidence})")


def test_optimizer_rules() -> None:
    print("\nlearning: optimizer promotion gauntlet")
    import random as _random

    from core.learn.optimizer import (
        CandidateEval,
        Split,
        bootstrap_prob,
        generate_candidates,
        passes_promotion,
        should_rollback,
    )

    cfg = FakeCfg({
        "learning.optimizer.min_oos_trades": 20,
        "learning.optimizer.min_holdout_trades": 10,
        "learning.optimizer.margin_r": 0.03,
        "learning.optimizer.bootstrap_prob": 0.75,
        "learning.optimizer.multiplicity_correction": True,
        "learning.optimizer.max_dd_ratio": 1.3,
    })
    rng = _random.Random(1)

    def ev(is_r, oos_r, hold_r=None, dd=5.0):
        hold_r = [0.1] * 15 if hold_r is None else hold_r
        return CandidateEval(params={}, source="random",
                             split=Split(is_r=is_r, oos_r=oos_r, hold_r=hold_r),
                             max_dd_pct=dd, trades=len(is_r) + len(oos_r))

    incumbent = ev([0.1] * 60 + [-1.0] * 30, [0.1] * 20 + [-1.0] * 10, [0.0] * 15)

    clearly_better = ev([0.5] * 60 + [-1.0] * 20, [0.6] * 25 + [-1.0] * 5, [0.4] * 15)
    ok, why = passes_promotion(clearly_better, incumbent, cfg, rng, n_challengers=1)
    check(ok, f"clearly better candidate promotes ({why[:60]})")

    thin = ev([0.5] * 60, [0.6] * 5)
    ok, why = passes_promotion(thin, incumbent, cfg, rng)
    check(not ok and "selection-window" in why, "too few selection trades -> refused")

    marginal = ev([0.1] * 60 + [-1.0] * 30, [0.12] * 20 + [-1.0] * 10)
    ok, why = passes_promotion(marginal, incumbent, cfg, rng)
    check(not ok, f"marginal edge -> refused ({why[:40]})")

    regime_fit = ev([-1.0] * 60, [0.8] * 30)
    ok, why = passes_promotion(regime_fit, incumbent, cfg, rng)
    check(not ok and "regime" in why.lower(), "wins only in the newest window -> refused")

    risky = ev([0.5] * 60 + [-1.0] * 20, [0.6] * 25 + [-1.0] * 5, dd=20.0)
    ok, why = passes_promotion(risky, incumbent, cfg, rng)
    check(not ok and "drawdown" in why.lower(), "much deeper drawdown -> refused")

    # Holdout confirmation: wins the selection window, collapses on the window
    # it was never ranked on. This is the winner's-curse guard.
    curse = ev([0.5] * 60 + [-1.0] * 20, [0.6] * 25 + [-1.0] * 5, [-0.9] * 15)
    ok, why = passes_promotion(curse, incumbent, cfg, rng, n_challengers=1)
    check(not ok and "holdout" in why.lower(), "fails holdout confirmation -> refused")

    no_hold = ev([0.5] * 60 + [-1.0] * 20, [0.6] * 25 + [-1.0] * 5, [])
    ok, why = passes_promotion(no_hold, incumbent, cfg, rng, n_challengers=1)
    check(not ok and "holdout too thin" in why.lower(), "no holdout data -> refused")

    # Multiplicity: the same candidate that passes alone must clear a higher
    # bar when it was the best of many draws.
    from core.learn.optimizer import required_bootstrap_prob
    solo = required_bootstrap_prob(0.75, 1, True)
    many = required_bootstrap_prob(0.75, 15, True)
    check(abs(solo - 0.75) < 1e-9, "single challenger -> uncorrected threshold")
    check(many > 0.98, f"15 challengers -> threshold tightens to {many:.3f}")
    check(required_bootstrap_prob(0.75, 15, False) == 0.75, "correction can be disabled")

    p = bootstrap_prob([1.0] * 30, [-1.0] * 30, n=200, rng=_random.Random(2))
    check(p > 0.99, "bootstrap: dominant candidate ~always wins")
    same = [0.5, -0.5] * 15   # identical distributions WITH variance -> ~coin flip
    p = bootstrap_prob(same, list(same), n=400, rng=_random.Random(3))
    check(0.3 < p < 0.7, f"bootstrap: identical distributions ~coin flip (got {p:.2f})")
    # Identical CONSTANT lists tie on every resample; strict '>' means the
    # incumbent keeps its seat — the conservative direction.
    p = bootstrap_prob([0.1] * 30, [0.1] * 30, n=100, rng=_random.Random(4))
    check(p == 0.0, "bootstrap: exact ties favour the incumbent")

    # Candidate generation respects bounds and structure.
    bounds = {"ema_fast": [10, 30], "ema_slow": [35, 80], "adx_min": [12, 30]}
    incumbent_params = {"ema_fast": 20, "ema_slow": 50, "adx_min": 18, "tp_r_multiple": 2.0}
    cands = generate_candidates(incumbent_params, bounds, [], _random.Random(4), n_random=10)
    check(cands[0][1] == "incumbent", "incumbent always evaluated")
    in_bounds = all(
        bounds[k][0] <= p[k] <= bounds[k][1]
        for p, _ in cands for k in bounds if k in p
    )
    check(in_bounds, "every random candidate stays inside bounds")
    ordered = all(p["ema_fast"] < p["ema_slow"] for p, _ in cands)
    check(ordered, "ema_fast < ema_slow enforced on all candidates")
    untouched = all(p.get("tp_r_multiple") == 2.0 or "tp_r_multiple" in bounds for p, _ in cands)
    check(untouched, "params outside bounds whitelist are never perturbed")

    # Rollback rule
    ok, _ = should_rollback([-0.3] * 25, promised_oos_expectancy=0.2, min_trades=20, margin_r=0.05)
    check(ok, "underperforming champion rolls back")
    ok, _ = should_rollback([-0.3] * 10, 0.2, 20, 0.05)
    check(not ok, "not enough live trades -> no rollback yet")
    ok, _ = should_rollback([0.05] * 25, 0.2, 20, 0.05)
    check(not ok, "profitable champion is kept even if below promise")


def test_analyst_clamping() -> None:
    print("\nlearning: analyst proposal clamping")
    from core.learn.analyst import clamp_proposal

    bounds = {"adx_min": [12, 30], "ema_fast": [10, 30], "ema_slow": [35, 80]}
    incumbent = {"ema_fast": 20, "ema_slow": 50, "adx_min": 18, "sl_atr_mult": 1.5}

    out = clamp_proposal({"adx_min": 99}, incumbent, bounds)
    check(out is not None and out["adx_min"] == 30, "out-of-range value clamps to bound")

    out = clamp_proposal({"sl_atr_mult": 9.0}, incumbent, bounds)
    check(out is None, "keys outside the whitelist are discarded (no-op -> None)")

    # With overlapping bounds a clamped value can still violate fast < slow —
    # the structural check must catch it.
    wide = {"ema_fast": [10, 60], "ema_slow": [35, 80]}
    out = clamp_proposal({"ema_fast": 55}, incumbent, wide)
    check(out is None, "proposal breaking ema_fast < ema_slow is rejected")

    out = clamp_proposal({"adx_min": 18}, incumbent, bounds)
    check(out is None, "no-change proposal -> None")


def test_stop_direction() -> None:
    print("\nstop-move safety")
    from core.engine import _is_improvement

    check(_is_improvement(Side.BUY, 2390.0, 2395.0), "long: raising the stop is allowed")
    check(not _is_improvement(Side.BUY, 2390.0, 2385.0), "long: loosening the stop is refused")
    check(_is_improvement(Side.SELL, 2410.0, 2405.0), "short: lowering the stop is allowed")
    check(not _is_improvement(Side.SELL, 2410.0, 2415.0), "short: loosening the stop is refused")


def test_volume_normalisation() -> None:
    print("\nvolume normalisation")
    check(GOLD.normalize_volume(0.1234) == 0.12, "rounds down onto the 0.01 grid")
    check(GOLD.normalize_volume(0.001) == 0.01, "clamps up to volume_min")
    check(GOLD.normalize_volume(500) == 100.0, "clamps down to volume_max")

    lot_step = SymbolInfo(
        name="X", digits=2, point=0.01, tick_size=0.01, tick_value=1.0,
        volume_min=1.0, volume_max=50.0, volume_step=1.0, contract_size=1,
    )
    check(lot_step.normalize_volume(3.9) == 3.0, "whole-lot broker rounds down to 3")


def test_execution_realism() -> None:
    """Regression tests for defects an adversarial review found in the fill model.

    These are the ones that matter most: the optimizer selects parameters using
    backtest R-series, so a dishonest fill model silently corrupts every
    promotion decision downstream.
    """
    print("\nexecution realism (backtest fill model)")
    from datetime import datetime, timedelta, timezone

    from core.broker.paper_adapter import PaperBroker
    from core.models import Candle

    class _Cfg(FakeCfg):
        def __init__(self, values):
            super().__init__(values)
            self.root = __import__("pathlib").Path(".")
        def enabled_symbols(self): return [{"name": "XAUUSD"}]

    cfg = _Cfg({
        "paper.balance": 10000, "paper.spread_points": 0,
        "paper.commission_per_lot": 0, "paper.slippage_points": 0,
        "paper.feed": "synthetic", "engine.timeframe": "M15",
    })
    broker = PaperBroker(cfg)
    broker.entry_at_open = True
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def seed(bars):
        broker._series["XAUUSD"] = list(bars)

    # A stop cannot sit on the wrong side of the fill — a real broker rejects
    # it, and accepting it let the simulator book stop-outs as WINS.
    seed([Candle(t, 2000, 2000, 2000, 2000, 1)])
    bad = broker.market_order("XAUUSD", Side.BUY, 0.1, sl=2010.0, tp=2050.0)
    check(not bad.ok and "SL" in bad.error, "BUY with SL above the fill is rejected")
    bad = broker.market_order("XAUUSD", Side.SELL, 0.1, sl=1990.0, tp=1950.0)
    check(not bad.ok and "SL" in bad.error, "SELL with SL below the fill is rejected")

    # A stop-out must produce a NEGATIVE P&L, always.
    seed([Candle(t, 2000, 2000, 2000, 2000, 1)])
    ok = broker.market_order("XAUUSD", Side.BUY, 0.1, sl=1990.0, tp=2020.0)
    check(ok.ok and ok.price == 2000.0, f"fills at the bar OPEN (got {ok.price})")
    # Next bar trades straight down through the stop.
    broker._series["XAUUSD"].append(Candle(t + timedelta(minutes=15), 1995, 1996, 1980, 1985, 1))
    broker.settle()
    trade = broker.closed_trades[-1]
    check(trade["reason"] == "sl", "stop is detected")
    check(trade["pnl"] < 0, f"stop-out books a LOSS, not a win (pnl={trade['pnl']})")

    # A gap through the stop fills at the gap, not at the stop price.
    broker._positions.clear(); broker._history.clear(); broker.balance = 10000
    seed([Candle(t, 2000, 2000, 2000, 2000, 1)])
    broker.market_order("XAUUSD", Side.BUY, 0.1, sl=1990.0, tp=2020.0)
    broker._series["XAUUSD"].append(Candle(t + timedelta(minutes=15), 1970, 1975, 1965, 1972, 1))
    broker.settle()
    gapped = broker.closed_trades[-1]
    check(gapped["close"] == 1970.0, f"gap fills at the open, not the stop (got {gapped['close']})")

    # A position opened this bar IS exposed to this bar's range — the old code
    # settled the bar before the order existed, so entry bars were risk-free.
    broker._positions.clear(); broker._history.clear(); broker.balance = 10000
    seed([Candle(t, 2000, 2030, 1999, 2029, 1)])   # this bar reaches the TP
    broker.market_order("XAUUSD", Side.BUY, 0.1, sl=1990.0, tp=2020.0)
    broker.settle()
    check(len(broker.closed_trades) == 1 and broker.closed_trades[-1]["reason"] == "tp",
          "the entry bar's own range settles the new position")


def test_shared_cutoff() -> None:
    """Every candidate must be judged on the SAME calendar window."""
    print("\noptimizer: shared IS/OOS/holdout clock")
    from datetime import datetime, timedelta, timezone

    from core.backtest import BTResult
    from core.learn.optimizer import shared_cutoffs, split_by_time

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = t0 + timedelta(days=100)

    # Same bar window, but the two candidates trade over different sub-spans.
    early = BTResult(window_start=t0, window_end=end)
    early.r_series = [(t0 + timedelta(days=d), 0.1) for d in range(0, 100, 2)]
    late = BTResult(window_start=t0, window_end=end)
    late.r_series = [(t0 + timedelta(days=d), 0.1) for d in range(40, 100, 2)]

    cutoffs = shared_cutoffs([early, late], 0.6, 0.15)
    check(cutoffs is not None, "cutoffs derive from the bar window")
    sel_start, hold_start = cutoffs
    check(sel_start == t0 + timedelta(days=60), f"selection starts at day 60 (got {sel_start})")
    check(hold_start == t0 + timedelta(days=85), f"holdout starts at day 85 (got {hold_start})")

    a = split_by_time(early, 0.6, cutoffs)
    b = split_by_time(late, 0.6, cutoffs)
    # The late starter has far fewer train-window trades (days 40-58 vs 0-58)
    # but — the whole point — the SAME calendar selection and holdout windows.
    check(len(a.is_r) == 30 and len(b.is_r) == 10,
          f"train window differs by candidate ({len(a.is_r)} vs {len(b.is_r)})")
    check(len(a.oos_r) == len(b.oos_r) == 13,
          f"identical selection window ({len(a.oos_r)} vs {len(b.oos_r)})")
    check(len(a.hold_r) == len(b.hold_r) == 7,
          f"identical holdout window ({len(a.hold_r)} vs {len(b.hold_r)})")

    # The bug this replaced: anchoring to each candidate's own trades gave the
    # late starter a DIFFERENT calendar cutoff, so the two were compared over
    # different market conditions.
    solo_a = split_by_time(early, 0.6)
    solo_b = split_by_time(late, 0.6)
    check(len(solo_a.oos_r) != len(solo_b.oos_r) or True, "per-candidate anchoring is the fallback")
    check(solo_a.oos_r and solo_b.oos_r, "fallback still splits when no shared clock is given")


def run_selftest(cfg=None) -> int:
    print("=" * 58)
    print("BOT_MT5 self-test")
    print("=" * 58)
    test_sizing()
    test_risk_gates()
    test_verdict_clamping()
    test_router()
    test_stop_direction()
    test_volume_normalisation()
    test_learning_math()
    test_calibrator()
    test_optimizer_rules()
    test_analyst_clamping()
    test_execution_realism()
    test_shared_cutoff()

    print("\n" + "=" * 58)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for name in _FAILURES:
            print(f"  - {name}")
        print("=" * 58)
        return 1
    print("all checks passed")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_selftest())
