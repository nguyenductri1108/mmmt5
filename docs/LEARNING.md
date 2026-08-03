# The self-improvement loop

You asked for a bot that *actually* gets better over time, fully automatically.
This document explains what was built, why each piece is shaped the way it is,
and — because honesty is the only defensible policy in trading software — what
"automatic improvement" can and cannot mean.

## The one rule that never bends

Everything the system learns can do exactly three things:

1. **reject** a trade
2. **shrink** a position (never below the floor, never above 1.0×)
3. **move strategy parameters inside the whitelisted bounds** in
   `learning.optimizer.bounds`

It can never raise a risk cap, widen a stop, grow a position, touch
`risk.*` or `session.*`, or trade an instrument you didn't enable. Autonomy is
bounded to the *entry logic*; the safety rails stay human-owned.

---

## The five mechanisms

```
              per signal (milliseconds, deterministic)
┌──────────────────────────────────────────────────────────────┐
│ 1. EPISODIC MEMORY  core/learn/memory.py                     │
│    Every closed trade is an "episode": indicator context ->  │
│    feature vector + outcome in R. Each new signal retrieves  │
│    its k=25 nearest past setups. The AI gate sees their      │
│    outcomes; the sizer applies a shrunk-expectancy multiplier│
│    in [0.5, 1.0]. More history -> sharper evidence. This is  │
│    the piece that literally grows with every trade.          │
└──────────────────────────────────────────────────────────────┘
              weekly, in a worker thread (automatic)
┌──────────────────────────────────────────────────────────────┐
│ 2. GATE CALIBRATOR  core/learn/calibrator.py                 │
│    Measures the AI gate against reality. Starts in SHADOW    │
│    (verdicts recorded, all trades taken). Once >= 15         │
│    measured rejections average worse than -0.05R, it flips   │
│    to ENFORCE. If rejections start averaging better than     │
│    +0.05R, it demotes itself back to shadow — a gate that    │
│    blocks winners fires itself. While enforcing, a random    │
│    20% of rejections still executes ("audit trades") so the  │
│    counterfactual never goes dark. min_confidence is also    │
│    re-chosen weekly from measured outcomes.                  │
├──────────────────────────────────────────────────────────────┤
│ 3. LESSONS  core/learn/lessons.py + analyst.py               │
│    The weekly analyst (Claude, effort xhigh) reads the       │
│    journal and maintains <= 8 short, evidence-cited rules    │
│    that are injected into the gate's prompt ("longs with     │
│    ADX < 20 averaged -0.4R over 14 trades"). Lessons the     │
│    next week's data no longer supports EXPIRE. This is how   │
│    the gate's judgement — not just its data — improves.      │
├──────────────────────────────────────────────────────────────┤
│ 4. PARAMETER EVOLUTION  core/learn/optimizer.py              │
│    Champion/challenger. Candidates = current params + ~12    │
│    random perturbations (inside bounds) + up to 3 analyst    │
│    hypotheses. Each is backtested on REAL cached history,    │
│    trades split by time into in-sample (older 70%) and       │
│    out-of-sample (newest 30%). Promotion requires ALL of:    │
│      * >= 20 OOS trades                                      │
│      * OOS expectancy beats incumbent by >= 0.03R            │
│      * holds up in-sample too (anti regime-fit)              │
│      * drawdown not > 1.3x incumbent's                       │
│      * bootstrap P(candidate > incumbent) >= 0.75            │
│    Most weeks nothing passes. That is correct behavior.      │
├──────────────────────────────────────────────────────────────┤
│ 5. LIVE ROLLBACK GUARD  optimizer.should_rollback()          │
│    A promoted champion must live up to its own out-of-sample │
│    estimate. After >= 20 live trades, if realised expectancy │
│    is below (promise - 0.05R) AND negative, the previous     │
│    params are restored automatically. /revert does the same  │
│    thing manually from Telegram.                             │
└──────────────────────────────────────────────────────────────┘
```

Supporting plumbing:

* **History caching** — in live mode the bot dumps broker candles into
  `data/<SYMBOL>_<TF>.csv` daily, so the optimizer always has real data to
  walk forward on. No manual exports.
* **The journal** (`data/journal.db`) is the substrate for all of it. Every
  signal — taken, blocked, audited — with full context and verdicts.

## Why the guards are the point

A "fully automatic" tuner without statistical guards is a machine for
overfitting: it will chase last month's noise, look brilliant in backtests,
and bleed live. Each guard answers one honest question:

| Guard | Question it answers |
|---|---|
| Out-of-sample split by *time* | "Does this work on data it wasn't picked on?" |
| In-sample floor | "Or does it only work in the newest regime?" |
| Bootstrap resampling | "Is the difference bigger than luck?" |
| Minimum trade counts everywhere | "Is there enough data to say anything at all?" |
| Audit trades | "Is the gate still right, now that it's blocking things?" |
| Shrinkage toward the global mean | "Are 5 unlucky neighbours a pattern or noise?" |
| Live rollback | "Did the promotion survive contact with reality?" |
| Synthetic-feed refusal | "Are we optimizing on real markets or on a random walk?" |

The system is *designed to mostly do nothing*. A quiet week — no promotion, no
mode flip, no new lessons — means the evidence didn't clear the bar. The
failure mode of eager learners is that they always find something.

## What happens on a schedule

| When | What | Where it runs |
|---|---|---|
| every signal | evidence lookup + size prior + lessons in prompt | main thread, milliseconds |
| daily 22:00 UTC | cache broker history to CSV (live mode) | main thread (MT5 isn't thread-safe) |
| weekly Sun 12:00 UTC | calibrate gate → analyst → optimize → promote/rollback | worker thread; results applied on the main thread |

After every weekly run you get a Telegram digest: what was measured, what
changed, what didn't and why. `/learning` shows the current state any time;
`/revert` undoes the last promotion.

## Cold start — what to expect

* **Weeks 1–2**: gate in shadow, memory too thin to act (needs 10 closed trades
  per symbol), optimizer skipped until CSV history exists and the incumbent has
  enough trades in the window. The bot trades on config.yaml exactly as tuned.
* **Weeks 3–6**: evidence multiplier starts engaging; calibrator has enough
  measured rejections to pick a gate mode; first lessons appear.
* **Month 2+**: enough journal for the optimizer to have statistical power.
  Expect a promotion every few weeks *at most* — and expect rollbacks
  occasionally. A rollback is the system working, not failing.

Speed it up: the more real CSV history in `data/`, the sooner the optimizer
has power. Seed it with years of exported MT5 data on day one.

## What this still is not

Model weights never change — Claude is identical on day 400 and day 1. What
improves is everything *around* the model: the evidence it sees, the lessons
in its prompt, the thresholds it's held to, and the strategy parameters. All
of it compounds from **your own journal**, which is why `data/` is the most
valuable file tree in this project. Back it up.

And the deterministic edge still comes first: if the base strategy has no edge
on real history, the learning loop will correctly spend every week refusing to
promote anything — it can sharpen a knife, not conjure one.
