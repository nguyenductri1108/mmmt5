# How to actually get value from Claude / Codex in a trading bot

You have API tokens and a bot. The question is where to point the model so it
adds edge instead of cost and latency. This is the opinionated answer, plus the
protocol for proving whether it worked.

---

## 1. The framing that matters

An LLM is **bad** at the thing everyone asks it to do (predict the next price)
and **good** at the thing most retail bots never do (apply consistent judgement
to structured context, and analyse your own trading history without ego).

| What people ask the model | Why it fails |
|---|---|
| "Will XAUUSD go up?" | No edge — price is the most efficiently-priced data on earth, and the model's training data ends before your candle. |
| "Pick the entry and exit" | 1–15 s latency, non-deterministic output, and **you can never backtest it**. You lose the ability to know if your system works. |
| "Read this chart image and trade it" | Expensive, slow, and the model's chart reading is worse than three lines of pandas. |
| "Size this position" | One hallucinated decimal is an account. Never let a probabilistic system touch a number that scales risk. |

The rule that follows: **deterministic code owns every number; the model owns
judgement calls that never touch a number directly.**

That rule is enforced in the code, not just in the prompt — `parse_verdict()` in
[core/ai/base.py](core/ai/base.py) clamps `size_multiplier` to a hard maximum of
`1.0`, so even a model returning `"size_multiplier": 50` cannot enlarge a trade.
Verified by [scripts/selftest.py](scripts/selftest.py).

---

## 2. The five tiers, ordered by latency budget

Deploy them in this order. Each one is independently useful — you do not need
the later tiers for the earlier ones to pay off.

### Tier 0 — Build the bot (no latency, highest ROI today)

The highest-value use of your tokens is not in the trading loop at all. It is
writing the adapters, the indicators, the backtester, and the analysis scripts.
This repo is an example. Keep doing this — it is where AI has an unambiguous,
already-proven edge, and it costs you nothing at runtime.

### Tier 1 — Daily regime briefing (1 call/day, ~$0.01)

Once per day before the session, one call that looks at higher-timeframe data
and the economic calendar, and returns a **config patch**, not a trade:

```json
{
  "regime": "trending",
  "risk_multiplier": 1.0,
  "symbols_enabled": ["XAUUSD", "EURUSD"],
  "adx_min_override": 18,
  "notes": "FOMC 18:00 UTC — flatten by 17:30"
}
```

Why this is the best value-per-token in the whole system:
- One call a day, so cost and latency are irrelevant.
- The output is a diff against a config file: fully auditable, trivially
  reversible, and you can replay any day's config to reproduce behaviour.
- It targets the thing that actually kills trend systems — trading a trend
  strategy through a chop regime, or holding through a known event.

*Not yet implemented here.* It is the highest-value next addition: a
`core/ai/regime.py` writing `data/regime.json`, which `Engine.tick()` reads on
the day roll.

### Tier 2 — Per-signal veto gate (seconds, implemented)

This is what's built. The deterministic strategy produces a candidate with a
fixed entry, stop, target and size; the model gets one narrow decision:
**let it through, or don't** — plus an optional size reduction.

The design decisions that make this work rather than backfire:

| Decision | Why |
|---|---|
| **Approve by default**, reject only with a cited number | Models are trained to be cautious. Ask "is this a good trade?" and you get a hedged rejection on 60%+ of signals, which destroys the strategy's edge. The prompt in [core/ai/base.py](core/ai/base.py) enumerates what *is not* a valid rejection reason. |
| The model **cannot** move entry / SL / TP | Nothing it returns is used as a price. Its entire output surface is one enum, one float, and some strings. |
| `size_multiplier` clamped to `[0.0, 1.0]` | It can shrink, never grow. Bounded downside on a hallucination. |
| Schema-constrained JSON output | The engine never parses prose, so a chatty response cannot break the loop. |
| **Fail closed** (`ai.on_error: reject`) | If the API is down, take no trades. A quiet hour costs nothing; an unvetted hour can cost a lot. Flip to `approve` only once you trust the strategy standalone. |
| Every **rejected** signal is journalled | Without this you can never measure the gate — see §4. |

### Tier 3 — Weekly post-mortem analyst (offline, high effort) — *now implemented*

> **Update:** this tier is now built and fully automatic — see
> [LEARNING.md](LEARNING.md). The analyst runs weekly, maintains evidence-cited
> "lessons" in the gate prompt, and its parameter hypotheses are validated by a
> walk-forward optimizer before promotion, with auto-rollback. The description
> below is the original design rationale.

Once a week, feed the model the journal — executed trades, rejected signals,
and outcomes — and ask for **hypotheses, not changes**:

> "Here are 60 trades and 200 rejected signals with full indicator context.
> Find patterns that separate winners from losers. For each pattern, state the
> specific parameter change that would exploit it and the number of trades
> supporting it. Do not suggest anything backed by fewer than 20 trades."

Then the loop that keeps you honest:

```
hypothesis → parameter change → run `python run.py backtest`
          → survives? → paper for 2 weeks → then a human approves it live
```

**Never let the model write parameters straight to `config.yaml`.** This is the
tier that quietly compounds: it is a tireless, unsentimental analyst that will
tell you your favourite setup loses money.

Use a high effort setting here (`effort: xhigh` or `max`) — this call happens
once a week, so depth is free.

### Tier 4 — Event/news feature extraction (optional)

If you trade news-sensitive instruments, have a model convert headlines into a
**numeric feature the strategy consumes** — not a trade decision:

```json
{"symbol": "XAUUSD", "event_risk_next_2h": 0.8, "direction_bias": 0}
```

The strategy then widens stops or stands down when `event_risk` is high. Note
the discipline: the model produces a *feature*, deterministic code decides what
to do with it. Same principle as everywhere else.

---

## 3. Which model, where

| Job | Model | Setting | Why |
|---|---|---|---|
| Per-signal gate | `claude-opus-5` | `effort: low` | Best structured-JSON reliability. Low effort keeps it ~2–4 s. |
| Per-signal gate (cost-optimised) | `claude-haiku-4-5` | — | ~5× cheaper. The gate is a narrow classification; Haiku handles it. Swap in `ai.claude.model` if volume grows. |
| Daily regime briefing | `claude-opus-5` | `effort: high` | Once a day — buy the depth. |
| Weekly analyst | `claude-opus-5` | `effort: xhigh` | Hardest reasoning task in the system, run 4×/month. |
| Fallback provider | your OpenAI/Codex model | — | Configured as `ai.fallback`. |

**On the two-provider setup you chose:** the fallback exists for *availability*,
not for a second opinion. It fires only when the primary errors or times out.

If you want genuine model diversity, that's a different design: call both and
require agreement. It roughly halves your trade count, so only do it if §4 shows
your gate is rejecting too little rather than too much.

**A note on thinking:** Claude Opus 5 has thinking on by default. This code
leaves it on and uses `effort: low` rather than `thinking: disabled` — on this
model, disabling thinking can cause tool calls to leak into plain text and
`<thinking>` tags to appear in output. Low effort is both cheaper and safer.

### Cost, concretely

Per-signal gate on M15 with 2 symbols ≈ 5–20 calls/day. With prompt caching on
the stable system block (already wired up in
[core/ai/claude_advisor.py](core/ai/claude_advisor.py)), each call is ~1.5k input
(~1k of it cached at 10% price) and ~300 output tokens.

**≈ $0.30/day on Opus 5, so roughly $5–10/month.** On Haiku, about $2/month.
This is not the thing to optimise — one avoided bad trade pays for a year of it.

---

## 4. Measuring whether the gate is worth it

**This is the part almost everyone skips, and it is the only part that matters.**

A gate that rejects signals *feels* protective. But if it rejects your winners,
it is quietly destroying your edge and you will never notice, because a trade
you didn't take has no P&L to look at.

### The protocol

> **Update:** with `learning.enabled: true` (the default) this protocol now
> runs itself — the calibrator starts in shadow, flips to enforce only once
> rejections are proven to lose money, and keeps a 20% audit slice so the
> measurement never stops. See [LEARNING.md](LEARNING.md). The manual protocol
> below still applies if you turn the learning system off.

**Weeks 1–4: shadow mode.**

```yaml
ai:
  enabled: true
  shadow_mode: true    # ask the AI, log the verdict, trade anyway
```

Every signal is traded, and the gate's verdict is recorded alongside. After a
few weeks you have the counterfactual you need:

```bash
python run.py report
```

```
AI gate value (90d, shadow mode):
  gate approved             48 trades  win  44.0%  net    1180.00  avg   24.58
  gate would have blocked   17 trades  win  23.5%  net    -540.00  avg  -31.76
```

That is a gate earning its keep — the trades it wanted to block lost money.
If the "would have blocked" bucket is **profitable**, the gate is costing you
money and `run.py report` will say so explicitly.

**Weeks 5+: turn shadow mode off** — but only if the numbers justify it.

### The four numbers to watch

| Metric | Healthy | If it's off |
|---|---|---|
| **Blocked-bucket P&L** | Negative | Positive → the gate is hurting. Loosen the prompt or drop the gate. |
| **Rejection rate** | 10–25% | >40% → the model is second-guessing your strategy. Strengthen the approve-by-default framing. <5% → it's rubber-stamping; not harmful, but you're paying for nothing. |
| **Verdict stability** | Same payload → same verdict | Unstable → your prompt is too vague. Add specifics until it's reproducible. |
| **Latency** | < 5 s | Slower than a bar close is a real problem on M1/M5. Drop to Haiku or lower effort. |

`python run.py report` prints the first two directly from the journal.

### The honesty check

Note what the backtester does **not** do: `python run.py backtest` runs with
`ai.enabled = False`. That's deliberate. It measures the deterministic edge in
isolation. If your strategy loses money without the AI, no gate will save it —
a filter can only remove trades, and removing trades from a negative-expectancy
system just loses money more slowly.

**Fix the strategy first. The AI layer is a multiplier on an existing edge, not
a substitute for one.**

---

## 5. Suggested rollout

| Week | Do this | Gate you must clear to continue |
|---|---|---|
| 1 | `run.py backtest` on **real CSV history**, not the synthetic feed. Tune params. | Profit factor > 1.2 over 200+ trades |
| 2 | Demo account, `mode: live`, `ai.enabled: false`. Watch Telegram. | Live fills match backtest expectations |
| 3–6 | Demo + `ai.shadow_mode: true` | Blocked bucket is losing money |
| 7 | Shadow off, gate live, still demo | Rejection rate 10–25% |
| 8+ | Live, smallest size your broker allows, `risk_pct: 0.25` | 4 consecutive profitable weeks |
| Then | Add Tier 1 regime briefing, then Tier 3 weekly analyst | — |

Do not compress this. The expensive failures all come from skipping straight to
week 8.

---

## 6. Things deliberately not built, and why

| Not built | Why |
|---|---|
| LLM picks entries | Unbacktestable. You would lose the ability to know if the system works. |
| LLM sets stop/target | One bad number is an account. |
| Chart-image analysis | Slower and worse than pandas at the same job. |
| Sentiment scraping into trades | Noisy, unbackestable, and the alpha decayed years ago. |
| Auto-applying AI parameter suggestions | The model optimises what you measure; with live control it will overfit your recent history. Human in the loop, always. |

---

## 7. Where to start tomorrow

1. Get real CSV history into `data/` and re-run the backtest. Everything else is
   noise until the deterministic edge is real.
2. Run 2 weeks of demo with `ai.enabled: false` to prove the plumbing.
3. Then turn on `shadow_mode` and let the data tell you whether the gate helps.

The bot is built to make step 3 a one-line config change. Use it.
