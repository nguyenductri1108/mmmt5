# BOT_MT5

MetaTrader 5 trading bot with Telegram alerts, remote control, an AI signal
gate (Claude primary, OpenAI/Codex fallback), and a fully automatic
self-improvement loop.

```
strategy → risk engine → episodic memory → AI gate → broker
    │           │              │              │         │
    └───────────┴──────────────┴──────────────┴─────────┴──→ Telegram + journal
                                                                    │
              weekly, automatic:  calibrate gate ◄──────────────────┤
                                  distill lessons ◄─────────────────┤
                                  evolve params (walk-forward) ◄────┘
```

Every stage can block a trade. Only the risk engine sets numbers — everything
learned can reject, shrink, or retune within whitelisted bounds; it can never
enlarge a position, widen a stop, or raise a risk cap. The learning loop is
documented in [docs/LEARNING.md](docs/LEARNING.md).

---

## ⚠️ Read this first

**The `MetaTrader5` Python package is Windows-only.** It does not install on
macOS or Linux. That's why the bot has two broker adapters behind one interface:

| Mode | Adapter | Runs on | Use for |
|---|---|---|---|
| `paper` | simulated feed + fills | macOS, Linux, Windows | development, backtests, testing your Telegram wiring |
| `live` | real MetaTrader 5 | **Windows only** | demo and real accounts |

Develop on the Mac in `paper`, deploy on the Windows PC in `live`. Same strategy
code, one flag.

**This is auto-execute.** Signals are placed without asking you first. Telegram
gives you `/pause`, `/flat` and `/kill` — set them up before going live.

---

## Quick start (macOS — paper mode, works right now)

```bash
cd ~/Documents/BOT_MT5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then fill in your Telegram token

python run.py selftest        # 27 checks on the risk + AI safety logic
python run.py backtest        # replay the strategy, no AI, no Telegram
python run.py check           # pre-flight: config, broker, AI, Telegram
python run.py --mode paper    # run the full bot against the simulated feed
```

## Windows setup (live mode)

1. Install MT5, log into your **demo** account, and enable the **Algo Trading**
   toolbar button. Without it, every order is rejected.
2. Install Python 3.11+ (tick *Add to PATH*), then:
   ```powershell
   cd C:\BOT_MT5
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env      # fill in MT5_* and TELEGRAM_*
   python run.py check --mode live
   ```
3. `python run.py check --mode live` must be clean before you trade. It prints
   the exact spread, digits, lot step and tick value for every symbol — if a
   symbol name is wrong it fails here rather than at 3am.
4. `python run.py --mode live`

**Symbol names**: most brokers add a suffix — `XAUUSD.m`, `EURUSD_raw`,
`XAUUSD.pro`. Copy the exact name from MT5's Market Watch into `config.yaml`.

**Keep it running**: Windows sleep kills the bot mid-trade. Set power to *never
sleep*, or move to a VPS. Task Scheduler with "run whether user is logged on or
not" plus a restart-on-failure action is the usual setup.

---

## Telegram

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token
   into `.env` as `TELEGRAM_BOT_TOKEN`.
2. Send your new bot any message, then run `python run.py chatid` and paste the
   number into `.env` as `TELEGRAM_CHAT_ID`.
3. `python run.py check` sends a test message to confirm.

### What you get sent

| Event | Message |
|---|---|
| Signal approved | direction, entry, SL, TP, R:R, lots, money at risk, the setup rationale, and the AI's verdict + reasons |
| Filled | ticket, fill price, slippage vs the signal price |
| Signal skipped | which stage blocked it (risk / AI / sizing) and why |
| Stop moved | breakeven or trailing adjustment |
| Closed | open → close price, P&L, and whether it was SL / TP / manual |
| Halted | daily loss or trade cap hit |

### Commands

| Command | Effect |
|---|---|
| `/status` | equity, day P&L, open positions with live R-multiples |
| `/positions` | open positions only |
| `/pause` | stop opening new trades; **open positions are still managed** |
| `/resume` | resume |
| `/close <ticket>` or `/close XAUUSD` | close one position, or all on a symbol |
| `/flat` | close everything now |
| `/kill` | flatten everything and stop the bot |
| `/learning` | self-improvement status: gate mode, lessons, active params |
| `/revert` | undo the last auto-promoted parameter set |

Only your configured chat id is accepted; anything else is logged and ignored.

---

## The safety layers

Ordered from outermost in. Each is independent — a bug in one doesn't disable
the others.

1. **Session filter** — only trades inside configured UTC windows and weekdays.
   Flattens everything before the weekend gap (`session.friday_flat_at`).
2. **Risk engine** ([core/risk.py](core/risk.py)) — daily loss cap, daily trade
   cap, max open positions, max per symbol, free-margin floor, spread ceiling,
   and a hard `max_lot`. Position size is derived from the stop distance so a
   stop-out costs exactly `risk_pct` of equity.
3. **AI gate** — can reject or shrink. Fails closed by default: if the API is
   down, no trades.
4. **Stops only ever move to reduce risk** — the breakeven/trailing logic
   refuses to widen a stop. Unit-tested.
5. **Kill switch** — `/kill` from your phone.

The `magic` number in `config.yaml` tags this bot's orders. It will not touch
manual trades or another EA's positions.

---

## Commands

| Command | What it does |
|---|---|
| `python run.py` | run the bot (mode from `config.yaml`) |
| `python run.py --mode paper` / `--mode live` | override the mode |
| `python run.py check` | pre-flight: config, broker, symbols, AI, Telegram |
| `python run.py selftest` | 27 offline checks on sizing, risk gates, AI clamping |
| `python run.py backtest --bars 20000` | replay the strategy, no AI, no Telegram |
| `python run.py report` | journal summary + AI gate effectiveness |
| `python run.py chatid` | find your Telegram chat id |

---

## Backtesting on real data

The default paper feed is a **synthetic random walk**. It proves the plumbing
works; it tells you nothing about edge — no strategy has an edge on noise.

For real numbers, export history from MT5 and point the feed at it:

```yaml
paper:
  feed: csv
  csv_dir: data
```

Files go in `data/<SYMBOL>_<TIMEFRAME>.csv` (e.g. `data/XAUUSD_M15.csv`) with
columns `time,open,high,low,close,volume`. Minimum 250 bars; a few years is
better.

---

## Layout

```
run.py                     entry point / CLI
config.yaml                behaviour  (no secrets)
.env                       secrets    (never commit)
core/
  engine.py                the trading loop
  risk.py                  sizing + all hard limits
  backtest.py              programmatic replay (CLI + optimizer share it)
  journal.py               SQLite: every signal, taken or not
  models.py                broker-agnostic types
  indicators.py            EMA / ATR / ADX / RSI
  broker/
    base.py                the interface
    mt5_adapter.py         live (Windows)
    paper_adapter.py       simulated (everywhere)
  strategy/
    ema_atr.py             trend + pullback, ATR stop, R-multiple target
  ai/
    base.py                shared prompt, JSON schema, clamping
    claude_advisor.py      Anthropic
    openai_advisor.py      OpenAI / Codex
    router.py              primary → fallback, failure policy
  learn/                   ← the self-improvement loop (docs/LEARNING.md)
    memory.py              episodic memory: k-NN over your own closed trades
    calibrator.py          auto shadow/enforce gate switching from outcomes
    lessons.py             evidence-cited rules injected into the gate prompt
    analyst.py             weekly LLM journal review
    optimizer.py           walk-forward champion/challenger params + rollback
    scheduler.py           orchestration, threading, state files
  notify/telegram.py       alerts out, commands in
scripts/
  backtest.py              CLI wrapper for core/backtest.py
  selftest.py              offline correctness checks (50 checks)
docs/
  AI_STRATEGY.md           how to get value from the AI tokens
  LEARNING.md              ← how the bot improves itself, and the guard rails
data/                      journal.db + learned state — BACK THIS UP
```

---

## The strategy

`ema_atr` — a plain, auditable trend-following baseline, not a magic system:

- **Regime**: price above EMA200 and EMA20 > EMA50 (long; inverse for short)
- **Strength**: ADX ≥ 18, to skip chop
- **Trigger**: pullback to within 0.8 ATR of EMA20, then a rejection close
- **Stop**: 1.5 ATR, pushed beyond the recent 20-bar swing if that's further
- **Target**: 2R

It's a starting point that you can reason about and that the AI gate has real
context to judge. Add your own by subclassing `Strategy` and registering it in
[core/strategy/\_\_init\_\_.py](core/strategy/__init__.py).

---

## The AI layer

Read **[docs/AI_STRATEGY.md](docs/AI_STRATEGY.md)** — it's the design rationale
and, more importantly, the protocol for proving whether the gate actually helps.

The short version:

- The deterministic strategy owns every number. The model only gets a veto.
- `size_multiplier` is clamped to a hard maximum of 1.0 in code, so a
  hallucinated value cannot enlarge a position.
- Output is schema-constrained JSON — the engine never parses prose.
- **Run `ai.shadow_mode: true` for the first few weeks.** It trades every signal
  while recording what the gate *would* have done, so `python run.py report`
  can tell you whether the gate is protecting you or costing you money. A
  rejected trade you never took has no P&L to compare against — this is the
  only way to find out.

---

## Disclaimer

Trading leveraged instruments carries substantial risk of loss. This is
software, not financial advice, and it comes with no warranty. Backtest results
— especially on the synthetic feed — do not predict live performance. Run it on
a demo account until *you* have evidence it works, and never risk money you
can't afford to lose.
