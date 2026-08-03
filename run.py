#!/usr/bin/env python3
"""BOT_MT5 entry point.

  python run.py                    # run with mode from config.yaml
  python run.py --mode paper       # simulated broker (works on macOS/Linux)
  python run.py --mode live        # MetaTrader 5 (Windows only)
  python run.py chatid             # print your Telegram chat id
  python run.py check              # verify config, connection and AI keys
  python run.py backtest           # replay the paper feed, no Telegram, no AI
  python run.py report             # journal summary
"""

from __future__ import annotations

import argparse
import sys

from core.config import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(prog="run.py", description="MT5 trading bot")
    parser.add_argument(
        "command",
        nargs="?",
        default="trade",
        choices=["trade", "check", "chatid", "backtest", "report", "selftest"],
    )
    parser.add_argument("--mode", choices=["paper", "live"], help="override config mode")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--bars", type=int, default=5000, help="backtest: bars to replay")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    mode = args.mode or str(cfg.get("mode", "paper"))

    if args.command == "chatid":
        return cmd_chatid()
    if args.command == "check":
        return cmd_check(cfg, mode, log)
    if args.command == "report":
        return cmd_report(cfg)
    if args.command == "selftest":
        from scripts.selftest import run_selftest

        return run_selftest(cfg)
    if args.command == "backtest":
        from scripts.backtest import run_backtest

        return run_backtest(cfg, bars=args.bars)

    from core.engine import Engine

    if mode == "live":
        log.warning("LIVE MODE — real orders will be sent to the broker.")

    Engine(cfg, mode).run()
    return 0


# ---------------------------------------------------------------------------
def cmd_chatid() -> int:
    """Print chat ids that have messaged the bot, so you can fill in .env."""
    import requests

    from core.config import env

    token = env("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20)
    data = resp.json()
    if not data.get("ok"):
        print(f"Telegram error: {data.get('description')}")
        return 1

    seen: dict[str, str] = {}
    for update in data.get("result", []):
        chat = (update.get("message") or {}).get("chat") or {}
        if chat.get("id"):
            name = chat.get("username") or chat.get("title") or chat.get("first_name") or "?"
            seen[str(chat["id"])] = name

    if not seen:
        print("No messages yet. Open Telegram, send your bot any message, then rerun this.")
        return 1

    print("Found chat id(s) — put one in .env as TELEGRAM_CHAT_ID:\n")
    for chat_id, name in seen.items():
        print(f"  {chat_id}   ({name})")
    return 0


def cmd_check(cfg, mode: str, log) -> int:
    """Pre-flight: config, broker connection, symbol specs, AI, Telegram."""
    import os

    ok = True
    print(f"\n=== BOT_MT5 pre-flight ({mode}) ===\n")

    symbols = [s["name"] for s in cfg.enabled_symbols()]
    print(f"config      : strategy={cfg.get('strategy.name')} tf={cfg.get('engine.timeframe')}")
    print(f"symbols     : {', '.join(symbols) or '(none enabled!)'}")
    ok &= bool(symbols)

    # Telegram
    from core.notify import Telegram

    tg = Telegram(cfg)
    if tg.enabled:
        tg.send("✅ <b>Pre-flight check</b> — Telegram is wired up correctly.")
        print("telegram    : OK (test message sent)")
    else:
        print("telegram    : DISABLED (missing token/chat id, or turned off in config)")

    # AI
    from core.ai import make_advisor

    ai = make_advisor(cfg)
    for advisor in ai.chain:
        state = "ready" if advisor.available() else "unavailable (SDK or API key missing)"
        print(f"ai/{advisor.provider:<8}: {state}")
    if ai.enabled and not ai.usable:
        print(f"              ^ on_error={ai.on_error} will apply to every signal")
        ok = ok and ai.on_error == "approve"

    if os.environ.get("ANTHROPIC_API_KEY") and ai.enabled:
        print("              (run a live probe with: python run.py backtest)")

    # Broker
    try:
        from core.broker import make_broker

        broker = make_broker(cfg, mode)
        broker.connect()
        account = broker.account()
        print(f"broker      : OK — equity {account.equity:.2f} {account.currency}")
        for name in symbols:
            info = broker.symbol_info(name)
            tick = broker.tick(name)
            print(
                f"  {name:<10} spread={tick.spread_points(info):.0f}pt "
                f"digits={info.digits} lot={info.volume_min}-{info.volume_max}"
                f"/{info.volume_step} tick_value={info.tick_value}"
            )
        broker.shutdown()
    except Exception as exc:  # noqa: BLE001
        print(f"broker      : FAILED — {exc}")
        ok = False

    print(f"\n=== {'ALL GOOD' if ok else 'PROBLEMS FOUND — see above'} ===\n")
    return 0 if ok else 1


def cmd_report(cfg) -> int:
    from core.journal import Journal

    journal = Journal(cfg.root / "data" / "journal.db")
    for window in (7, 30, 90):
        stats = journal.stats(window)
        pf = stats["profit_factor"]
        print(
            f"last {window:>3}d: {stats['trades']:>4} trades  "
            f"win {stats['win_rate']:>5.1f}%  net {stats['net']:>10.2f}  "
            f"PF {pf if pf is not None else 'n/a'}"
        )

    print("\nAI gate (30d):")
    report = journal.ai_gate_report(30)
    if not report:
        print("  no signals recorded yet")
    for key, count in sorted(report.items()):
        print(f"  {key:<28} {count}")

    shadow = journal.shadow_report(90)
    if shadow:
        print("\nAI gate value (90d, shadow mode):")
        for bucket, stats in shadow.items():
            print(
                f"  {bucket:<24} {stats['trades']:>4} trades  "
                f"win {stats['win_rate']:>5.1f}%  net {stats['net']:>10.2f}  "
                f"avg {stats['avg_pnl']:>8.2f}"
            )
        blocked = shadow.get("gate would have blocked")
        if blocked and blocked["net"] > 0:
            print("  ^ the gate wanted to block PROFITABLE trades — loosen or drop it")

    print("\nRecent closed trades:")
    for trade in journal.recent_closed(10):
        print(
            f"  {trade['close_time'][:16]}  {trade['symbol']:<8} {trade['side']:<4} "
            f"{trade['volume']:>5}  {trade['pnl']:>9.2f}  {trade['close_reason']}"
        )
    journal.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
