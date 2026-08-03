"""CLI wrapper around core.backtest — the shared runner also used by the
weekly optimizer, so what you eyeball here is exactly what promotions are
judged on.

    python run.py backtest --bars 20000
"""

from __future__ import annotations

import logging

from core.backtest import run_quiet

log = logging.getLogger("backtest")


def run_backtest(cfg, bars: int = 5000) -> int:
    feed = str(cfg.get("paper.feed", "synthetic")).lower()
    symbols = [s["name"] for s in cfg.enabled_symbols()]
    timeframe = str(cfg.get("engine.timeframe", "M15")).upper()

    print(f"\nReplaying {bars} bars · {timeframe} · {', '.join(symbols)} · feed={feed}\n")

    result = run_quiet(cfg, bars=bars, allow_synthetic=True)

    print("=" * 58)
    print("BACKTEST RESULT")
    print("=" * 58)
    print(f"signals generated : {result.signals}")
    print(f"trades executed   : {result.trades}")

    if result.trades:
        losses = result.trades - result.wins
        print(f"win rate          : {result.win_rate:.1f}%  ({result.wins}W / {losses}L)")
        print(f"expectancy        : {result.expectancy_r:+.3f}R per trade")
        pf = result.profit_factor
        print(f"profit factor     : {pf if pf is not None else 'n/a'}")
        print(f"net P&L           : {result.net:+.2f}  "
              f"({100 * result.net / result.start_balance:+.2f}%)")
        print(f"max drawdown      : {result.max_dd_pct:.2f}%")
        print(f"end balance       : {result.end_balance:.2f}")
    else:
        print("no trades were executed — check the risk filters below")

    if result.blocked:
        print("\nsignals blocked by risk:")
        for reason, count in sorted(result.blocked.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<42} {count}")

    if result.exits:
        print("\nexits:", ", ".join(f"{k}={v}" for k, v in sorted(result.exits.items())))

    print("=" * 58 + "\n")
    if feed != "csv":
        print(
            "NOTE: synthetic-feed results measure plumbing, not edge. Point\n"
            "      paper.feed at real CSV history before believing any number.\n"
        )
    return 0
