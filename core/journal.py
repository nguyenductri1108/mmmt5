"""SQLite trade journal.

Every signal is recorded — including the ones that were blocked — together with
the AI verdict. That table is the raw material for the weekly review loop
described in docs/AI_STRATEGY.md: without rejected signals you can never tell
whether the gate is helping or just costing you trades.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import AIVerdict, Position, Signal

log = logging.getLogger("journal")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    bar_time        TEXT,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT,
    side            TEXT    NOT NULL,
    entry           REAL,
    sl              REAL,
    tp              REAL,
    rr              REAL,
    volume          REAL,
    risk_amount     REAL,
    strategy_reason TEXT,
    context_json    TEXT,
    outcome         TEXT    NOT NULL,   -- executed | blocked_risk | blocked_ai | order_failed
    block_reason    TEXT,
    ai_provider     TEXT,
    ai_model        TEXT,
    ai_decision     TEXT,
    ai_confidence   REAL,
    ai_size_mult    REAL,
    ai_reasons      TEXT,
    ai_risks        TEXT,
    ai_latency_ms   INTEGER,
    ticket          INTEGER
);

CREATE TABLE IF NOT EXISTS trades (
    ticket        INTEGER PRIMARY KEY,
    signal_id     INTEGER,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    volume        REAL,
    open_time     TEXT,
    open_price    REAL,
    sl            REAL,
    tp            REAL,
    close_time    TEXT,
    close_price   REAL,
    pnl           REAL,
    close_reason  TEXT,
    r_multiple    REAL
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_trades_close ON trades(close_time);
"""


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover
            pass

    # ---------------------------------------------------------------- writes
    def record_signal(
        self,
        signal: Signal,
        outcome: str,
        verdict: AIVerdict | None = None,
        block_reason: str = "",
        ticket: int = 0,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO signals (
                ts, bar_time, symbol, timeframe, side, entry, sl, tp, rr, volume,
                risk_amount, strategy_reason, context_json, outcome, block_reason,
                ai_provider, ai_model, ai_decision, ai_confidence, ai_size_mult,
                ai_reasons, ai_risks, ai_latency_ms, ticket
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _now(),
                signal.bar_time.isoformat() if signal.bar_time else None,
                signal.symbol,
                signal.timeframe,
                signal.side.value,
                signal.entry,
                signal.sl,
                signal.tp,
                round(signal.rr, 3),
                signal.volume,
                signal.risk_amount,
                signal.reason,
                json.dumps(signal.context, default=str),
                outcome,
                block_reason,
                verdict.provider if verdict else None,
                verdict.model if verdict else None,
                verdict.decision if verdict else None,
                verdict.confidence if verdict else None,
                verdict.size_multiplier if verdict else None,
                json.dumps(verdict.reasons) if verdict else None,
                json.dumps(verdict.risks) if verdict else None,
                verdict.latency_ms if verdict else None,
                ticket or None,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def open_trade(self, ticket: int, signal: Signal, price: float, signal_id: int = 0) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trades
                (ticket, signal_id, symbol, side, volume, open_time, open_price, sl, tp)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                ticket,
                signal_id or None,
                signal.symbol,
                signal.side.value,
                signal.volume,
                _now(),
                price,
                signal.sl,
                signal.tp,
            ),
        )
        self.conn.commit()

    def close_trade(self, ticket: int, price: float, pnl: float, reason: str) -> None:
        row = self.conn.execute(
            "SELECT open_price, sl, side FROM trades WHERE ticket = ?", (ticket,)
        ).fetchone()

        r_multiple = None
        if row and row["sl"] is not None and row["open_price"] is not None:
            risk = abs(row["open_price"] - row["sl"])
            if risk > 0:
                sign = 1 if row["side"] == "BUY" else -1
                r_multiple = round((price - row["open_price"]) * sign / risk, 3)

        self.conn.execute(
            """
            UPDATE trades
               SET close_time = ?, close_price = ?, pnl = ?, close_reason = ?, r_multiple = ?
             WHERE ticket = ?
            """,
            (_now(), price, round(pnl, 2), reason, r_multiple, ticket),
        )
        self.conn.commit()

    def update_stops(self, ticket: int, sl: float, tp: float) -> None:
        self.conn.execute("UPDATE trades SET sl = ?, tp = ? WHERE ticket = ?", (sl, tp, ticket))
        self.conn.commit()

    # ---------------------------------------------------------------- reads
    def known_tickets(self) -> set[int]:
        rows = self.conn.execute("SELECT ticket FROM trades WHERE close_time IS NULL").fetchall()
        return {int(r["ticket"]) for r in rows}

    def open_trade_row(self, ticket: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,)).fetchone()
        return dict(row) if row else None

    def recent_closed(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT symbol, side, volume, open_price, close_price, pnl, close_reason, close_time
              FROM trades
             WHERE close_time IS NOT NULL
             ORDER BY close_time DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def stats(self, days: int = 30) -> dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT COUNT(*)                                  AS trades,
                   COALESCE(SUM(pnl), 0)                     AS net,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) AS gross_win,
                   COALESCE(SUM(CASE WHEN pnl < 0 THEN -pnl ELSE 0 END), 0) AS gross_loss
              FROM trades
             WHERE close_time IS NOT NULL
               AND close_time >= datetime('now', ?)
            """,
            (f"-{days} days",),
        ).fetchone()

        trades = int(row["trades"] or 0)
        wins = int(row["wins"] or 0)
        gross_loss = float(row["gross_loss"] or 0)
        return {
            "trades": trades,
            "wins": wins,
            "win_rate": round(100.0 * wins / trades, 1) if trades else 0.0,
            "net": round(float(row["net"] or 0), 2),
            "profit_factor": round(float(row["gross_win"]) / gross_loss, 2) if gross_loss else None,
        }

    def ai_gate_report(self, days: int = 30) -> dict[str, Any]:
        """How the gate is behaving: approvals vs rejections and why."""
        rows = self.conn.execute(
            """
            SELECT outcome, ai_decision, COUNT(*) AS n
              FROM signals
             WHERE ts >= datetime('now', ?)
             GROUP BY outcome, ai_decision
            """,
            (f"-{days} days",),
        ).fetchall()
        return {f"{r['outcome']}/{r['ai_decision']}": r["n"] for r in rows}


    def shadow_report(self, days: int = 90) -> dict[str, Any]:
        """Did the AI gate add value?

        Compares executed trades the gate approved against executed trades it
        wanted to reject (traded anyway in shadow mode, or as calibration
        audits while enforcing). If the rejected bucket is *profitable*, the
        gate is costing you money — and the calibrator will demote it to
        shadow automatically.
        """
        rows = self.conn.execute(
            """
            SELECT CASE WHEN s.ai_decision = 'reject'
                        THEN 'blocked' ELSE 'approved' END    AS bucket,
                   COUNT(*)                                   AS trades,
                   COALESCE(SUM(t.pnl), 0)                    AS net,
                   COALESCE(AVG(t.pnl), 0)                    AS avg_pnl,
                   COALESCE(AVG(CASE WHEN s.risk_amount > 0
                                     THEN t.pnl / s.risk_amount END), 0) AS avg_r,
                   COALESCE(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
              FROM signals s
              JOIN trades  t ON t.ticket = s.ticket
             WHERE s.ts >= datetime('now', ?)
               AND t.close_time IS NOT NULL
               AND s.outcome = 'executed'
               AND s.ai_decision IN ('approve', 'reject')
             GROUP BY bucket
            """,
            (f"-{days} days",),
        ).fetchall()

        out: dict[str, Any] = {}
        for row in rows:
            trades = int(row["trades"])
            label = "gate approved" if row["bucket"] == "approved" else "gate would have blocked"
            out[label] = {
                "trades": trades,
                "net": round(float(row["net"]), 2),
                "avg_pnl": round(float(row["avg_pnl"]), 2),
                "avg_r": round(float(row["avg_r"]), 3),
                "win_rate": round(100.0 * int(row["wins"]) / trades, 1) if trades else 0.0,
            }
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
