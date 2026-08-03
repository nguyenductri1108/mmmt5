"""Telegram: alerts out, commands in.

Outbound is synchronous but never raises into the trading loop — a Telegram
outage must not stop or crash the bot.

Inbound runs on a daemon thread doing long-poll `getUpdates`, pushing commands
onto a queue the engine drains once per tick. Only the configured chat id is
honoured; anything else is ignored and logged.
"""

from __future__ import annotations

import html
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from core.config import env
from core.models import AccountInfo, AIVerdict, Position, Signal

log = logging.getLogger("telegram")

API = "https://api.telegram.org/bot{token}/{method}"


class Telegram:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("telegram.enabled", True))
        self.token = env("TELEGRAM_BOT_TOKEN")
        self.chat_id = env("TELEGRAM_CHAT_ID")
        self.commands_enabled = bool(cfg.get("telegram.commands_enabled", True))

        self.notify_signals = bool(cfg.get("telegram.notify_signals", True))
        self.notify_rejects = bool(cfg.get("telegram.notify_rejects", True))
        self.notify_fills = bool(cfg.get("telegram.notify_fills", True))
        self.notify_closes = bool(cfg.get("telegram.notify_closes", True))

        self.commands: queue.Queue[tuple[str, list[str]]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._session = requests.Session()

        if self.enabled and not (self.token and self.chat_id):
            log.warning(
                "Telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set "
                "in .env — notifications are disabled."
            )
            self.enabled = False

    # ---------------------------------------------------------------- transport
    def _call(self, method: str, payload: dict[str, Any], timeout: int = 10) -> dict | None:
        if not self.token:
            return None
        try:
            resp = self._session.post(
                API.format(token=self.token, method=method), json=payload, timeout=timeout
            )
            data = resp.json()
            if not data.get("ok"):
                log.warning("telegram %s failed: %s", method, data.get("description"))
                return None
            return data
        except requests.RequestException as exc:
            log.warning("telegram %s error: %s", method, exc)
            return None

    def send(self, text: str, silent: bool = False) -> None:
        """Fire-and-forget message. Never raises."""
        if not self.enabled:
            return
        self._call(
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
                "disable_notification": silent,
            },
        )

    # ---------------------------------------------------------------- commands
    def start_command_listener(self) -> None:
        if not (self.enabled and self.commands_enabled) or self._thread:
            return
        self._thread = threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True)
        self._thread.start()
        log.info("Telegram command listener started")

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            data = self._call(
                "getUpdates",
                {"offset": self._offset, "timeout": 25, "allowed_updates": ["message"]},
                timeout=35,
            )
            if not data:
                time.sleep(3)
                continue
            for update in data.get("result", []):
                self._offset = update["update_id"] + 1
                self._handle_update(update)

    def _handle_update(self, update: dict) -> None:
        message = update.get("message") or {}
        chat = str((message.get("chat") or {}).get("id", ""))
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        if chat != str(self.chat_id):
            log.warning("ignoring command from unauthorised chat %s: %s", chat, text)
            return

        parts = text.split()
        command = parts[0].lstrip("/").split("@")[0].lower()
        self.commands.put((command, parts[1:]))
        log.info("telegram command received: /%s %s", command, " ".join(parts[1:]))

    def drain_commands(self) -> list[tuple[str, list[str]]]:
        out = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out

    # ---------------------------------------------------------------- messages
    def signal_alert(self, signal: Signal, verdict: AIVerdict, account: AccountInfo) -> None:
        if not self.notify_signals:
            return
        arrow = "🟢 BUY" if signal.side.value == "BUY" else "🔴 SELL"
        risk_pct = 100.0 * signal.risk_amount / account.equity if account.equity else 0.0

        lines = [
            f"<b>{arrow} {esc(signal.symbol)}</b>  <i>{esc(signal.timeframe)}</i>",
            "",
            f"Entry   <code>{signal.entry}</code>",
            f"SL      <code>{signal.sl}</code>  ({fmt_dist(signal.risk_distance)})",
            f"TP      <code>{signal.tp}</code>  ({fmt_dist(signal.reward_distance)})",
            f"R:R     <code>1 : {signal.rr:.2f}</code>",
            f"Lots    <code>{signal.volume}</code>",
            f"Risk    <code>{signal.risk_amount:.2f} {esc(account.currency)}</code>"
            f" ({risk_pct:.2f}% of equity)",
            "",
            f"<b>Setup</b>: {esc(signal.reason)}",
        ]
        lines += _verdict_lines(verdict)
        self.send("\n".join(lines))

    def reject_alert(self, signal: Signal, reason: str, stage: str) -> None:
        if not self.notify_rejects:
            return
        self.send(
            f"⚪️ <b>SKIPPED</b> {esc(signal.symbol)} {esc(signal.side.value)}"
            f"  <i>{esc(signal.timeframe)}</i>\n"
            f"Blocked at: <b>{esc(stage)}</b>\n"
            f"{esc(reason)}",
            silent=True,
        )

    def fill_alert(self, signal: Signal, ticket: int, price: float) -> None:
        if not self.notify_fills:
            return
        arrow = "🟢" if signal.side.value == "BUY" else "🔴"
        slip = price - signal.entry
        self.send(
            f"{arrow} <b>FILLED</b> #{ticket}\n"
            f"{esc(signal.symbol)} {esc(signal.side.value)} <code>{signal.volume}</code> lots\n"
            f"Price   <code>{price}</code>  (slip {slip:+.5f})\n"
            f"SL <code>{signal.sl}</code>   TP <code>{signal.tp}</code>"
        )

    def order_failed(self, signal: Signal, error: str) -> None:
        self.send(
            f"⚠️ <b>ORDER FAILED</b> {esc(signal.symbol)} {esc(signal.side.value)}\n"
            f"<code>{esc(error)}</code>"
        )

    def close_alert(self, position: Position, price: float, pnl: float, why: str) -> None:
        if not self.notify_closes:
            return
        icon = "✅" if pnl >= 0 else "❌"
        self.send(
            f"{icon} <b>CLOSED</b> #{position.ticket} {esc(position.symbol)} "
            f"{esc(position.side.value)}\n"
            f"Open <code>{position.price_open}</code> → <code>{price}</code>\n"
            f"P&amp;L <b>{pnl:+.2f}</b>   ({esc(why)})"
        )

    def status(
        self,
        account: AccountInfo,
        positions: list[Position],
        paused: bool,
        halted: str,
        day_pnl: float,
        day_pnl_pct: float,
        trades_today: int,
        mode: str,
    ) -> None:
        state = "⏸ PAUSED" if paused else ("⛔️ HALTED" if halted else "▶️ RUNNING")
        lines = [
            f"<b>{state}</b>  <i>({esc(mode)})</i>",
            "",
            f"Equity   <code>{account.equity:.2f} {esc(account.currency)}</code>",
            f"Balance  <code>{account.balance:.2f}</code>",
            f"Free mgn <code>{account.free_margin_pct:.1f}%</code>",
            f"Day P&amp;L <b>{day_pnl:+.2f}</b> ({day_pnl_pct:+.2f}%)",
            f"Trades   <code>{trades_today}</code> today",
        ]
        if halted:
            lines.append(f"\n⛔️ <b>Halt reason</b>: {esc(halted)}")
        lines.append("")
        if positions:
            lines.append(f"<b>Open positions ({len(positions)})</b>")
            for p in positions:
                lines.append(
                    f"#{p.ticket} {esc(p.symbol)} {esc(p.side.value)} "
                    f"<code>{p.volume}</code> @ <code>{p.price_open}</code> "
                    f"→ <b>{p.profit:+.2f}</b> ({p.r_multiple():+.2f}R)"
                )
        else:
            lines.append("<i>No open positions.</i>")
        self.send("\n".join(lines))

    def help(self) -> None:
        self.send(
            "<b>Commands</b>\n"
            "/status — account, day P&amp;L, open positions\n"
            "/positions — open positions only\n"
            "/pause — stop opening new trades (existing ones keep running)\n"
            "/resume — resume trading\n"
            "/close &lt;ticket|symbol&gt; — close one position or all on a symbol\n"
            "/flat — close everything now\n"
            "/kill — flatten everything and stop the bot\n"
            "/learning — self-improvement status (gate mode, lessons, params)\n"
            "/revert — undo the last auto-promoted parameter set\n"
            "/help — this message"
        )

    def startup(self, mode: str, symbols: list[str], account: AccountInfo, ai: str) -> None:
        self.send(
            f"🤖 <b>BOT_MT5 started</b>\n"
            f"Mode     <code>{esc(mode)}</code>\n"
            f"Symbols  <code>{esc(', '.join(symbols))}</code>\n"
            f"Equity   <code>{account.equity:.2f} {esc(account.currency)}</code>\n"
            f"AI gate  <code>{esc(ai)}</code>\n"
            f"<i>{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</i>\n\n"
            "Send /help for controls."
        )

    def shutdown_notice(self, reason: str) -> None:
        self.send(f"🛑 <b>BOT_MT5 stopped</b>\n{esc(reason)}")


def _verdict_lines(verdict: AIVerdict) -> list[str]:
    if verdict.provider in ("disabled", ""):
        return []
    icon = "🧠✅" if verdict.approved else "🧠🚫"
    lines = [
        "",
        f"{icon} <b>AI {verdict.decision.upper()}</b>"
        f"  <i>{esc(verdict.provider)}/{esc(verdict.model or 'n/a')}"
        f" · {verdict.confidence:.0%} · {verdict.latency_ms}ms</i>",
    ]
    if verdict.size_multiplier < 1.0:
        lines.append(f"Size scaled to <b>{verdict.size_multiplier:.0%}</b>")
    for reason in verdict.reasons[:3]:
        lines.append(f"• {esc(reason)}")
    for risk in verdict.risks[:2]:
        lines.append(f"⚠️ {esc(risk)}")
    return lines


def esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def fmt_dist(distance: float) -> str:
    return f"{distance:.5f}".rstrip("0").rstrip(".")
