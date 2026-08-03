"""The trading loop.

Order of operations on every tick, and the reason for it:

  1. advance the feed (paper only)
  2. drain Telegram commands   — a /pause or /flat must be honoured *before*
                                 anything else this tick can open a trade
  3. reconcile positions       — detect closes, notify, journal
  4. manage open positions     — breakeven / trailing stops
  5. gate checks               — halted? paused? outside session? daily limit?
  6. per symbol: new closed bar -> strategy -> risk -> AI -> execute

Steps 3 and 4 run even while paused or halted: an existing position must keep
being managed no matter what state the entry side is in.
"""

from __future__ import annotations

import logging
import random
import signal as os_signal
import time
from datetime import datetime, time as dtime, timedelta, timezone

from core.ai import make_advisor
from core.ai.base import build_review_payload
from core.broker import make_broker
from core.config import Config
from core.indicators import atr, to_frame
from core.journal import Journal
from core.models import AccountInfo, Position, Side, Signal
from core.notify import Telegram
from core.risk import RiskManager
from core.strategy import make_strategy

log = logging.getLogger("engine")


class Engine:
    def __init__(self, cfg: Config, mode: str) -> None:
        self.cfg = cfg
        self.mode = mode
        self.timeframe = str(cfg.get("engine.timeframe", "M15")).upper()
        self.candle_count = int(cfg.get("engine.candles", 400))
        self.loop_seconds = float(cfg.get("engine.loop_seconds", 5))

        self.broker = make_broker(cfg, mode)
        self.risk = RiskManager(cfg)
        self.ai = make_advisor(cfg)
        self.tg = Telegram(cfg)
        self.journal = Journal(cfg.root / "data" / "journal.db")

        # Learning system: must exist BEFORE the strategy is built, because
        # promoted parameter overrides merge over config.yaml's params.
        self.learning = None
        if bool(cfg.get("learning.enabled", True)):
            from core.learn import LearningSystem

            self.learning = LearningSystem(cfg, self.journal, self.ai)
            overrides = self.learning.active_param_overrides()
            if overrides:
                merged = dict(cfg.get("strategy.params", {}) or {})
                merged.update(overrides)
                cfg.set("strategy.params", merged)
                log.info("Applied promoted parameter overrides: %s", overrides)
            # The calibrated confidence floor overrides the static config value.
            self.ai.min_confidence = self.learning.min_confidence

        self.strategy = make_strategy(cfg)

        self.running = True
        self.paused = False
        self.stop_reason = "shutdown requested"

        self._rng = random.Random()
        self._last_bar: dict[str, datetime] = {}
        self._position_cache: dict[int, Position] = {}
        self._last_heartbeat = datetime.now(timezone.utc)
        self._friday_flat_done: str = ""   # ISO date we already flattened on

    # ================================================================== lifecycle
    def run(self) -> None:
        self._install_signal_handlers()
        self.broker.connect()

        account = self.broker.account()
        self.risk.roll_day(account)
        self._position_cache = {p.ticket: p for p in self.broker.positions()}

        self.tg.start_command_listener()
        self.tg.startup(
            self.mode,
            [s["name"] for s in self.cfg.enabled_symbols()],
            account,
            self._ai_summary(),
        )
        log.info(
            "Engine running | mode=%s | tf=%s | symbols=%s",
            self.mode, self.timeframe,
            ", ".join(s["name"] for s in self.cfg.enabled_symbols()),
        )

        try:
            while self.running:
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - a bad tick must not kill the bot
                    log.exception("unhandled error in tick")
                time.sleep(self.loop_seconds)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("Shutting down: %s", self.stop_reason)
        self.tg.shutdown_notice(self.stop_reason)
        self.tg.stop()
        self.journal.close()
        self.broker.shutdown()

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.stop_reason = f"received signal {signum}"
            self.running = False

        os_signal.signal(os_signal.SIGINT, handler)
        os_signal.signal(os_signal.SIGTERM, handler)

    # ================================================================== main tick
    def tick(self) -> None:
        self.broker.pump()

        account = self.broker.account()
        if self.risk.roll_day(account):
            self._friday_flat_done = ""

        if self.learning:
            now = datetime.now(timezone.utc)
            self.learning.maybe_schedule(now, self.broker, self.mode)
            report = self.learning.poll_pending()
            if report is not None:
                self._apply_learning(report)

        self._handle_commands(account)
        if not self.running:
            return

        positions = self.broker.positions()
        self._reconcile_positions(positions)

        if self.cfg.get("manage.enabled", True):
            self._manage_positions(positions)

        self._heartbeat(account, positions)

        if self._friday_flat_check():
            return

        if self.paused:
            return

        halt = self.risk.daily_limit_breached(account)
        if halt:
            if self.risk.halted_reason != halt:
                self.risk.halted_reason = halt
                log.warning("Trading halted: %s", halt)
                self.tg.send(f"⛔️ <b>Trading halted for today</b>\n{halt}")
            return

        if not self._in_session():
            return

        for sym_conf in self.cfg.enabled_symbols():
            try:
                self._evaluate_symbol(sym_conf["name"], account)
            except Exception:  # noqa: BLE001 - one bad symbol must not stop the rest
                log.exception("error evaluating %s", sym_conf["name"])

    # ================================================================== entries
    def _evaluate_symbol(self, symbol: str, account: AccountInfo) -> None:
        candles = self.broker.candles(symbol, self.timeframe, self.candle_count)
        if len(candles) < 60:
            return

        # candles[-1] is still forming; candles[-2] is the bar we act on.
        closed_bar_time = candles[-2].time
        if self._last_bar.get(symbol) == closed_bar_time:
            return
        self._last_bar[symbol] = closed_bar_time

        info = self.broker.symbol_info(symbol)
        sig = self.strategy.generate(symbol, self.timeframe, candles, info)
        if sig is None:
            return

        log.info(
            "Signal %s %s @ %s SL %s TP %s (%s)",
            sig.side.value, symbol, sig.entry, sig.sl, sig.tp, sig.reason,
        )

        tick = self.broker.tick(symbol)
        positions = self.broker.positions()

        decision = self.risk.check(sig, account, info, tick, positions)
        if not decision.ok:
            log.info("Risk blocked %s: %s", symbol, decision.reason)
            self.journal.record_signal(sig, "blocked_risk", block_reason=decision.reason)
            self.tg.reject_alert(sig, decision.reason, "risk")
            return

        sig.volume = decision.volume
        sig.risk_amount = decision.risk_amount

        # --- episodic memory: how did similar past setups actually go? -------
        evidence = None
        if self.learning:
            evidence = self.learning.evidence_for(sig)
            if evidence.multiplier < 1.0:
                scaled = info.normalize_volume(sig.volume * evidence.multiplier)
                if scaled >= info.volume_min:
                    sig.volume = scaled
                    sig.risk_amount = self.risk.risk_for_volume(sig, info, scaled)
                    log.info(
                        "%s size scaled to %.0f%% by evidence (%s)",
                        symbol, evidence.multiplier * 100, evidence.note,
                    )

        payload = build_review_payload(
            sig,
            account,
            positions,
            sig.risk_amount,
            self.risk.daily_pnl_pct(account),
            self.risk.trades_today,
            self.journal.recent_closed(10),
        )
        if evidence is not None:
            payload["similar_past_setups"] = evidence.payload()

        lessons = self.learning.lessons_block() if self.learning else ""
        verdict = self.ai.review(payload, lessons)

        # Gate mode: calibrated automatically when learning is on, otherwise
        # the static config flag.
        if self.learning:
            shadow = self.learning.gate_mode == "shadow"
        else:
            shadow = bool(self.cfg.get("ai.shadow_mode", False))

        audit = False
        if not verdict.approved:
            detail = "; ".join(verdict.reasons) or verdict.error or "no reason given"
            if not shadow and self.learning:
                # Audit slice: while enforcing, a random fraction of rejections
                # still executes so the calibrator can keep measuring whether
                # rejections actually dodge losers.
                audit = self._rng.random() < self.learning.audit_fraction
            if not shadow and not audit:
                log.info("AI rejected %s: %s", symbol, detail)
                self.journal.record_signal(sig, "blocked_ai", verdict=verdict, block_reason=detail)
                self.tg.reject_alert(sig, detail, f"AI ({verdict.provider})")
                return
            # Shadow / audit: trade anyway. The executed journal row keeps
            # ai_decision='reject', which is exactly what the calibrator and
            # shadow report measure the gate against.
            tag = "AUDIT" if audit else "SHADOW"
            log.info("AI rejected %s (%s — trading anyway): %s", symbol, tag, detail)

        # The model may shrink a position but never grow one (clamped at parse
        # time). Shadow/audit trades skip the multiplier so the measured sample
        # stays comparable to normal approvals.
        if verdict.approved and verdict.size_multiplier < 1.0:
            scaled = info.normalize_volume(sig.volume * verdict.size_multiplier)
            if scaled < info.volume_min:
                reason = (
                    f"AI size multiplier {verdict.size_multiplier:.2f} takes volume "
                    f"below the broker minimum ({info.volume_min})"
                )
                self.journal.record_signal(sig, "blocked_ai", verdict=verdict, block_reason=reason)
                self.tg.reject_alert(sig, reason, "sizing")
                return
            sig.volume = scaled
            sig.risk_amount = self.risk.risk_for_volume(sig, info, scaled)

        self.tg.signal_alert(sig, verdict, account)
        self._execute(sig, verdict)

    def _execute(self, sig: Signal, verdict) -> None:
        result = self.broker.market_order(
            symbol=sig.symbol,
            side=sig.side,
            volume=sig.volume,
            sl=sig.sl,
            tp=sig.tp,
            comment=f"BOT {self.strategy.name}",
        )

        if not result.ok:
            log.error("Order failed for %s: %s", sig.symbol, result.error)
            self.journal.record_signal(sig, "order_failed", verdict=verdict, block_reason=result.error)
            self.tg.order_failed(sig, result.error)
            return

        signal_id = self.journal.record_signal(sig, "executed", verdict=verdict, ticket=result.ticket)
        self.journal.open_trade(result.ticket, sig, result.price, signal_id)
        self.risk.record_trade()

        log.info("Filled #%d %s %s %.2f @ %s",
                 result.ticket, sig.side.value, sig.symbol, sig.volume, result.price)
        self.tg.fill_alert(sig, result.ticket, result.price)

        # Refresh the cache immediately so reconciliation doesn't see a phantom close.
        for pos in self.broker.positions(sig.symbol):
            self._position_cache[pos.ticket] = pos

    # ================================================================== management
    def _reconcile_positions(self, positions: list[Position]) -> None:
        """Spot positions that vanished since last tick and report them."""
        live = {p.ticket for p in positions}
        for ticket, cached in list(self._position_cache.items()):
            if ticket in live:
                continue

            info = self.broker.closed_trade_info(ticket)
            if info is not None:
                price, pnl, why = info
            else:
                # Fall back to the last state we observed before it disappeared.
                price, pnl, why = cached.price_current, cached.profit, _infer_close_reason(cached)

            log.info("Position #%d closed @ %s pnl=%.2f (%s)", ticket, price, pnl, why)
            self.journal.close_trade(ticket, price, pnl, why)
            self.tg.close_alert(cached, price, pnl, why)
            del self._position_cache[ticket]

        for pos in positions:
            self._position_cache[pos.ticket] = pos

    def _manage_positions(self, positions: list[Position]) -> None:
        be_r = float(self.cfg.get("manage.breakeven_at_r", 0) or 0)
        be_offset_pts = float(self.cfg.get("manage.breakeven_offset_points", 0) or 0)
        trail_mult = float(self.cfg.get("manage.trail_atr_mult", 0) or 0)

        if be_r <= 0 and trail_mult <= 0:
            return

        for pos in positions:
            info = self.broker.symbol_info(pos.symbol)
            new_sl = pos.sl
            note = ""

            if be_r > 0 and pos.r_multiple() >= be_r:
                offset = be_offset_pts * info.point * pos.side.sign
                breakeven = pos.price_open + offset
                if _is_improvement(pos.side, pos.sl, breakeven):
                    new_sl = breakeven
                    note = "breakeven"

            if trail_mult > 0:
                candles = self.broker.candles(pos.symbol, self.timeframe, 100)
                df = to_frame(candles)
                if len(df) > 20:
                    atr_now = float(atr(df, 14).iloc[-2])
                    trail = pos.price_current - trail_mult * atr_now * pos.side.sign
                    if _is_improvement(pos.side, new_sl, trail):
                        new_sl = trail
                        note = "trail"

            if note and abs(new_sl - pos.sl) >= info.point:
                new_sl = info.normalize_price(new_sl)
                result = self.broker.modify_position(pos.ticket, new_sl, pos.tp)
                if result.ok:
                    log.info("#%d SL -> %s (%s)", pos.ticket, new_sl, note)
                    self.journal.update_stops(pos.ticket, new_sl, pos.tp)
                    self.tg.send(
                        f"🛡 <b>#{pos.ticket} {pos.symbol}</b> stop moved to "
                        f"<code>{new_sl}</code> ({note})",
                        silent=True,
                    )
                    pos.sl = new_sl
                else:
                    log.warning("#%d SL modify failed: %s", pos.ticket, result.error)

    # ================================================================== gates
    def _in_session(self) -> bool:
        if not self.cfg.get("session.enabled", True):
            return True

        now = datetime.now(timezone.utc)
        weekdays = self.cfg.get("session.weekdays", [0, 1, 2, 3, 4])
        if now.weekday() not in weekdays:
            return False

        windows = self.cfg.get("session.windows") or []
        if not windows:
            return True

        current = now.time()
        for start, end in windows:
            if _parse_hhmm(start) <= current <= _parse_hhmm(end):
                return True
        return False

    def _friday_flat_check(self) -> bool:
        """Flatten and stand down before the weekend gap. True if we acted."""
        flat_at = self.cfg.get("session.friday_flat_at")
        if not flat_at:
            return False

        now = datetime.now(timezone.utc)
        if now.weekday() != 4 or now.time() < _parse_hhmm(flat_at):
            return False

        today = now.date().isoformat()
        if self._friday_flat_done == today:
            return True

        self._friday_flat_done = today
        positions = self.broker.positions()
        if positions:
            log.info("Friday flat: closing %d position(s)", len(positions))
            self.broker.close_all()
            self.tg.send(
                f"🌙 <b>Weekend flat</b> — closed {len(positions)} position(s) "
                f"at {flat_at} UTC."
            )
        self.paused = True
        return True

    def _heartbeat(self, account: AccountInfo, positions: list[Position]) -> None:
        minutes = int(self.cfg.get("telegram.heartbeat_minutes", 0) or 0)
        if minutes <= 0:
            return
        now = datetime.now(timezone.utc)
        if now - self._last_heartbeat < timedelta(minutes=minutes):
            return
        self._last_heartbeat = now
        self.tg.send(
            f"💓 Alive · equity <code>{account.equity:.2f}</code> · "
            f"day <b>{self.risk.daily_pnl_pct(account):+.2f}%</b> · "
            f"{len(positions)} open",
            silent=True,
        )

    # ================================================================== commands
    def _handle_commands(self, account: AccountInfo) -> None:
        for command, args in self.tg.drain_commands():
            try:
                self._run_command(command, args, account)
            except Exception:  # noqa: BLE001
                log.exception("command /%s failed", command)
                self.tg.send(f"⚠️ Command <code>/{command}</code> failed — see logs.")

    def _run_command(self, command: str, args: list[str], account: AccountInfo) -> None:
        positions = self.broker.positions()

        if command in ("start", "help"):
            self.tg.help()

        elif command == "status":
            self.tg.status(
                account, positions, self.paused, self.risk.halted_reason,
                self.risk.daily_pnl(account), self.risk.daily_pnl_pct(account),
                self.risk.trades_today, self.mode,
            )

        elif command == "positions":
            if not positions:
                self.tg.send("<i>No open positions.</i>")
            else:
                lines = ["<b>Open positions</b>"] + [
                    f"#{p.ticket} {p.symbol} {p.side.value} <code>{p.volume}</code> "
                    f"@ <code>{p.price_open}</code> → <b>{p.profit:+.2f}</b> "
                    f"({p.r_multiple():+.2f}R)"
                    for p in positions
                ]
                self.tg.send("\n".join(lines))

        elif command == "pause":
            self.paused = True
            self.tg.send("⏸ <b>Paused.</b> No new trades. Open positions are still managed.")

        elif command == "resume":
            self.paused = False
            self.risk.halted_reason = ""
            self.tg.send("▶️ <b>Resumed.</b>")

        elif command == "close":
            if not args:
                self.tg.send("Usage: <code>/close &lt;ticket|symbol&gt;</code>")
                return
            target = args[0]
            if target.isdigit():
                result = self.broker.close_position(int(target))
                self.tg.send(
                    f"{'✅' if result.ok else '⚠️'} close #{target}: "
                    f"{'done' if result.ok else result.error}"
                )
            else:
                results = self.broker.close_all(target.upper())
                self.tg.send(f"Closed {sum(1 for r in results if r.ok)} position(s) on {target.upper()}.")

        elif command == "flat":
            results = self.broker.close_all()
            self.tg.send(f"🧹 Flattened {sum(1 for r in results if r.ok)} position(s).")

        elif command == "learning":
            self._learning_status()

        elif command == "revert":
            if not self.learning:
                self.tg.send("Learning system is disabled.")
                return
            previous = self.learning.rollback_params("manual /revert")
            if previous is None:
                self.tg.send("Nothing to revert — no promoted parameter set is active.")
            else:
                self.cfg.set("strategy.params", previous)
                self.strategy = make_strategy(self.cfg)
                self.tg.send(f"↩️ Reverted to previous params:\n<code>{previous}</code>")

        elif command == "kill":
            self.broker.close_all()
            self.paused = True
            self.running = False
            self.stop_reason = "/kill from Telegram"
            self.tg.send("🛑 <b>Kill switch.</b> Everything closed, bot stopping.")

        else:
            self.tg.send(f"Unknown command <code>/{command}</code>. Try /help.")

    def _learning_status(self) -> None:
        if not self.learning:
            self.tg.send("Learning system is disabled (learning.enabled: false).")
            return
        lessons = self.learning.lessons.active()
        champion = self.learning.champion()
        lines = [
            "🎓 <b>Learning status</b>",
            f"Gate mode   <code>{self.learning.gate_mode}</code>",
            f"Confidence  <code>{self.learning.min_confidence:.2f}</code>",
            f"Lessons     <code>{len(lessons)}</code> active",
        ]
        if champion:
            lines.append(
                f"Champion    promoted {str(champion.get('promoted_at', ''))[:10]} "
                f"({champion.get('source')}), promised "
                f"{float(champion.get('oos_expectancy', 0)):+.3f}R OOS"
            )
        else:
            lines.append("Champion    config.yaml defaults (nothing promoted)")
        params = dict(self.cfg.get("strategy.params", {}) or {})
        lines.append(f"Params      <code>{params}</code>")
        for lesson in lessons[:5]:
            lines.append(f"• {lesson.text}")
        self.tg.send("\n".join(lines))

    # ================================================================== learning
    def _apply_learning(self, report) -> None:
        """Apply a weekly learning report. Main thread only — this is the single
        place where learned state actually changes the running bot."""
        assert self.learning is not None
        lines = ["🎓 <b>Weekly learning run</b>"]

        if report.error:
            log.error("learning run failed: %s", report.error)
            self.tg.send(f"🎓⚠️ Weekly learning run failed:\n<code>{report.error[:300]}</code>")
            return

        if report.stats_line:
            lines.append(report.stats_line)

        # 1. Gate calibration -------------------------------------------------
        mode_changed = report.gate_mode != self.learning.gate_mode
        conf_changed = abs(report.gate_min_confidence - self.learning.min_confidence) > 1e-9
        if mode_changed or conf_changed:
            self.learning.save_gate_state(
                report.gate_mode, report.gate_min_confidence, report.gate_reason
            )
            self.ai.min_confidence = report.gate_min_confidence
        if mode_changed:
            icon = "🛡" if report.gate_mode == "enforce" else "👁"
            lines.append(f"{icon} Gate mode → <b>{report.gate_mode}</b>: {report.gate_reason}")
        else:
            lines.append(f"Gate stays <b>{report.gate_mode}</b> ({report.gate_reason})")
        if conf_changed:
            lines.append(
                f"Confidence floor → <b>{report.gate_min_confidence:.2f}</b> "
                f"({report.confidence_reason})"
            )

        # 2. Lessons ----------------------------------------------------------
        if report.lessons_update is not None:
            added = self.learning.lessons.apply(report.lessons_update)
            self.learning.invalidate_lessons_cache()
            active = len(self.learning.lessons.active())
            if added or report.lessons_update.drop_ids:
                lines.append(f"📚 Lessons: +{added} new, {active} active")
                for preview in report.lessons_added_preview[:3]:
                    lines.append(f"  • {preview}")

        # 3. Rollback ---------------------------------------------------------
        if report.rollback:
            previous = self.learning.rollback_params(report.rollback_reason)
            if previous is not None:
                self.cfg.set("strategy.params", previous)
                self.strategy = make_strategy(self.cfg)
                lines.append(f"↩️ <b>Params ROLLED BACK</b>: {report.rollback_reason}")
                log.warning("params rolled back: %s", report.rollback_reason)

        # 4. Promotion ---------------------------------------------------------
        elif report.optimizer is not None:
            outcome = report.optimizer
            if report.optimizer_ran:
                self.learning.stamp_optimizer_run(report.history_bars)
            if outcome.promoted:
                previous = dict(self.cfg.get("strategy.params", {}) or {})
                self.learning.promote_params(
                    outcome.promoted, previous,
                    outcome.promoted_oos_expectancy,
                    outcome.promoted_source, outcome.evidence,
                )
                self.cfg.set("strategy.params", outcome.promoted)
                self.strategy = make_strategy(self.cfg)
                diff = {
                    k: f"{previous.get(k)} → {v}"
                    for k, v in outcome.promoted.items()
                    if previous.get(k) != v
                }
                lines.append(
                    f"🏆 <b>Params promoted</b> ({outcome.promoted_source}): "
                    f"{outcome.evidence}"
                )
                lines.append(f"<code>{diff}</code>")
                lines.append("Send /revert to undo.")
                log.info("params promoted: %s (%s)", diff, outcome.evidence)
            elif outcome.ran:
                lines.append(
                    f"🔬 Optimizer: {outcome.evaluated} candidates — {outcome.evidence}"
                )
            elif outcome.skipped_reason:
                lines.append(f"🔬 Optimizer skipped: {outcome.skipped_reason}")

        if report.analyst_summary:
            lines.append(f"\n<i>{report.analyst_summary}</i>")

        self.tg.send("\n".join(lines))

    # ================================================================== misc
    def _ai_summary(self) -> str:
        if not self.ai.enabled:
            return "disabled"
        if not self.ai.usable:
            return f"unavailable (on_error={self.ai.on_error})"
        return " → ".join(a.provider for a in self.ai.usable)


def _is_improvement(side: Side, current_sl: float, candidate: float) -> bool:
    """Stops only ever move in the direction that reduces risk."""
    if not current_sl:
        return True
    return candidate > current_sl if side is Side.BUY else candidate < current_sl


def _infer_close_reason(pos: Position) -> str:
    """Best guess when the adapter can't tell us why a position went away."""
    if not pos.sl or not pos.tp:
        return "closed"
    if pos.side is Side.BUY:
        if pos.price_current <= pos.sl:
            return "stop loss"
        if pos.price_current >= pos.tp:
            return "take profit"
    else:
        if pos.price_current >= pos.sl:
            return "stop loss"
        if pos.price_current <= pos.tp:
            return "take profit"
    return "closed"


def _parse_hhmm(value: str) -> dtime:
    """"HH:MM" -> naive time, compared against `datetime.utcnow().time()`."""
    hour, _, minute = str(value).partition(":")
    return dtime(int(hour), int(minute or 0))
