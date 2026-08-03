"""Orchestrates the learning loop from inside the engine.

Threading contract (deliberately strict):
  * The WEEKLY job runs in a worker thread. It only reads: its own SQLite
    connection, CSV files, and the AI API. It NEVER touches the broker and
    NEVER writes state — it returns a LearningReport.
  * All state changes (lessons table, gate state, parameter promotion) are
    applied by the ENGINE on the main thread via `poll_pending()`.
  * The DAILY job (history caching) runs on the main thread because the MT5
    library is not documented thread-safe. It is a few candle pulls — cheap.

State files (all under data/):
  gate_state.json       {"mode", "min_confidence", ...}   — the calibrated gate
  champion.json         current promoted params + what to restore on rollback
  params_active.yaml    strategy overrides merged over config.yaml at boot
  params_history.jsonl  append-only audit of every promotion / rollback
  learning_state.json   last daily / weekly run timestamps
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.config import Config
from core.journal import Journal
from core.learn import analyst as analyst_mod
from core.learn import calibrator as cal
from core.learn import optimizer as opt
from core.learn.lessons import LessonStore, LessonsUpdate
from core.learn.memory import EpisodeMemory, Evidence
from core.models import Signal

log = logging.getLogger("learn.scheduler")


@dataclass
class LearningReport:
    """Everything a weekly run wants changed. Applied on the main thread."""
    gate_mode: str = "shadow"
    gate_min_confidence: float = 0.55
    gate_reason: str = ""
    confidence_reason: str = ""
    lessons_update: LessonsUpdate | None = None
    lessons_added_preview: list[str] = field(default_factory=list)
    analyst_summary: str = ""
    optimizer: opt.OptimizerOutcome | None = None
    optimizer_ran: bool = False
    history_bars: int = 0
    rollback: bool = False
    rollback_reason: str = ""
    stats_line: str = ""
    error: str = ""


class LearningSystem:
    def __init__(self, cfg: Config, journal: Journal, ai_router, data_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.journal = journal          # main-thread connection only
        self.ai = ai_router
        self.data_dir = data_dir or (cfg.root / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memory = EpisodeMemory(journal, cfg)
        self.lessons = LessonStore(journal, int(cfg.get("learning.analyst.max_lessons", 8)))
        self.audit_fraction = float(cfg.get("learning.gate.audit_fraction", 0.2))

        self._gate_state = self._load_json("gate_state.json") or {
            "mode": "shadow",           # cold start: measure before enforcing
            "min_confidence": float(cfg.get("ai.min_confidence", 0.55)),
            "reason": "cold start — gathering evidence before enforcing",
        }
        self._sched_state = self._load_json("learning_state.json") or {}
        self._lessons_text_cache: str | None = None

        self._lock = threading.Lock()
        self._pending: LearningReport | None = None
        self._worker: threading.Thread | None = None

    # ================================================================ gate state
    @property
    def gate_mode(self) -> str:
        return str(self._gate_state.get("mode", "shadow"))

    @property
    def min_confidence(self) -> float:
        return float(self._gate_state.get("min_confidence", 0.55))

    def stamp_optimizer_run(self, bars: int) -> None:
        """Record the data volume the optimizer last ran on (main thread)."""
        self._sched_state["optimizer_bars"] = int(bars)
        self._save_json("learning_state.json", self._sched_state)

    def save_gate_state(self, mode: str, min_confidence: float, reason: str) -> None:
        self._gate_state = {
            "mode": mode,
            "min_confidence": min_confidence,
            "reason": reason,
            "updated": _now(),
        }
        self._save_json("gate_state.json", self._gate_state)

    # ================================================================ per-signal
    def evidence_for(self, signal: Signal) -> Evidence:
        return self.memory.evidence_for(signal)

    def lessons_block(self) -> str:
        if self._lessons_text_cache is None:
            self._lessons_text_cache = self.lessons.render_block()
        return self._lessons_text_cache

    # ================================================================ params
    def active_param_overrides(self) -> dict[str, Any]:
        """Merged over config.yaml's strategy.params at boot."""
        path = self.data_dir / "params_active.yaml"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return dict(yaml.safe_load(fh) or {})
        except Exception as exc:  # noqa: BLE001 - corrupt file must not stop the bot
            log.error("could not read params_active.yaml (%s) — using config defaults", exc)
            return {}

    def promote_params(
        self, new_params: dict[str, Any], previous: dict[str, Any],
        oos_expectancy: float, source: str, evidence: str,
    ) -> None:
        champion = {
            "params": new_params,
            "previous": previous,
            "promoted_at": _now(),
            "oos_expectancy": oos_expectancy,
            "source": source,
            "evidence": evidence,
        }
        self._save_json("champion.json", champion)
        self._save_yaml("params_active.yaml", new_params)
        self._append_history({"action": "promote", **champion})

    def rollback_params(self, reason: str) -> dict[str, Any] | None:
        """Restore the pre-promotion params. Returns them, or None if nothing to do."""
        champion = self._load_json("champion.json")
        if not champion:
            return None
        previous = dict(champion.get("previous") or {})
        if previous:
            self._save_yaml("params_active.yaml", previous)
        else:
            (self.data_dir / "params_active.yaml").unlink(missing_ok=True)
        self._append_history({"action": "rollback", "restored": previous,
                              "reason": reason, "ts": _now()})
        (self.data_dir / "champion.json").unlink(missing_ok=True)
        return previous

    def champion(self) -> dict[str, Any] | None:
        return self._load_json("champion.json")

    # ================================================================ scheduling
    def maybe_schedule(self, now: datetime, broker, mode: str) -> None:
        """Called every tick from the main thread."""
        self._maybe_daily(now, broker, mode)
        self._maybe_weekly(now)

    def _maybe_daily(self, now: datetime, broker, mode: str) -> None:
        if now.hour != int(self.cfg.get("learning.schedule.daily_hour_utc", 22)):
            return
        today = now.date().isoformat()
        if self._sched_state.get("last_daily") == today:
            return
        self._sched_state["last_daily"] = today
        self._save_json("learning_state.json", self._sched_state)

        # History caching runs on the MAIN thread (MT5 is not thread-safe).
        if mode == "live":
            try:
                self.cache_history(broker)
            except Exception:  # noqa: BLE001
                log.exception("daily history caching failed")

    def _maybe_weekly(self, now: datetime) -> None:
        if now.weekday() != int(self.cfg.get("learning.schedule.weekly_day", 6)):
            return
        if now.hour < int(self.cfg.get("learning.schedule.weekly_hour_utc", 12)):
            return
        week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        if self._sched_state.get("last_weekly") == week:
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._sched_state["last_weekly"] = week
        self._save_json("learning_state.json", self._sched_state)

        current_params = dict(self.cfg.get("strategy.params", {}) or {})
        seed = int(now.isocalendar().week) * 1000 + now.isocalendar().year % 1000
        self._worker = threading.Thread(
            target=self._weekly_job,
            args=(current_params, seed),
            name="learn-weekly",
            daemon=True,
        )
        self._worker.start()
        log.info("weekly learning job started (%s)", week)

    def poll_pending(self) -> LearningReport | None:
        with self._lock:
            report, self._pending = self._pending, None
        return report

    def invalidate_lessons_cache(self) -> None:
        self._lessons_text_cache = None

    # ================================================================ weekly job
    def _weekly_job(self, current_params: dict[str, Any], seed: int) -> None:
        report = LearningReport(
            gate_mode=self.gate_mode,
            gate_min_confidence=self.min_confidence,
        )
        journal: Journal | None = None
        try:
            # Own connection — never share SQLite across threads.
            journal = Journal(self.journal.path)
            report = self._compute_weekly(journal, current_params, seed, report)
        except Exception as exc:  # noqa: BLE001 - a failed run must not kill anything
            log.exception("weekly learning job failed")
            report.error = str(exc)
        finally:
            if journal is not None:
                journal.close()
        with self._lock:
            self._pending = report

    def _compute_weekly(
        self, journal: Journal, current_params: dict[str, Any], seed: int,
        report: LearningReport,
    ) -> LearningReport:
        # 1. Gate calibration ------------------------------------------------
        rows = cal.fetch_gate_rows(journal, days=90)
        assessment = cal.assess(rows, self.gate_mode, self.min_confidence, self.cfg)
        report.gate_mode = assessment.mode
        report.gate_min_confidence = assessment.min_confidence
        report.gate_reason = assessment.mode_reason
        report.confidence_reason = assessment.confidence_reason
        report.stats_line = (
            f"approved {assessment.n_approved} trades avg "
            f"{assessment.approved_avg_r:+.3f}R · measured rejects "
            f"{assessment.n_rejected} avg {assessment.rejected_avg_r:+.3f}R"
        )

        # 2. Analyst ---------------------------------------------------------
        store = LessonStore(journal, int(self.cfg.get("learning.analyst.max_lessons", 8)))
        lessons = store.active()
        dataset = analyst_mod.gather_dataset(
            journal, assessment, lessons, current_params,
            dict(self.cfg.get("learning.optimizer.bounds", {}) or {}),
        )
        result = analyst_mod.run_analyst(self.ai, dataset, self.cfg)
        report.lessons_update = result.lessons_update
        report.analyst_summary = result.summary
        report.lessons_added_preview = [
            str(n.get("text", ""))[:100] for n in result.lessons_update.new
        ]

        # 3. Rollback check BEFORE trying to promote anything new ------------
        champion = self.champion()
        if champion:
            live_r = self._live_r_since(journal, str(champion.get("promoted_at", "")))
            do_rollback, why = opt.should_rollback(
                live_r,
                float(champion.get("oos_expectancy", 0.0)),
                int(self.cfg.get("learning.rollback.min_trades", 20)),
                float(self.cfg.get("learning.rollback.margin_r", 0.05)),
            )
            if do_rollback:
                report.rollback = True
                report.rollback_reason = why
                return report  # revert first; optimize again next week

        # 4. Optimizer -------------------------------------------------------
        # Gate on genuinely new data. Re-drawing fresh random candidates every
        # week against an essentially unchanged window is repeated sampling
        # until noise clears the bar — the exact failure this system exists to
        # avoid. `bars_seen` is reported back and stamped on the main thread.
        min_new = int(self.cfg.get("learning.optimizer.min_new_bars", 400))
        bars_now = self._count_history_bars()
        bars_at_last_run = int(self._sched_state.get("optimizer_bars", 0))
        report.history_bars = bars_now

        if min_new > 0 and bars_now - bars_at_last_run < min_new:
            report.optimizer = opt.OptimizerOutcome(
                skipped_reason=(
                    f"only {bars_now - bars_at_last_run} new bars since the last "
                    f"optimizer run (need {min_new}) — nothing new to learn from"
                )
            )
            return report

        report.optimizer = opt.run_optimizer(
            self.cfg, current_params, result.param_proposals, seed
        )
        report.optimizer_ran = True
        return report

    def _count_history_bars(self) -> int:
        """Total cached CSV bars across symbols — the optimizer's data volume."""
        timeframe = str(self.cfg.get("engine.timeframe", "M15")).upper()
        total = 0
        for sym in self.cfg.enabled_symbols():
            path = self.data_dir / f"{sym['name']}_{timeframe}.csv"
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    total += max(0, sum(1 for _ in fh) - 1)
            except OSError:
                continue
        return total

    @staticmethod
    def _live_r_since(journal: Journal, since_iso: str) -> list[float]:
        if not since_iso:
            return []
        rows = journal.conn.execute(
            """
            SELECT t.pnl, s.risk_amount
              FROM signals s JOIN trades t ON t.ticket = s.ticket
             WHERE s.outcome = 'executed' AND t.close_time IS NOT NULL
               AND s.risk_amount > 0 AND s.ts >= ?
            """,
            (since_iso,),
        ).fetchall()
        return [float(r["pnl"]) / float(r["risk_amount"]) for r in rows]

    # ================================================================ history cache
    def cache_history(self, broker, bars: int = 5000) -> None:
        """Merge fresh broker candles into data/<SYM>_<TF>.csv (live mode).

        This is what lets the optimizer walk-forward on real data without you
        ever exporting anything by hand.
        """
        timeframe = str(self.cfg.get("engine.timeframe", "M15")).upper()
        for sym in self.cfg.enabled_symbols():
            name = sym["name"]
            try:
                candles = broker.candles(name, timeframe, bars)
            except Exception as exc:  # noqa: BLE001
                log.warning("history cache: %s failed: %s", name, exc)
                continue

            path = self.data_dir / f"{name}_{timeframe}.csv"
            existing: dict[str, list] = {}
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        existing[row["time"]] = row

            for candle in candles:
                key = candle.time.strftime("%Y-%m-%d %H:%M:%S")
                existing[key] = {
                    "time": key,
                    "open": candle.open, "high": candle.high,
                    "low": candle.low, "close": candle.close,
                    "volume": candle.volume,
                }

            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["time", "open", "high", "low", "close", "volume"]
                )
                writer.writeheader()
                for key in sorted(existing):
                    writer.writerow(existing[key])
            tmp.replace(path)
            log.info("history cache: %s now has %d bars", path.name, len(existing))

    # ================================================================ files
    def _load_json(self, name: str) -> dict | None:
        path = self.data_dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            log.error("corrupt %s (%s) — ignoring", name, exc)
            return None

    def _save_json(self, name: str, data: dict) -> None:
        path = self.data_dir / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def _save_yaml(self, name: str, data: dict) -> None:
        path = self.data_dir / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _append_history(self, entry: dict) -> None:
        with (self.data_dir / "params_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
