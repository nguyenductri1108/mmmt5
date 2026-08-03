"""Lessons: distilled, evidence-cited rules injected into the AI gate's prompt.

This is the mechanism that makes the *gate* smarter over time. The weekly
analyst reads the journal, writes short rules ("longs taken with ADX < 20
averaged -0.4R over 14 trades — weigh chop harder"), and prunes the ones the
fresh data no longer supports. Lessons the analyst doesn't re-confirm expire.

Stored in the journal DB so they travel with the trading history. Rendered as
a stable text block so Claude's prompt cache only re-writes once a week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from core.journal import Journal

log = logging.getLogger("learn.lessons")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created        TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    text           TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    active         INTEGER NOT NULL DEFAULT 1
);
"""


@dataclass
class Lesson:
    id: int
    text: str
    evidence: str
    created: str
    last_confirmed: str


@dataclass
class LessonsUpdate:
    """Produced by the analyst (worker thread), applied on the main thread."""
    keep_ids: list[int]
    drop_ids: list[int]
    new: list[dict]        # [{"text": ..., "evidence": ...}]


class LessonStore:
    def __init__(self, journal: Journal, max_lessons: int = 8) -> None:
        self.journal = journal
        self.max_lessons = max_lessons
        self.journal.conn.executescript(_SCHEMA)
        self.journal.conn.commit()

    def active(self) -> list[Lesson]:
        rows = self.journal.conn.execute(
            "SELECT * FROM lessons WHERE active = 1 ORDER BY last_confirmed DESC, id DESC"
        ).fetchall()
        return [
            Lesson(
                id=r["id"], text=r["text"], evidence=r["evidence"],
                created=r["created"], last_confirmed=r["last_confirmed"],
            )
            for r in rows
        ]

    def apply(self, update: LessonsUpdate) -> int:
        """Reconcile: confirm keeps, retire drops + unmentioned, insert new."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn = self.journal.conn

        current_ids = {lesson.id for lesson in self.active()}
        keep = set(update.keep_ids) & current_ids
        # Anything active that the analyst neither kept nor is new gets retired —
        # non-reconfirmation IS the expiry mechanism.
        retire = current_ids - keep

        for lesson_id in keep:
            conn.execute("UPDATE lessons SET last_confirmed = ? WHERE id = ?", (now, lesson_id))
        for lesson_id in retire:
            conn.execute("UPDATE lessons SET active = 0 WHERE id = ?", (lesson_id,))

        added = 0
        for item in update.new[: self.max_lessons]:
            text = str(item.get("text", "")).strip()[:280]
            evidence = str(item.get("evidence", "")).strip()[:280]
            if not text:
                continue
            conn.execute(
                "INSERT INTO lessons (created, last_confirmed, text, evidence, active) "
                "VALUES (?, ?, ?, ?, 1)",
                (now, now, text, evidence),
            )
            added += 1

        # Enforce the cap: keep the most recently confirmed.
        conn.execute(
            """
            UPDATE lessons SET active = 0
             WHERE active = 1 AND id NOT IN (
                SELECT id FROM lessons WHERE active = 1
                 ORDER BY last_confirmed DESC, id DESC LIMIT ?
             )
            """,
            (self.max_lessons,),
        )
        conn.commit()
        log.info("lessons: kept %d, retired %d, added %d", len(keep), len(retire), added)
        return added

    def render_block(self) -> str:
        """Prompt block for the gate. Empty string when there are no lessons."""
        lessons = self.active()
        if not lessons:
            return ""
        lines = [
            "LEARNED LESSONS from this system's own trade history. Each is backed",
            "by journalled outcomes and is re-validated weekly — treat them as",
            "strong evidence about THIS strategy on THESE instruments:",
            "",
        ]
        for lesson in lessons:
            lines.append(f"- {lesson.text}  [{lesson.evidence}]")
        return "\n".join(lines)
