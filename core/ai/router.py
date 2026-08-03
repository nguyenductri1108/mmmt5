"""Primary → fallback routing between the two advisors.

Failure policy is explicit and configured, not implicit:
  ai.on_error: reject   -> both providers down means no trades (default, safest)
  ai.on_error: approve  -> fall back to the deterministic strategy's own verdict
"""

from __future__ import annotations

import logging
from typing import Any

from core.ai.base import Advisor
from core.ai.claude_advisor import ClaudeAdvisor
from core.ai.openai_advisor import OpenAIAdvisor
from core.models import AIVerdict

log = logging.getLogger("ai.router")


class AdvisorRouter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("ai.enabled", True))
        self.on_error = str(cfg.get("ai.on_error", "reject")).lower()
        self.min_confidence = float(cfg.get("ai.min_confidence", 0.55))
        self.allow_size_reduction = bool(cfg.get("ai.allow_size_reduction", True))

        pool: dict[str, Advisor] = {
            "claude": ClaudeAdvisor(cfg),
            "openai": OpenAIAdvisor(cfg),
        }
        primary = str(cfg.get("ai.primary", "claude")).lower()
        fallback = str(cfg.get("ai.fallback", "none")).lower()

        self.chain: list[Advisor] = []
        for key in (primary, fallback):
            advisor = pool.get(key)
            if advisor is not None and advisor not in self.chain:
                self.chain.append(advisor)

        self.usable = [a for a in self.chain if a.available()]
        if self.enabled and not self.usable:
            log.warning(
                "AI gate is enabled but no provider is usable "
                "(missing SDK or API key). Every signal will be handled by "
                "ai.on_error=%s.",
                self.on_error,
            )
        elif self.usable:
            log.info(
                "AI gate: %s",
                " -> ".join(f"{a.provider}" for a in self.usable),
            )

    def review(self, payload: dict[str, Any], lessons: str = "") -> AIVerdict:
        """Try each provider in order; first success wins."""
        if not self.enabled:
            return AIVerdict(
                decision="approve",
                confidence=1.0,
                reasons=["AI gate disabled in config"],
                provider="disabled",
            )

        errors: list[str] = []
        for advisor in self.usable:
            try:
                verdict = advisor.review(payload, lessons)
            except Exception as exc:  # noqa: BLE001 - any provider failure -> next provider
                log.warning("%s advisor failed: %s", advisor.provider, exc)
                errors.append(f"{advisor.provider}: {exc}")
                continue

            # A confident-enough approval is the only thing that lets a trade through.
            if verdict.approved and verdict.confidence < self.min_confidence:
                verdict.decision = "reject"
                verdict.reasons.insert(
                    0,
                    f"confidence {verdict.confidence:.2f} below floor {self.min_confidence:.2f}",
                )
            if not self.allow_size_reduction:
                verdict.size_multiplier = 1.0
            return verdict

        return self._error_verdict(errors)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 8000,
        effort: str | None = None,
    ) -> dict[str, Any] | None:
        """Failover wrapper for one-shot structured calls (weekly analyst).

        Returns None when every provider fails — callers must treat that as
        "skip this analysis cycle", never as an empty result.
        """
        for advisor in self.usable:
            try:
                return advisor.complete_json(system, user, schema, max_tokens, effort)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s complete_json failed: %s", advisor.provider, exc)
        return None

    def _error_verdict(self, errors: list[str]) -> AIVerdict:
        detail = "; ".join(errors) or "no AI provider configured"
        if self.on_error == "approve":
            log.warning("AI unavailable (%s) — failing open per ai.on_error", detail)
            return AIVerdict(
                decision="approve",
                confidence=self.min_confidence,
                reasons=["AI unavailable; falling back to strategy verdict"],
                provider="fallback",
                error=detail,
            )
        log.error("AI unavailable (%s) — rejecting trade per ai.on_error", detail)
        return AIVerdict(
            decision="reject",
            confidence=0.0,
            reasons=["AI gate unavailable and ai.on_error=reject"],
            provider="fallback",
            error=detail,
        )
