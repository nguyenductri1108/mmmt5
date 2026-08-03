"""Claude advisor (Anthropic SDK).

Uses schema-constrained output so the response is always parseable JSON, and
prompt caching on the system block so repeated calls are cheap.

Model note: `claude-opus-5` is the default. It has thinking on by default —
we leave it on and use `effort: "low"` instead of disabling it, which is both
cheaper and avoids the known failure modes of thinking-off on this model.
For a very high-frequency gate, `claude-haiku-4-5` is the cost-optimised swap
(set it in config.yaml under ai.claude.model).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from core.ai.base import (
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    Advisor,
    parse_verdict,
    render_user_message,
)
from core.models import AIVerdict

log = logging.getLogger("ai.claude")

try:  # pragma: no cover
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


class ClaudeAdvisor(Advisor):
    provider = "claude"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = str(cfg.get("ai.claude.model", "claude-opus-5"))
        self.effort = str(cfg.get("ai.claude.effort", "low"))
        self.max_tokens = int(cfg.get("ai.claude.max_tokens", 4000))
        self.timeout = float(cfg.get("ai.timeout_seconds", 20))
        self._client = None

    def available(self) -> bool:
        if anthropic is None:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) itself.
            self._client = anthropic.Anthropic(max_retries=1)
        return self._client

    def review(self, payload: dict[str, Any], lessons: str = "") -> AIVerdict:
        client = self._get_client().with_options(timeout=self.timeout)
        started = time.perf_counter()

        # Two system blocks: the frozen prompt, then the weekly-refreshed
        # lessons. Each gets its own cache breakpoint, so a lessons update
        # only invalidates the second block, never the big stable prefix.
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if lessons:
            system_blocks.append(
                {
                    "type": "text",
                    "text": lessons,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            messages=[{"role": "user", "content": render_user_message(payload)}],
        )

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.stop_reason == "refusal":
            raise RuntimeError("claude refused the request")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"claude hit max_tokens ({self.max_tokens}) before finishing JSON — "
                "raise ai.claude.max_tokens or lower ai.claude.effort"
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise RuntimeError("claude returned no text block")

        verdict = parse_verdict(text, self.provider, self.model, latency_ms)
        usage = response.usage
        log.debug(
            "claude verdict=%s conf=%.2f in=%d cached=%d out=%d %dms",
            verdict.decision, verdict.confidence,
            usage.input_tokens, getattr(usage, "cache_read_input_tokens", 0) or 0,
            usage.output_tokens, latency_ms,
        )
        return verdict

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 8000,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """Generic schema-constrained call — used by the weekly analyst.

        Long-form analysis, so a much longer timeout than the trade gate.
        """
        import json

        client = self._get_client().with_options(timeout=300.0)
        response = client.messages.create(
            model=str(self.cfg.get("learning.analyst.model", self.model)),
            max_tokens=max_tokens,
            system=system,
            output_config={
                "effort": effort or "high",
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("claude refused the analyst request")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"claude analyst hit max_tokens ({max_tokens})")
        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise RuntimeError("claude analyst returned no text block")
        return json.loads(text)
