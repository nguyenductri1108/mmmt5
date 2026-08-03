"""OpenAI / Codex advisor.

Same prompt, same schema, same clamped verdict as the Claude advisor — so the
two are genuinely interchangeable and the router can fail over between them.

Set `ai.openai.model` in config.yaml to a model your account can actually call.
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

log = logging.getLogger("ai.openai")

try:  # pragma: no cover
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class OpenAIAdvisor(Advisor):
    provider = "openai"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = str(cfg.get("ai.openai.model", "gpt-5"))
        self.max_tokens = int(cfg.get("ai.openai.max_tokens", 4000))
        self.timeout = float(cfg.get("ai.timeout_seconds", 20))
        self._client = None
        self._token_param: str | None = "max_completion_tokens"

    def available(self) -> bool:
        if OpenAI is None:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(max_retries=1, timeout=self.timeout)
        return self._client

    def review(self, payload: dict[str, Any], lessons: str = "") -> AIVerdict:
        client = self._get_client()
        started = time.perf_counter()

        system = SYSTEM_PROMPT + (f"\n\n{lessons}" if lessons else "")
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": render_user_message(payload)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trade_verdict",
                    "strict": True,
                    "schema": VERDICT_SCHEMA,
                },
            },
        )
        if self._token_param:
            kwargs[self._token_param] = self.max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - we only special-case one shape
            # Older/newer models disagree on max_tokens vs max_completion_tokens.
            # Drop the param once and retry rather than failing the trade over it.
            if self._token_param and "max_completion_tokens" in str(exc):
                log.warning("model %s rejected max_completion_tokens; retrying without", self.model)
                self._token_param = None
                kwargs.pop("max_completion_tokens", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        latency_ms = int((time.perf_counter() - started) * 1000)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"openai hit the token limit ({self.max_tokens}) before finishing JSON"
            )
        content = choice.message.content
        if not content:
            raise RuntimeError(f"openai returned empty content (finish={choice.finish_reason})")

        verdict = parse_verdict(content, self.provider, self.model, latency_ms)
        log.debug(
            "openai verdict=%s conf=%.2f %dms", verdict.decision, verdict.confidence, latency_ms
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
        """Generic schema-constrained call — used by the weekly analyst."""
        import json

        client = self._get_client().with_options(timeout=300.0)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # strict=False: analyst schemas use optional keys (partial param
            # proposals), which OpenAI strict mode rejects. Non-strict still
            # guides the output shape; parse-side clamping does the enforcement.
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "analysis", "strict": False, "schema": schema},
            },
        )
        if self._token_param:
            kwargs[self._token_param] = max_tokens
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if not choice.message.content:
            raise RuntimeError(f"openai analyst returned empty content ({choice.finish_reason})")
        return json.loads(choice.message.content)
