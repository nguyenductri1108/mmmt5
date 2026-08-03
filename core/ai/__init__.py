"""AI gate: a second opinion on every signal the strategy produces."""

from __future__ import annotations

from core.ai.base import Advisor, build_review_payload
from core.ai.router import AdvisorRouter

__all__ = ["Advisor", "AdvisorRouter", "build_review_payload", "make_advisor"]


def make_advisor(cfg) -> AdvisorRouter:
    return AdvisorRouter(cfg)
