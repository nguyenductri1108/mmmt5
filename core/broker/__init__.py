"""Broker adapters. `make_broker()` picks the right one for the run mode."""

from __future__ import annotations

from core.broker.base import Broker

__all__ = ["Broker", "make_broker"]


def make_broker(cfg, mode: str) -> Broker:
    """Build the adapter for `mode` ("live" -> MT5, "paper" -> simulator)."""
    if mode == "live":
        from core.broker.mt5_adapter import MT5Broker  # imported lazily: Windows-only dep

        return MT5Broker(cfg)
    if mode == "paper":
        from core.broker.paper_adapter import PaperBroker

        return PaperBroker(cfg)
    raise ValueError(f"unknown mode: {mode!r} (expected 'live' or 'paper')")
