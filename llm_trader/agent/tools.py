"""
agent/tools.py — The analytical tools the chart-reading agent can call.

Design
------
The agent does NOT get to choose a date or ticker when calling a tool — that
would open a leakage hole. Instead, a ToolExecutor is constructed for ONE
(ticker, as_of) context with the leakage-trimmed window already fixed. The
agent calls tools by name; the executor runs them against that fixed window
and returns JSON. This keeps the temporal guard outside the LLM's control.

Two tools are exposed:
  1. compute_indicators   → EMA/RSI/MACD/BB/ATR/ADX snapshot + trend/momentum
  2. detect_sr_and_obs    → clustered support/resistance levels + order blocks
"""
import json
import logging
from typing import Any, Dict, List

import pandas as pd

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.indicators.sr_detector import SRDetector
from llm_trader.indicators.ob_detector import OrderBlockDetector

logger = logging.getLogger(__name__)


# ── OpenAI-style tool schemas (also readable by Anthropic with light mapping) ──
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compute_indicators",
            "description": (
                "Compute technical indicators for the chart currently under "
                "analysis: EMA(20/50/200), RSI(14), MACD(12/26/9), Bollinger "
                "Bands(20), ATR(14), ADX(14), plus a derived trend and momentum "
                "label. Returns exact numeric values to back up your visual read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you need the indicators (for the audit trail).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_sr_and_obs",
            "description": (
                "Detect horizontal support/resistance levels (via swing-pivot "
                "clustering) and Smart-Money-Concept order blocks for the chart "
                "under analysis. Use this to get precise price levels rather than "
                "estimating them by eye from the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you need the levels (for the audit trail).",
                    }
                },
                "required": [],
            },
        },
    },
]


class ToolExecutor:
    """
    Runs analytical tools against a single fixed (ticker, as_of) window.
    Constructed once per decision; reused across the agent's tool-call rounds.
    """

    def __init__(
        self,
        window:  pd.DataFrame,
        ticker:  str,
        as_of:   str,
        config:  AgentConfig = DEFAULT_CONFIG,
    ):
        self.window = window
        self.ticker = ticker
        self.as_of  = as_of
        self.cfg    = config

        self._ind = IndicatorEngine(config)
        self._sr  = SRDetector(config)
        self._ob  = OrderBlockDetector(config)

        self.current_price = float(window["Close"].iloc[-1])
        self.calls: List[str] = []          # audit trail of tool names used

    # ─────────────────────────────────────────────────────────────────────────
    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """Run a tool by name; always returns a JSON string for the tool message."""
        self.calls.append(name)
        try:
            if name == "compute_indicators":
                return self._compute_indicators()
            if name == "detect_sr_and_obs":
                return self._detect_sr_and_obs()
            return json.dumps({"error": f"unknown tool '{name}'"})
        except Exception as exc:                       # never crash the agent loop
            logger.error("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc)})

    # ── Tool implementations ─────────────────────────────────────────────────
    def _compute_indicators(self) -> str:
        snap = self._ind.snapshot(self.window, self.ticker, self.as_of)
        return json.dumps(self._ind.to_dict(snap))

    def _detect_sr_and_obs(self) -> str:
        levels = self._sr.detect(self.window, self.current_price)
        obs    = self._ob.detect(self.window)
        payload = {
            "current_price": round(self.current_price, 2),
            "sr_levels": [
                {"price": round(l.price, 2), "kind": l.kind,
                 "strength": l.strength, "age_bars": l.age_bars}
                for l in levels[:8]
            ],
            "order_blocks": [
                {"kind": o.kind, "bottom": round(o.bottom, 2),
                 "top": round(o.top, 2), "age_bars": o.age_bars}
                for o in obs[:5]
            ],
        }
        return json.dumps(payload)

    # ── Snapshot for the mock agent (no LLM) ─────────────────────────────────
    def indicator_snapshot(self):
        return self._ind.snapshot(self.window, self.ticker, self.as_of)
