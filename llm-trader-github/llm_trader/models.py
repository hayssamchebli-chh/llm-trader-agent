"""
models.py — Shared data contracts for the single-agent pipeline.

Change from v1: SentimentOutput and the separate VisionOutput are gone.
The chart-reading agent now produces one AgentDecision that carries both
its visual reading and its final signal; the Risk Filter turns that into
a TradeSignal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Action(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SRLevel:
    price:    float
    kind:     str    # "support" | "resistance"
    strength: int
    age_bars: int


@dataclass
class OrderBlock:
    top:      float
    bottom:   float
    kind:     str    # "bullish" | "bearish"
    age_bars: int


@dataclass
class QuantSnapshot:
    """Numeric indicators for one bar — produced by the compute_indicators tool."""
    ticker:      str
    as_of:       str
    close:       float
    volume:      int
    ema20:       float
    ema50:       float
    ema200:      float
    rsi:         float
    macd:        float
    macd_signal: float
    macd_hist:   float
    bb_upper:    float
    bb_lower:    float
    bb_pct:      float
    atr:         float
    adx:         float
    trend:       str = "neutral"
    momentum:    str = "neutral"


@dataclass
class AgentDecision:
    """
    Raw decision emitted by the ChartReadingAgent (pre risk-filter).
    Combines the agent's chart reading with its trading call.
    """
    ticker:          str
    as_of:           str
    action:          str            # "BUY" | "SELL" | "HOLD"
    confidence:      float          # 0–1
    size_pct:        float          # requested fraction of portfolio
    stop_loss_pct:   float
    take_profit_pct: float
    # Explainability
    trend:           str            # agent's read of the trend
    chart_reading:   str            # narrative of what the agent saw on the chart
    rationale:       str            # why this action
    key_levels:      List[float] = field(default_factory=list)
    tools_used:      List[str]   = field(default_factory=list)
    raw:             Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Final, risk-checked signal ready for the backtest engine."""
    ticker:      str
    as_of:       str
    action:      Action
    confidence:  float
    size_pct:    float
    stop_loss:   float
    take_profit: Optional[float]
    rationale:   str
    chart_reading: str
    approved:    bool = True
    reject_reason: str = ""
