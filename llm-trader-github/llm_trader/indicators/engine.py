"""
indicators/engine.py — analytical module, exposed to the agent as the
`compute_indicators` tool.

All indicator columns are .shift(1)-ed so the value read at bar T reflects
only information available at bar T-1 (primary leakage guard).
"""
import logging
from typing import Dict, Any

import pandas as pd
import pandas_ta as ta

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.models import QuantSnapshot

logger = logging.getLogger(__name__)


class IndicatorEngine:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config

    # ─────────────────────────────────────────────────────────────────────────
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df enriched with indicator columns (shifted by 1 bar)."""
        df = df.copy()
        c = self.cfg

        for p in c.ema_periods:
            df[f"ema{p}"] = ta.ema(df["Close"], length=p)

        df["rsi"] = ta.rsi(df["Close"], length=c.rsi_period)

        macd = ta.macd(df["Close"], fast=c.macd_fast, slow=c.macd_slow, signal=c.macd_signal)
        if macd is not None:
            df["macd"]        = macd.iloc[:, 0]
            df["macd_hist"]   = macd.iloc[:, 1]
            df["macd_signal"] = macd.iloc[:, 2]

        bb = ta.bbands(df["Close"], length=c.bb_period, std=c.bb_std)
        if bb is not None:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"]   = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
            span = (df["bb_upper"] - df["bb_lower"]).replace(0, float("nan"))
            df["bb_pct"] = (df["Close"] - df["bb_lower"]) / span

        df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=c.atr_period)

        adx = ta.adx(df["High"], df["Low"], df["Close"], length=c.adx_period)
        if adx is not None:
            df["adx"] = adx.iloc[:, 0]

        ind_cols = [x for x in df.columns
                    if x not in ("Open", "High", "Low", "Close", "Volume")]
        df[ind_cols] = df[ind_cols].shift(1)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    def snapshot(self, df: pd.DataFrame, ticker: str, as_of: str) -> QuantSnapshot:
        enriched = self.compute(df)
        row = enriched.iloc[-1]

        def g(col, default=float("nan")):
            return float(row[col]) if col in row.index and pd.notna(row[col]) else default

        close = float(df["Close"].iloc[-1])
        ema20, ema50 = g("ema20"), g("ema50")

        if ema20 > ema50 and close > ema20:
            trend = "uptrend"
        elif ema20 < ema50 and close < ema20:
            trend = "downtrend"
        else:
            trend = "neutral"

        rsi = g("rsi", 50.0)
        momentum = "bullish" if rsi > 60 else "bearish" if rsi < 40 else "neutral"

        return QuantSnapshot(
            ticker=ticker, as_of=as_of, close=close, volume=int(df["Volume"].iloc[-1]),
            ema20=ema20, ema50=ema50, ema200=g("ema200"),
            rsi=rsi, macd=g("macd"), macd_signal=g("macd_signal"), macd_hist=g("macd_hist"),
            bb_upper=g("bb_upper"), bb_lower=g("bb_lower"), bb_pct=g("bb_pct"),
            atr=g("atr"), adx=g("adx"), trend=trend, momentum=momentum,
        )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def to_dict(q: QuantSnapshot) -> Dict[str, Any]:
        """Tool-friendly JSON dict returned to the agent."""
        return {
            "ticker": q.ticker, "as_of": q.as_of,
            "close": round(q.close, 2), "volume": q.volume,
            "ema20": round(q.ema20, 2), "ema50": round(q.ema50, 2),
            "ema200": round(q.ema200, 2), "rsi": round(q.rsi, 1),
            "macd": round(q.macd, 4), "macd_signal": round(q.macd_signal, 4),
            "macd_hist": round(q.macd_hist, 4),
            "bb_upper": round(q.bb_upper, 2), "bb_lower": round(q.bb_lower, 2),
            "bb_pct": round(q.bb_pct * 100, 1) if q.bb_pct == q.bb_pct else None,
            "atr": round(q.atr, 2), "adx": round(q.adx, 1),
            "trend": q.trend, "momentum": q.momentum,
        }
