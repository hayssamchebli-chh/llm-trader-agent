"""
chart/renderer.py — UC: Render chart image
Renders a candlestick chart with EMA + Bollinger overlays and returns a
base64 PNG. This image is the agent's primary visual input.
"""
import io
import os
import base64
import logging
from typing import Optional

import pandas as pd

from llm_trader.config import AgentConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ChartRenderer:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config
        os.makedirs(config.chart_dir, exist_ok=True)

    def render(self, df: pd.DataFrame, ticker: str, as_of: str,
               save_path: Optional[str] = None) -> str:
        import mplfinance as mpf
        import matplotlib.pyplot as plt

        window = df.tail(self.cfg.chart_lookback).copy()
        if len(window) < 10:
            raise ValueError(f"Too few bars to render: {len(window)}")
        if window.index.tz is not None:
            window.index = window.index.tz_localize(None)

        adds = []
        for col, colour in [("ema20", "#2196F3"), ("ema50", "#FF9800"), ("ema200", "#E91E63")]:
            if col in window.columns and window[col].notna().any():
                adds.append(mpf.make_addplot(window[col], color=colour, width=1.2))
        for col in ("bb_upper", "bb_lower"):
            if col in window.columns and window[col].notna().any():
                adds.append(mpf.make_addplot(window[col], color="#78909C",
                                             width=0.7, linestyle="dashed"))

        buf = io.BytesIO()
        fig, _ = mpf.plot(window, type="candle", style="charles",
                          title=f"{ticker} — {as_of}", addplot=adds or None,
                          returnfig=True, figsize=(10, 6), tight_layout=True)
        fig.savefig(buf, format="png", dpi=self.cfg.chart_dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        png = buf.read()

        path = save_path or os.path.join(self.cfg.chart_dir, f"{ticker}_{as_of}.png")
        with open(path, "wb") as f:
            f.write(png)
        logger.debug("Chart saved: %s", path)
        return base64.b64encode(png).decode("utf-8")

    @staticmethod
    def encode_file(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
