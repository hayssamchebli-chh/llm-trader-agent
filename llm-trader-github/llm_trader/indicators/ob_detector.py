"""
indicators/ob_detector.py — analytical module, part of the
`detect_sr_and_obs` tool exposed to the agent.

Bullish OB : last bearish candle before a strong bullish impulse.
Bearish OB : last bullish candle before a strong bearish impulse.
An OB is discarded once price re-enters its zone (mitigated) or it ages out.
"""
import logging
from typing import List

import pandas as pd

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.models import OrderBlock

logger = logging.getLogger(__name__)


class OrderBlockDetector:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        if len(df) < 5:
            return []

        close, open_ = df["Close"].values, df["Open"].values
        high, low    = df["High"].values, df["Low"].values
        n, thr       = len(df), self.cfg.ob_impulse_pct
        obs: List[OrderBlock] = []

        for i in range(1, n - 1):
            ret_next = (close[i + 1] - close[i]) / close[i]
            if ret_next >= thr and close[i] < open_[i]:
                ob = OrderBlock(float(high[i]), float(low[i]), "bullish", n - 1 - i)
                if self._valid(ob, df, i):
                    obs.append(ob)
            elif ret_next <= -thr and close[i] > open_[i]:
                ob = OrderBlock(float(high[i]), float(low[i]), "bearish", n - 1 - i)
                if self._valid(ob, df, i):
                    obs.append(ob)

        obs = [o for o in obs if o.age_bars <= self.cfg.ob_max_age]
        obs.sort(key=lambda x: x.age_bars)
        return obs

    def _valid(self, ob: OrderBlock, df: pd.DataFrame, ob_idx: int) -> bool:
        fut = df.iloc[ob_idx + 1:]
        if fut.empty:
            return True
        overlap = (fut["Low"] < ob.top) & (fut["High"] > ob.bottom)
        return not overlap.any()
