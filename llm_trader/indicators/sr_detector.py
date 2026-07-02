"""
indicators/sr_detector.py — analytical module, part of the
`detect_sr_and_obs` tool exposed to the agent.
"""
import logging
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.models import SRLevel

logger = logging.getLogger(__name__)


class SRDetector:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config

    def detect(self, df: pd.DataFrame, current_price: float) -> List[SRLevel]:
        df = df.tail(self.cfg.sr_lookback).copy()
        if len(df) < self.cfg.sr_swing_order * 2 + 1:
            return []
        pivots = self._pivots(df)
        if len(pivots) < 2:
            return []
        n_clusters = min(self.cfg.sr_n_clusters, len(pivots))
        return sorted(self._cluster(pivots, n_clusters, current_price, df),
                      key=lambda x: x.price)

    def _pivots(self, df: pd.DataFrame) -> np.ndarray:
        order = self.cfg.sr_swing_order
        closes = df["Close"].values
        hi = argrelextrema(closes, np.greater_equal, order=order)[0]
        lo = argrelextrema(closes, np.less_equal,    order=order)[0]
        return np.concatenate([df["High"].values[hi], df["Low"].values[lo]])

    def _cluster(self, pivots, n_clusters, current_price, df) -> List[SRLevel]:
        X = pivots.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)

        levels = []
        for label, centre in enumerate(km.cluster_centers_.flatten()):
            members  = pivots[km.labels_ == label]
            kind     = "resistance" if centre > current_price else "support"
            levels.append(SRLevel(
                price=float(centre), kind=kind,
                strength=int(len(members)), age_bars=self._age(centre, df),
            ))
        return levels

    def _age(self, level_price: float, df: pd.DataFrame) -> int:
        band = max((df["High"] - df["Low"]).mean() * 0.5, level_price * 0.005)
        touched = df[(df["High"] >= level_price - band) & (df["Low"] <= level_price + band)]
        if touched.empty:
            return len(df)
        return len(df) - 1 - df.index.get_loc(touched.index[-1])
