"""
backtest/evaluator.py — UC: Evaluate & compare
Computes the performance metrics reported in the thesis Experiments chapter
and builds a strategy-vs-baseline comparison table.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from llm_trader.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    label:            str
    total_return_pct: float
    cagr_pct:         float
    sharpe:           float
    sortino:          float
    max_drawdown_pct: float
    win_rate_pct:     float
    profit_factor:    float
    avg_trade_pct:    float
    n_trades:         int
    equity_curve:     Optional[pd.Series] = field(default=None, repr=False)


class Evaluator:
    TRADING_DAYS = 252

    def evaluate(self, result: BacktestResult, label: str = "Strategy") -> Metrics:
        ec, trades = result.equity_curve, result.trades
        m = Metrics(
            label=label,
            total_return_pct=self._total_return(ec) * 100,
            cagr_pct=self._cagr(ec) * 100,
            sharpe=self._sharpe(ec),
            sortino=self._sortino(ec),
            max_drawdown_pct=self._max_dd(ec) * 100,
            win_rate_pct=self._win_rate(trades) * 100,
            profit_factor=self._profit_factor(trades),
            avg_trade_pct=self._avg_trade(trades) * 100,
            n_trades=len(trades),
            equity_curve=ec,
        )
        self._log(m)
        return m

    def buy_and_hold(self, price: pd.Series, capital: float,
                     label: str = "Buy-and-Hold") -> Metrics:
        ec = price / price.iloc[0] * capital
        dummy = type("R", (), {"equity_curve": ec, "trades": []})()
        return self.evaluate(dummy, label=label)

    def comparison_table(self, metrics: List[Metrics]) -> pd.DataFrame:
        rows = {
            "Total return (%)": [m.total_return_pct for m in metrics],
            "CAGR (%)":         [m.cagr_pct for m in metrics],
            "Sharpe ratio":     [m.sharpe for m in metrics],
            "Sortino ratio":    [m.sortino for m in metrics],
            "Max drawdown (%)": [m.max_drawdown_pct for m in metrics],
            "Win rate (%)":     [m.win_rate_pct for m in metrics],
            "Profit factor":    [m.profit_factor for m in metrics],
            "Avg trade (%)":    [m.avg_trade_pct for m in metrics],
            "# Trades":         [m.n_trades for m in metrics],
        }
        return pd.DataFrame(rows, index=[m.label for m in metrics]).T.round(3)

    # ── metrics ──────────────────────────────────────────────────────────────
    def _total_return(self, ec):
        return (ec.iloc[-1] / ec.iloc[0]) - 1.0 if len(ec) > 1 else 0.0

    def _cagr(self, ec):
        if len(ec) < 2:
            return 0.0
        years = len(ec) / self.TRADING_DAYS
        return (ec.iloc[-1] / ec.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0

    def _rets(self, ec):
        return ec.pct_change().dropna()

    def _sharpe(self, ec):
        r = self._rets(ec)
        return float(r.mean() / r.std() * np.sqrt(self.TRADING_DAYS)) if r.std() else 0.0

    def _sortino(self, ec):
        r = self._rets(ec)
        d = r[r < 0]
        return float(r.mean() / d.std() * np.sqrt(self.TRADING_DAYS)) if len(d) and d.std() else 0.0

    def _max_dd(self, ec):
        if len(ec) < 2:
            return 0.0
        dd = (ec - ec.cummax()) / ec.cummax()
        return float(abs(dd.min()))

    def _win_rate(self, trades):
        return sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) if trades else 0.0

    def _profit_factor(self, trades):
        gp = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
        gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
        return gp / gl if gl else (float("inf") if gp > 0 else 1.0)

    def _avg_trade(self, trades):
        return float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0

    def _log(self, m):
        logger.info("%-16s ret=%+.1f%% CAGR=%+.1f%% Sharpe=%.2f MaxDD=%.1f%% "
                    "Win=%.1f%% Trades=%d", m.label, m.total_return_pct, m.cagr_pct,
                    m.sharpe, m.max_drawdown_pct, m.win_rate_pct, m.n_trades)
