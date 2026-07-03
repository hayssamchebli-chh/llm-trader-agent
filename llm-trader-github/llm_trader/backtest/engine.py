"""
backtest/engine.py — UC: Run backtest simulation
Chronological loop driving the single chart-reading agent over a date range.

Per bar:
  1. leakage-safe window  (data strictly before the bar)
  2. render chart         → base64 image
  3. build ToolExecutor   (fixes the window; tools can't see the future)
  4. agent.analyze()      → AgentDecision
  5. risk filter          → TradeSignal
  6. execute / manage position, record equity
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.data.fetcher import DataFetcher
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.chart.renderer import ChartRenderer
from llm_trader.agent.tools import ToolExecutor
from llm_trader.agent.chart_agent import ChartReadingAgent
from llm_trader.decision.risk_filter import RiskFilter
from llm_trader.models import Action, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker:      str
    entry_price: float
    shares:      float
    stop_loss:   float
    take_profit: Optional[float]
    entry_date:  str


@dataclass
class BacktestResult:
    config:       AgentConfig
    ticker:       str
    equity_curve: pd.Series
    signals:      List[TradeSignal] = field(default_factory=list)
    trades:       List[Dict]        = field(default_factory=list)


class BacktestEngine:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg      = config
        self.fetcher  = DataFetcher(config)
        self.ind_eng  = IndicatorEngine(config)
        self.renderer = ChartRenderer(config)
        self.agent    = ChartReadingAgent(config)
        self.risk     = RiskFilter(config)

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, ticker: str, verbose: bool = False) -> BacktestResult:
        logger.info("Backtest %s  %s → %s  (mock=%s, synthetic=%s)",
                    ticker, self.cfg.start_date, self.cfg.end_date,
                    self.cfg.use_mock_llm, self.cfg.use_synthetic)

        full  = self.fetcher.fetch(ticker)
        dates = full.loc[self.cfg.start_date: self.cfg.end_date].index

        portfolio = self.cfg.initial_capital
        position: Optional[Position] = None
        equity: Dict[pd.Timestamp, float] = {}
        signals: List[TradeSignal] = []
        trades:  List[Dict] = []

        for date in dates:
            as_of = date.strftime("%Y-%m-%d")
            close = float(full.loc[date, "Close"])

            # 1) exits
            if position is not None:
                portfolio, position = self._check_exit(position, close, as_of,
                                                        portfolio, trades)

            # 2) new signal
            try:
                signal = self._signal(ticker, as_of, close,
                                      self._drawdown(equity),
                                      1 if position else 0)
            except Exception as exc:
                logger.error("Signal failed %s: %s", as_of, exc)
                signal = None

            # 3) execute
            if signal and signal.approved:
                signals.append(signal)
                portfolio, position = self._execute(signal, position, close,
                                                     as_of, portfolio, trades)
                if verbose:
                    print(f"{as_of}  {signal.action.value:4s} "
                          f"conf={signal.confidence:.2f}  ${portfolio:,.0f}")

            equity[date] = portfolio

        # force-close at end
        if position is not None:
            last = dates[-1].strftime("%Y-%m-%d")
            portfolio = self._close(position, float(full.loc[dates[-1], "Close"]),
                                    last, portfolio, trades, "end_of_test")

        ec = pd.Series(equity, name="portfolio_value")
        logger.info("Done %s  final=%.2f  trades=%d", ticker, portfolio, len(trades))
        return BacktestResult(self.cfg, ticker, ec, signals, trades)

    # ─────────────────────────────────────────────────────────────────────────
    def _signal(self, ticker, as_of, price, dd, open_n) -> Optional[TradeSignal]:
        window = self.fetcher.fetch_window(ticker, as_of=as_of)
        if len(window) < 60:
            return None

        enriched  = self.ind_eng.compute(window)
        image_b64 = self.renderer.render(enriched, ticker, as_of)
        executor  = ToolExecutor(window, ticker, as_of, self.cfg)

        decision = self.agent.analyze(image_b64, executor, ticker, as_of)
        return self.risk.filter(decision, price, dd, open_n)

    # ── position management ──────────────────────────────────────────────────
    def _execute(self, signal, position, close, as_of, portfolio, trades):
        if signal.action == Action.BUY and position is None:
            fill = close * (1 + self.cfg.slippage_pct)
            notional = portfolio * signal.size_pct
            comm = notional * self.cfg.commission_pct
            shares = (notional - comm) / fill
            portfolio -= comm
            position = Position(signal.ticker, fill, shares,
                                signal.stop_loss, signal.take_profit, as_of)
            logger.info("OPEN  %s %.4f @ %.2f", signal.ticker, shares, fill)
        elif signal.action == Action.SELL and position is not None:
            portfolio = self._close(position, close, as_of, portfolio, trades, "signal")
            position = None
        return portfolio, position

    def _check_exit(self, position, close, as_of, portfolio, trades):
        hit_sl = close <= position.stop_loss
        hit_tp = position.take_profit and close >= position.take_profit
        if hit_sl or hit_tp:
            portfolio = self._close(position, close, as_of, portfolio, trades,
                                    "stop_loss" if hit_sl else "take_profit")
            position = None
        return portfolio, position

    def _close(self, position, close, as_of, portfolio, trades, reason) -> float:
        fill = close * (1 - self.cfg.slippage_pct)
        gross = position.shares * fill
        comm = gross * self.cfg.commission_pct
        portfolio += gross - comm
        pnl = (fill - position.entry_price) / position.entry_price
        trades.append({
            "ticker": position.ticker, "entry_date": position.entry_date,
            "exit_date": as_of, "entry_price": position.entry_price,
            "exit_price": fill, "shares": position.shares,
            "pnl_pct": pnl, "reason": reason,
        })
        logger.info("CLOSE %s @ %.2f  PnL=%.2f%%  (%s)",
                    position.ticker, fill, pnl * 100, reason)
        return portfolio

    @staticmethod
    def _drawdown(equity: Dict[pd.Timestamp, float]) -> float:
        if not equity:
            return 0.0
        vals = list(equity.values())
        peak = max(vals)
        return max(0.0, (peak - vals[-1]) / peak) if peak > 0 else 0.0
