"""
tests/test_pipeline.py — smoke + correctness tests (no API, synthetic data).
Run:  PYTHONPATH=. python -m pytest llm_trader/tests/ -v
Or:   PYTHONPATH=. python -m llm_trader.tests.test_pipeline
"""
import pandas as pd

from llm_trader.config import AgentConfig
from llm_trader.data.fetcher import DataFetcher
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.agent.tools import ToolExecutor
from llm_trader.agent.chart_agent import ChartReadingAgent
from llm_trader.decision.risk_filter import RiskFilter
from llm_trader.models import Action, AgentDecision


def _cfg():
    return AgentConfig(use_mock_llm=True, use_synthetic=True)


def test_leakage_window_excludes_as_of():
    """fetch_window must return only bars strictly before as_of."""
    cfg = _cfg()
    fetcher = DataFetcher(cfg)
    as_of = "2024-09-16"
    window = fetcher.fetch_window("AAPL", as_of=as_of)
    cutoff = pd.Timestamp(as_of, tz="UTC")
    assert window.index.max() < cutoff, "leakage: window includes as_of bar or later"
    print("PASS test_leakage_window_excludes_as_of")


def test_indicators_are_shifted():
    """Indicator at last row must equal raw indicator at second-to-last row."""
    cfg = _cfg()
    window = DataFetcher(cfg).fetch_window("MSFT", as_of="2024-09-16")
    eng = IndicatorEngine(cfg)
    enriched = eng.compute(window)
    # Recompute RSI without shift on the same data and compare offset
    import pandas_ta as ta
    raw_rsi = ta.rsi(window["Close"], length=cfg.rsi_period)
    assert abs(enriched["rsi"].iloc[-1] - raw_rsi.iloc[-2]) < 1e-6, "indicators not shifted"
    print("PASS test_indicators_are_shifted")


def test_tools_dispatch():
    cfg = _cfg()
    window = DataFetcher(cfg).fetch_window("NVDA", as_of="2024-09-16")
    ex = ToolExecutor(window, "NVDA", "2024-09-16", cfg)
    import json
    ind = json.loads(ex.dispatch("compute_indicators", {}))
    sr  = json.loads(ex.dispatch("detect_sr_and_obs", {}))
    assert "rsi" in ind and "trend" in ind
    assert "sr_levels" in sr and "order_blocks" in sr
    assert ex.calls == ["compute_indicators", "detect_sr_and_obs"]
    print("PASS test_tools_dispatch")


def test_mock_agent_returns_valid_decision():
    cfg = _cfg()
    window = DataFetcher(cfg).fetch_window("AAPL", as_of="2024-09-16")
    ex = ToolExecutor(window, "AAPL", "2024-09-16", cfg)
    dec = ChartReadingAgent(cfg).analyze("", ex, "AAPL", "2024-09-16")
    assert dec.action in ("BUY", "SELL", "HOLD")
    assert 0.0 <= dec.confidence <= 1.0
    assert 0.0 <= dec.size_pct <= cfg.max_position_pct
    assert dec.tools_used  # mock agent exercises the tools
    print("PASS test_mock_agent_returns_valid_decision")


def test_risk_filter_confidence_gate():
    cfg = _cfg()
    low_conf = AgentDecision(
        ticker="AAPL", as_of="2024-09-16", action="BUY", confidence=0.3,
        size_pct=0.05, stop_loss_pct=0.05, take_profit_pct=0.10,
        trend="uptrend", chart_reading="", rationale="",
    )
    sig = RiskFilter(cfg).filter(low_conf, current_price=100.0)
    assert not sig.approved and sig.action == Action.HOLD
    print("PASS test_risk_filter_confidence_gate")


def test_risk_filter_position_cap():
    cfg = _cfg()
    oversized = AgentDecision(
        ticker="AAPL", as_of="2024-09-16", action="BUY", confidence=0.9,
        size_pct=0.50, stop_loss_pct=0.05, take_profit_pct=0.10,
        trend="uptrend", chart_reading="", rationale="",
    )
    sig = RiskFilter(cfg).filter(oversized, current_price=100.0)
    assert sig.approved and sig.size_pct <= cfg.max_position_pct
    print("PASS test_risk_filter_position_cap")


if __name__ == "__main__":
    test_leakage_window_excludes_as_of()
    test_indicators_are_shifted()
    test_tools_dispatch()
    test_mock_agent_returns_valid_decision()
    test_risk_filter_confidence_gate()
    test_risk_filter_position_cap()
    print("\nAll tests passed.")
