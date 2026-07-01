"""
debug_one.py — Run the agent on a SINGLE date and print the full decision.

Useful for inspecting the tool-call trace and the agent's reasoning without
running a whole backtest.

    python debug_one.py --ticker AAPL --date 2024-09-16 --mock --synthetic
"""
import argparse
import logging

from llm_trader.config import AgentConfig
from llm_trader.data.fetcher import DataFetcher
from llm_trader.indicators.engine import IndicatorEngine
from llm_trader.chart.renderer import ChartRenderer
from llm_trader.agent.tools import ToolExecutor
from llm_trader.agent.chart_agent import ChartReadingAgent
from llm_trader.decision.risk_filter import RiskFilter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--date", default="2024-09-16", help="Decision date YYYY-MM-DD")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--model", default="gpt-4o")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = AgentConfig(use_mock_llm=args.mock, use_synthetic=args.synthetic,
                      agent_model=args.model)

    fetcher = DataFetcher(cfg)
    window  = fetcher.fetch_window(args.ticker, as_of=args.date)
    print(f"Window: {len(window)} bars ending {window.index[-1].date()}")

    enriched = IndicatorEngine(cfg).compute(window)
    image    = ChartRenderer(cfg).render(enriched, args.ticker, args.date)
    print(f"Chart rendered ({len(image)} base64 chars)")

    executor = ToolExecutor(window, args.ticker, args.date, cfg)
    agent    = ChartReadingAgent(cfg)
    decision = agent.analyze(image, executor, args.ticker, args.date)

    print("\n─── AGENT DECISION ─────────────────────────────")
    print(f"action        : {decision.action}")
    print(f"confidence    : {decision.confidence}")
    print(f"size_pct      : {decision.size_pct}")
    print(f"trend         : {decision.trend}")
    print(f"tools_used    : {decision.tools_used}")
    print(f"key_levels    : {decision.key_levels}")
    print(f"chart_reading : {decision.chart_reading}")
    print(f"rationale     : {decision.rationale}")

    signal = RiskFilter(cfg).filter(decision, float(window['Close'].iloc[-1]))
    print("\n─── AFTER RISK FILTER ──────────────────────────")
    print(f"action    : {signal.action.value}")
    print(f"approved  : {signal.approved}")
    if not signal.approved:
        print(f"reject    : {signal.reject_reason}")
    else:
        print(f"stop_loss : {signal.stop_loss:.2f}")
        print(f"take_profit: {signal.take_profit}")


if __name__ == "__main__":
    main()
