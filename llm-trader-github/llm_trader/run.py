"""
run.py — First-run entry point for the LLM Trader Agent.

Examples
--------
# 1) Offline wiring test — no network, no API key, deterministic:
python run.py --ticker AAPL --mock --synthetic --verbose

# 2) Real data, mock agent (checks yfinance + charts, still no API cost):
python run.py --ticker AAPL --mock

# 3) Full run — real data + real GPT-4o agent (needs OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
python run.py --ticker AAPL --start 2024-06-01 --end 2024-09-30 --verbose

The mock+synthetic combination is the recommended FIRST run: it exercises the
entire pipeline (data → indicators → chart → tools → agent → risk → backtest →
metrics) end to end without any external dependency.
"""
import argparse
import logging
import sys

from llm_trader.config import AgentConfig
from llm_trader.backtest.engine import BacktestEngine
from llm_trader.backtest.evaluator import Evaluator


def main():
    p = argparse.ArgumentParser(description="LLM Trader Agent — backtest runner")
    p.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    p.add_argument("--mock", action="store_true",
                   help="Use deterministic rule-based agent (no API calls)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic OHLCV (no network)")
    p.add_argument("--model", default="gpt-4o", help="Agent model name")
    p.add_argument("--verbose", action="store_true", help="Print each executed signal")
    p.add_argument("--quiet", action="store_true", help="Suppress info logging")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = AgentConfig(
        tickers=[args.ticker],
        use_mock_llm=args.mock,
        use_synthetic=args.synthetic,
        agent_model=args.model,
    )
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end

    # Short default window for a quick first run
    if not args.start and not args.end:
        cfg.start_date, cfg.end_date = "2024-06-01", "2024-09-30"

    print("=" * 66)
    print(f" LLM Trader Agent — run")
    print(f"   ticker      : {args.ticker}")
    print(f"   period      : {cfg.start_date} → {cfg.end_date}")
    print(f"   agent       : {'MOCK (rule-based)' if args.mock else cfg.agent_model}")
    print(f"   data source : {'SYNTHETIC' if args.synthetic else 'yfinance'}")
    print("=" * 66)

    engine = BacktestEngine(cfg)
    try:
        result = engine.run(args.ticker, verbose=args.verbose)
    except Exception as exc:
        print(f"\n[ERROR] Backtest failed: {exc}", file=sys.stderr)
        if not args.mock:
            print("Tip: try a first run with  --mock --synthetic  to test wiring.",
                  file=sys.stderr)
        raise

    ev = Evaluator()
    strat = ev.evaluate(result, label="Chart Agent")

    price = engine.fetcher.fetch(args.ticker).loc[cfg.start_date: cfg.end_date, "Close"]
    bnh = ev.buy_and_hold(price, cfg.initial_capital)

    print("\n" + "=" * 66)
    print(" RESULTS")
    print("=" * 66)
    table = ev.comparison_table([strat, bnh])
    print(table.to_string())
    print(f"\n Trades executed : {len(result.trades)}")
    print(f" Signals approved: {len(result.signals)}")
    if result.signals:
        s = result.signals[0]
        print(f"\n Sample decision ({s.as_of}):")
        print(f"   action     : {s.action.value}  (confidence {s.confidence:.2f})")
        print(f"   chart read : {s.chart_reading[:120]}")
        print(f"   rationale  : {s.rationale[:120]}")
    print("=" * 66)


if __name__ == "__main__":
    main()
