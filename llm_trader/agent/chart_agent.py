"""
agent/chart_agent.py — UC: Reason & generate signal
The single chart-reading agent (v2 architecture).

This class replaces the former VisionAgent + LLMOrchestrator pair. It:
  1. Receives a rendered candlestick chart (base64 PNG).
  2. Runs a ReAct-style loop: reason → optionally call a tool → observe →
     reason again → emit a final structured trading decision.
  3. Tools (compute_indicators, detect_sr_and_obs) give it precise numeric
     backing for what it reads visually.

Two execution modes:
  - Real LLM   (config.use_mock_llm=False): calls GPT-4o with function calling.
  - Mock agent (config.use_mock_llm=True):  deterministic rule-based decision,
    no API calls — for wiring/ smoke tests and CI.
"""
import json
import logging
import re
from typing import Any, Dict, List

from llm_trader.config import AgentConfig, DEFAULT_CONFIG
from llm_trader.agent.tools import TOOL_SCHEMAS, ToolExecutor
from llm_trader.models import AgentDecision

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an expert technical analyst that reads stock candlestick charts and
issues a single trading decision. You are shown ONE chart image with EMA and
Bollinger Band overlays. Reason like a discretionary chart trader.

Workflow:
1. Read the chart visually: overall trend, momentum, notable candlestick
   structure, position of price relative to the moving averages and bands.
2. Call `compute_indicators` to confirm your read with exact indicator values.
3. Call `detect_sr_and_obs` to get precise support/resistance levels and order
   blocks instead of estimating them by eye.
4. Synthesise the visual read + the tool numbers into ONE decision.

Rules:
- Never request a position size above 10% of the portfolio.
- If the evidence is mixed or weak, choose HOLD — do not force a trade.
- Your confidence must reflect genuine uncertainty.
- Explain your reasoning by citing BOTH what you saw on the chart AND the tool
  numbers you retrieved.

When you are done reasoning, respond with ONLY a JSON object (no markdown,
no prose outside the JSON) using exactly this schema:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "size_pct": 0.0-0.10,
  "stop_loss_pct": 0.02-0.08,
  "take_profit_pct": 0.04-0.20,
  "trend": "uptrend" | "downtrend" | "sideways",
  "chart_reading": "what you saw on the chart, 1-3 sentences",
  "rationale": "why this action, citing chart + indicator + level evidence",
  "key_levels": [floats]
}
"""


class ChartReadingAgent:
    def __init__(self, config: AgentConfig = DEFAULT_CONFIG):
        self.cfg = config
        self._client = None                      # lazy OpenAI client

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_b64: str,
        executor:  ToolExecutor,
        ticker:    str,
        as_of:     str,
    ) -> AgentDecision:
        if self.cfg.use_mock_llm:
            return self._mock_decision(executor, ticker, as_of)
        return self._llm_decision(image_b64, executor, ticker, as_of)

    # ─────────────────────────────────────────────────────────────────────────
    # Real LLM path — ReAct tool-calling loop (OpenAI function calling)
    # ─────────────────────────────────────────────────────────────────────────

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            import os
            base_url = os.environ.get("OPENAI_BASE_URL")  # set for OpenRouter
            api_key  = os.environ.get("OPENAI_API_KEY")
            if base_url:
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self._client = OpenAI(api_key=api_key)
        return self._client

    def _llm_decision(self, image_b64, executor, ticker, as_of) -> AgentDecision:
        client = self._client_lazy()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}",
                                   "detail": "high"}},
                    {"type": "text",
                     "text": (f"Analyse this {ticker} chart as of {as_of}. "
                              f"Use your tools to confirm levels and indicators, "
                              f"then return the JSON decision.")},
                ],
            },
        ]

        for _ in range(self.cfg.agent_max_iters):
            resp = client.chat.completions.create(
                model=self.cfg.agent_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=self.cfg.agent_max_tokens,
                temperature=self.cfg.agent_temperature,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(msg)                       # record the tool request
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = executor.dispatch(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue                                    # loop again with tool results

            # No tool call → this is the final answer
            return self._parse(msg.content, executor, ticker, as_of)

        # Ran out of iterations → force one final JSON-only turn
        messages.append({"role": "user",
                         "content": "Return the JSON decision now, nothing else."})
        resp = client.chat.completions.create(
            model=self.cfg.agent_model, messages=messages,
            max_tokens=self.cfg.agent_max_tokens, temperature=self.cfg.agent_temperature,
            response_format={"type": "json_object"},
        )
        return self._parse(resp.choices[0].message.content, executor, ticker, as_of)

    # ─────────────────────────────────────────────────────────────────────────
    # Response parsing / validation
    # ─────────────────────────────────────────────────────────────────────────

    def _parse(self, content, executor, ticker, as_of) -> AgentDecision:
        content = (content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Agent JSON parse error: %s\n%s", e, content)
            raw = {}

        action = str(raw.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        conf = self._clamp(raw.get("confidence", 0.5), 0.0, 1.0)
        size = self._clamp(raw.get("size_pct", 0.05), 0.0, self.cfg.max_position_pct)
        sl   = self._clamp(raw.get("stop_loss_pct", self.cfg.stop_loss_pct), 0.02, 0.08)
        tp   = self._clamp(raw.get("take_profit_pct", sl * 2), 0.02, 0.20)

        return AgentDecision(
            ticker=ticker, as_of=as_of,
            action=action, confidence=conf, size_pct=size,
            stop_loss_pct=sl, take_profit_pct=tp,
            trend=str(raw.get("trend", "sideways")),
            chart_reading=str(raw.get("chart_reading", "")),
            rationale=str(raw.get("rationale", "")),
            key_levels=[float(x) for x in raw.get("key_levels", []) if _is_num(x)],
            tools_used=list(executor.calls),
            raw=raw,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Mock path — deterministic rule-based agent (no API)
    # ─────────────────────────────────────────────────────────────────────────

    def _mock_decision(self, executor, ticker, as_of) -> AgentDecision:
        """
        Simple, transparent baseline so the pipeline runs without an LLM:
        EMA20>EMA50 & RSI in (50,70) → BUY; EMA20<EMA50 & RSI in (30,50) → SELL;
        else HOLD. Confidence scaled by |RSI-50|.
        """
        # ensure tools are exercised so the audit trail is realistic
        executor.dispatch("compute_indicators", {})
        executor.dispatch("detect_sr_and_obs", {})
        q = executor.indicator_snapshot()

        action, conf = "HOLD", 0.4
        if q.ema20 > q.ema50 and 50 < q.rsi < 70:
            action = "BUY"
            conf = min(0.9, 0.55 + (q.rsi - 50) / 40)
        elif q.ema20 < q.ema50 and 30 < q.rsi < 50:
            action = "SELL"
            conf = min(0.9, 0.55 + (50 - q.rsi) / 40)

        reading = (f"EMA20={q.ema20:.2f} vs EMA50={q.ema50:.2f}, RSI={q.rsi:.1f}, "
                   f"MACD hist={q.macd_hist:.3f}, trend read as {q.trend}.")
        return AgentDecision(
            ticker=ticker, as_of=as_of, action=action, confidence=round(conf, 2),
            size_pct=0.05, stop_loss_pct=self.cfg.stop_loss_pct,
            take_profit_pct=self.cfg.stop_loss_pct * 2,
            trend=q.trend,
            chart_reading=reading,
            rationale=f"[MOCK AGENT] Rule-based decision from EMA cross + RSI. {reading}",
            key_levels=[], tools_used=list(executor.calls),
            raw={"mock": True},
        )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _clamp(v, lo, hi) -> float:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return lo


def _is_num(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False
