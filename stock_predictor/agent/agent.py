"""OpenAI-powered investment recommendation agent.

Uses LangChain with tool-calling to provide stock investment recommendations
backed by YFinance data, social media sentiment, and an AutoML prediction model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from stock_predictor.agent.tools import (
    scan_trending_stocks_tool,
    social_media_listener_tool,
    stock_predictor_tool,
    yfinance_tool,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert investment analyst specializing in NASDAQ stocks.
Your goal is to identify stocks with high potential returns in the next 3 months.

You have access to the following tools:

1. **yfinance_tool**: Fetches real-time stock data, technical indicators, and
   fundamentals from YFinance. Use this to get detailed financial data on any stock.

2. **social_media_listener_tool**: Gathers social media sentiment from Reddit,
   Finviz news, and StockTwits. Use this to gauge retail investor sentiment,
   trending mentions, and bullish/bearish signals.

3. **stock_predictor_tool**: Runs a trained AutoML model to predict the 3-month
   forward return of a stock. The model was trained on historical YFinance data
   combined with social media sentiment features.

4. **scan_trending_stocks_tool**: Scans currently trending NASDAQ stocks from
   social media and runs the prediction model on all of them to find the best
   candidates.

## Your Process:
1. When asked for recommendations, first use **scan_trending_stocks_tool** to
   identify candidates with high predicted returns.
2. For promising candidates, use **yfinance_tool** to get detailed fundamentals
   and technicals.
3. Use **social_media_listener_tool** to check sentiment and momentum.
4. Use **stock_predictor_tool** to get the model's return prediction.
5. Synthesize all data into a clear recommendation with reasoning.

## Valuation Analysis:
When analyzing a stock, always assess whether it is **undervalued**, **fairly valued**,
or **overvalued** using these fundamentals from yfinance_tool:

1. **P/E Ratio** (trailingPE, forwardPE): Compare to sector median (~15-25 for tech).
   Forward P/E < trailing P/E suggests improving earnings.
2. **PEG Ratio** (pegRatio): PEG < 1.0 = undervalued relative to growth; PEG > 2.0 = expensive.
3. **Price-to-Book** (priceToBook): P/B < 1.0 may signal undervaluation (but check profitability).
4. **Price-to-Sales** (priceToSalesTrailing12Months): P/S < 2 is cheap for most sectors.
5. **EV/EBITDA** (enterpriseToEbitda): EV/EBITDA < 10 is generally attractive; > 20 is expensive.
6. **EV/Revenue** (enterpriseToRevenue): Compare to sector peers; high-growth SaaS can justify > 10x.
7. **Analyst Target Price** (targetMeanPrice): Compare current price to consensus target.
   Current price well below target = potential upside.
8. **Earnings Growth** (earningsGrowth, earningsQuarterlyGrowth): High growth justifies higher multiples.
9. **Profit Margins** (profitMargins, operatingMargins): Expanding margins support higher valuations.
10. **Return on Equity** (returnOnEquity): ROE > 15% indicates efficient capital use.

Synthesize these into a clear **Valuation Verdict**: Undervalued / Fairly Valued / Overvalued,
with a brief explanation citing the specific metrics that support your conclusion.

## Important Guidelines:
- Always disclose that predictions are model-based estimates, not guarantees.
- Consider both quantitative (model, technicals) and qualitative (sentiment) factors.
- Highlight risks alongside potential returns.
- Be honest about the probabilities of achieving high returns in a short timeframe.
- Provide a diversified set of recommendations when possible.
- Always cite specific data points from the tools to support your analysis.

## Output Format:
Provide structured analysis with:
- **Stock**: Ticker and company name
- **Valuation**: Undervalued / Fairly Valued / Overvalued — with key metrics cited
- **Model Predicted Return (3M)**: From the prediction tool
- **Current Price & Technicals**: Key technical indicators
- **Social Sentiment**: Summary of sentiment signals
- **Fundamentals**: Key financial metrics (P/E, PEG, P/B, EV/EBITDA, margins, ROE)
- **Risk Assessment**: Potential downside risks
- **Recommendation**: Buy/Hold/Avoid with reasoning
"""


def create_agent(
    model: str = "gpt-4o",
    temperature: float = 0.1,
    api_key: str | None = None,
) -> tuple:
    """Create the investment recommendation agent.

    Args:
        model: OpenAI model name.
        temperature: Sampling temperature.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.

    Returns:
        Tuple of (llm, tools, system_message).
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    tools = [
        yfinance_tool,
        social_media_listener_tool,
        stock_predictor_tool,
        scan_trending_stocks_tool,
    ]

    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools, SystemMessage(content=SYSTEM_PROMPT)


def run_agent(
    query: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    api_key: str | None = None,
    max_iterations: int = 10,
) -> str:
    """Run the agent on a user query and return the final response.

    This implements a simple ReAct-style loop: the agent calls tools,
    gets results, and continues until it produces a final text response.

    Args:
        query: User's investment question.
        model: OpenAI model name.
        temperature: Sampling temperature.
        api_key: OpenAI API key.
        max_iterations: Maximum tool-calling iterations.

    Returns:
        Agent's final text response.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    llm_with_tools, tools, system_msg = create_agent(model, temperature, api_key)
    tool_map = {t.name: t for t in tools}

    messages = [system_msg, HumanMessage(content=query)]

    for i in range(max_iterations):
        logger.info("Agent iteration %d", i + 1)
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info("Calling tool: %s(%s)", tool_name, tool_args)

            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # If we hit max iterations (or max_iterations=0), return the last AI message
    if max_iterations <= 0:
        return "Agent reached maximum iterations."
    return response.content if response.content else "Agent reached maximum iterations."
