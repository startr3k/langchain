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
Your goal is to identify stocks with high potential returns (targeting 100%+ in 6 months).

You have access to the following tools:

1. **yfinance_tool**: Fetches real-time stock data, technical indicators, and
   fundamentals from YFinance. Use this to get detailed financial data on any stock.

2. **social_media_listener_tool**: Gathers social media sentiment from Reddit,
   Finviz news, and StockTwits. Use this to gauge retail investor sentiment,
   trending mentions, and bullish/bearish signals.

3. **stock_predictor_tool**: Runs a trained AutoML model to predict the 6-month
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

## Important Guidelines:
- Always disclose that predictions are model-based estimates, not guarantees.
- Consider both quantitative (model, technicals) and qualitative (sentiment) factors.
- Highlight risks alongside potential returns.
- A 100% return in 6 months is extremely ambitious — be honest about probabilities.
- Provide a diversified set of recommendations when possible.
- Always cite specific data points from the tools to support your analysis.

## Output Format:
Provide structured analysis with:
- **Stock**: Ticker and company name
- **Model Predicted Return (6M)**: From the prediction tool
- **Current Price & Technicals**: Key technical indicators
- **Social Sentiment**: Summary of sentiment signals
- **Fundamentals**: Key financial metrics
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

    # If we hit max iterations, return the last AI message
    return response.content if response.content else "Agent reached maximum iterations."
