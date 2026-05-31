"""
agents/sender_agent.py — Mode 1: ICP and value proposition generation.

ReAct loop using Anthropic's native tool use API.
Returns a dict when the agent calls finish().
"""

import json
import os
import anthropic
from typing import Callable
from dotenv import load_dotenv

from tools.definitions import SENDER_TOOLS
from tools.implementations import scrape_page, search_web
from prompts.prompts import SENDER_SYSTEM_PROMPT

load_dotenv()

_client        = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 4096
MAX_ITERATIONS = 10


def _execute_tool(name: str, inputs: dict) -> str:
    if name == "scrape_page":
        return scrape_page(inputs["url"], inputs["goal"])
    elif name == "search_web":
        return search_web(inputs["query"], inputs["goal"])
    return f"Unknown tool: {name}"


async def analyze_sender(url: str) -> dict:
    """
    Run the sender ReAct agent.
    Returns: { value_proposition, icp, pages_researched, iterations }
    """
    if not url.startswith("http"):
        url = "https://" + url

    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze this company and produce a value proposition and ICP: {url}\n\n"
                f"Start with the homepage, then decide what else you need."
            )
        }
    ]

    pages_researched = []
    iterations       = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = _client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SENDER_SYSTEM_PROMPT,
            tools=SENDER_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return {
                "value_proposition": text,
                "icp": {},
                "pages_researched": pages_researched,
                "error": "Agent ended without calling finish()"
            }

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_inputs = block.input
                tool_use_id = block.id

                if tool_name == "finish":
                    return {
                        "value_proposition": tool_inputs.get("value_proposition", ""),
                        "icp":               tool_inputs.get("icp", {}),
                        "pages_researched":  pages_researched,
                        "iterations":        iterations,
                    }

                if tool_name == "scrape_page":
                    pages_researched.append(tool_inputs.get("url", ""))
                
                result = _execute_tool(tool_name, tool_inputs)

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

    return {
        "value_proposition": "Max iterations reached.",
        "icp": {},
        "pages_researched": pages_researched,
        "error": f"Max iterations ({MAX_ITERATIONS}) reached"
    }
