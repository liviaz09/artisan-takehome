"""
agents/sender_agent.py — Mode 1: ICP and value proposition generation.

ReAct loop using Anthropic's native tool use API.

The loop:
  1. Call Claude with system prompt + tools + initial goal
  2. Claude returns tool_use block → execute tool → append observation → repeat
  3. Claude calls finish() → extract structured output → return

Claude drives all decisions: which pages to fetch, what to search for,
when it has enough evidence. We only execute what it asks for.
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from tools.definitions import SENDER_TOOLS
from tools.implementations import scrape_page, search_web
from prompts.prompts import SENDER_SYSTEM_PROMPT

load_dotenv()

_client      = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL        = "claude-sonnet-4-20250514"
MAX_TOKENS   = 4096
MAX_ITERATIONS = 10


def _execute_tool(name: str, inputs: dict) -> str:
    """Route a tool call to its implementation and return the result string."""
    if name == "scrape_page":
        return scrape_page(inputs["url"], inputs["goal"])
    elif name == "search_web":
        return search_web(inputs["query"], inputs["goal"])
    else:
        return f"Unknown tool: {name}"


async def analyze_sender(url: str) -> dict:
    """
    Run the sender ReAct agent for a given company URL.
    Returns: { value_proposition, icp, pages_researched }
    """
    if not url.startswith("http"):
        url = "https://" + url

    # Initial user message — gives the agent its goal
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

    print(f"[sender_agent] Starting ReAct loop for {url}")

    while iterations < MAX_ITERATIONS:
        iterations += 1
        print(f"[sender_agent] Iteration {iterations}")

        # ── Call Claude ──────────────────────────────────────────────────────
        response = _client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SENDER_SYSTEM_PROMPT,
            tools=SENDER_TOOLS,
            messages=messages,
        )

        # ── Append assistant response to message history ──────────────────────
        messages.append({"role": "assistant", "content": response.content})

        # ── Check stop reason ────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            # Claude decided to stop without calling finish() — shouldn't happen
            # with a well-written system prompt, but handle gracefully
            print(f"[sender_agent] end_turn without finish() — extracting from text")
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

        # ── Process tool calls ────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_inputs = block.input
                tool_use_id = block.id

                print(f"[sender_agent] Tool call: {tool_name}({json.dumps(tool_inputs)[:120]})")

                # ── finish() — extract output and return ──────────────────────
                if tool_name == "finish":
                    print(f"[sender_agent] Agent called finish() after {iterations} iterations")
                    return {
                        "value_proposition": tool_inputs.get("value_proposition", ""),
                        "icp":               tool_inputs.get("icp", {}),
                        "pages_researched":  pages_researched,
                        "iterations":        iterations,
                    }

                # ── Execute scrape_page or search_web ─────────────────────────
                result = _execute_tool(tool_name, tool_inputs)

                if tool_name == "scrape_page":
                    pages_researched.append(tool_inputs.get("url", ""))

                print(f"[sender_agent] Tool result: {result[:200]}...")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     result,
                })

            # Append all tool results in one user message
            messages.append({"role": "user", "content": tool_results})

    # Max iterations reached
    print(f"[sender_agent] Max iterations ({MAX_ITERATIONS}) reached")
    return {
        "value_proposition": "Max iterations reached — insufficient evidence collected.",
        "icp": {},
        "pages_researched": pages_researched,
        "error": f"Max iterations ({MAX_ITERATIONS}) reached without finish()"
    }
