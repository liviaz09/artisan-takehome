"""
agents/target_agent.py — Mode 2: target evaluation and outbound email drafting.

ReAct loop using Anthropic's native tool use API.

The loop:
  1. Call Claude with system prompt + tools + sender ICP + target URL + persona
  2. Claude returns tool_use block → execute tool → append observation → repeat
  3. Claude calls finish() → extract fit result → check threshold
  4. If fit_score >= 50 → dedicated email synthesis call → return full result
  5. If fit_score < 50  → return fit result only, skip emails

Hard threshold rationale:
  The decision to generate emails or not is a business rule, not a
  reasoning task. It belongs in code, not in the agent's system prompt.
  Claude evaluates fit; Python enforces the threshold.
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from tools.definitions import TARGET_TOOLS
from tools.implementations import scrape_page, search_web
from prompts.prompts import TARGET_SYSTEM_PROMPT, EMAIL_GENERATION_PROMPT

load_dotenv()

_client        = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 4096
MAX_ITERATIONS = 12
FIT_THRESHOLD  = 50


def _execute_tool(name: str, inputs: dict) -> str:
    """Route a tool call to its implementation."""
    if name == "scrape_page":
        return scrape_page(inputs["url"], inputs["goal"])
    elif name == "search_web":
        return search_web(inputs["query"], inputs["goal"])
    else:
        return f"Unknown tool: {name}"


def _generate_emails(
    sender_url:      str,
    value_prop:      str,
    target_url:      str,
    role:            str,
    seniority:       str,
    fit_result:      dict,
) -> dict:
    """
    Dedicated synthesis call for email generation.
    Only called if fit_score >= FIT_THRESHOLD.
    Receives the fit result (including evidence_snippets) from the ReAct loop.
    Claude's only job here is writing — all evidence was collected by the agent.
    """
    evidence_snippets = fit_result.get("evidence_snippets", [])
    formatted_evidence = "\n\n".join(
        f"[SOURCE: {s['source_url']}]\n{s['text']}"
        for s in evidence_snippets
    )

    prompt = EMAIL_GENERATION_PROMPT.format(
        sender_url=sender_url,
        value_prop=value_prop,
        target_url=target_url,
        company_profile=json.dumps(fit_result.get("company_profile", {}), indent=2),
        role=role,
        seniority=seniority,
        fit_summary=fit_result.get("fit_summary", ""),
        matched_signals=json.dumps(fit_result.get("matched_signals", []), indent=2),
        evidence_snippets=formatted_evidence,
    )

    response = _client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[target_agent] Email parse error: {e}")
        return {"email_a": {}, "email_b": {}}


async def analyze_target(
    sender_url:  str,
    sender_icp:  dict,
    value_prop:  str,
    target_url:  str,
    role:        str,
    seniority:   str,
) -> dict:
    """
    Run the target ReAct agent.

    Returns one of two shapes depending on fit score:

    Poor fit (score < 50):
      { fit_result, emails_generated: false, reason }

    Good fit (score >= 50):
      { fit_result, email_a, email_b, claim_map, emails_generated: true }
    """
    if not target_url.startswith("http"):
        target_url = "https://" + target_url
    if not sender_url.startswith("http"):
        sender_url = "https://" + sender_url

    # Initial message — gives the agent its full context upfront
    initial_message = (
        f"Research this target company and evaluate how well it fits our ICP.\n\n"
        f"Target company: {target_url}\n"
        f"Recipient persona: {role} ({seniority})\n\n"
        f"Sender value proposition: {value_prop}\n\n"
        f"Sender ICP (use this as your evaluation framework):\n"
        f"{json.dumps(sender_icp, indent=2)}\n\n"
        f"Start with the target's homepage to understand what they do, "
        f"then research whatever signals you need to evaluate fit against the ICP."
    )

    messages = [{"role": "user", "content": initial_message}]

    pages_researched = []
    iterations       = 0

    print(f"[target_agent] Starting ReAct loop for {target_url}")

    while iterations < MAX_ITERATIONS:
        iterations += 1
        print(f"[target_agent] Iteration {iterations}")

        # ── Call Claude ──────────────────────────────────────────────────────
        response = _client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=TARGET_SYSTEM_PROMPT,
            tools=TARGET_TOOLS,
            messages=messages,
        )

        # ── Append assistant response ─────────────────────────────────────────
        messages.append({"role": "assistant", "content": response.content})

        # ── Check stop reason ────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            print(f"[target_agent] end_turn without finish()")
            return {
                "fit_result":       {},
                "emails_generated": False,
                "reason":           "Agent ended without calling finish()",
                "pages_researched": pages_researched,
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

                print(f"[target_agent] Tool call: {tool_name}({json.dumps(tool_inputs)[:120]})")

                # ── finish() — evaluate threshold, conditionally generate emails ──
                if tool_name == "finish":
                    fit_score  = tool_inputs.get("fit_score", 0)
                    fit_result = tool_inputs

                    print(f"[target_agent] finish() called — fit score: {fit_score}")
                    print(f"[target_agent] Completed in {iterations} iterations")

                    # Hard threshold — business rule, not agent reasoning
                    if fit_score < FIT_THRESHOLD:
                        print(f"[target_agent] Score {fit_score} < {FIT_THRESHOLD} — skipping emails")
                        return {
                            "fit_result":       fit_result,
                            "emails_generated": False,
                            "reason":           (
                                f"Target scored {fit_score}/100 — below the {FIT_THRESHOLD} "
                                f"threshold for email generation. "
                                f"{fit_result.get('fit_summary', '')}"
                            ),
                            "pages_researched": pages_researched,
                            "iterations":       iterations,
                        }

                    # Score passes threshold — generate emails
                    print(f"[target_agent] Score {fit_score} >= {FIT_THRESHOLD} — generating emails")
                    email_data = _generate_emails(
                        sender_url=sender_url,
                        value_prop=value_prop,
                        target_url=target_url,
                        role=role,
                        seniority=seniority,
                        fit_result=fit_result,
                    )

                    # Build claim map from email claims
                    claim_map = []
                    for email_key in ["email_a", "email_b"]:
                        email = email_data.get(email_key, {})
                        for claim in email.get("claims", []):
                            claim_map.append({
                                "email":      email_key,
                                "angle":      email.get("angle", email_key),
                                "claim":      claim.get("claim", ""),
                                "source_url": claim.get("source_url", ""),
                                "snippet":    claim.get("snippet", ""),
                            })

                    return {
                        "fit_result":       fit_result,
                        "email_a":          email_data.get("email_a", {}),
                        "email_b":          email_data.get("email_b", {}),
                        "claim_map":        claim_map,
                        "emails_generated": True,
                        "pages_researched": pages_researched,
                        "iterations":       iterations,
                    }

                # ── Execute scrape_page or search_web ─────────────────────────
                result = _execute_tool(tool_name, tool_inputs)

                if tool_name == "scrape_page":
                    pages_researched.append(tool_inputs.get("url", ""))

                print(f"[target_agent] Tool result: {result[:200]}...")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

    print(f"[target_agent] Max iterations ({MAX_ITERATIONS}) reached")
    return {
        "fit_result":       {},
        "emails_generated": False,
        "reason":           f"Max iterations ({MAX_ITERATIONS}) reached",
        "pages_researched": pages_researched,
    }
