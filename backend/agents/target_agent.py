"""
agents/target_agent.py — Mode 2: target evaluation and outbound email drafting.

ReAct loop using Anthropic's native tool use API.
Returns a dict when the agent calls finish().

Hard threshold: fit_score < 50 skips email generation entirely.
"""

import json
import os
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Optional

from tools.definitions import TARGET_TOOLS
from tools.implementations import scrape_page, search_web
from prompts.prompts import TARGET_SYSTEM_PROMPT, EMAIL_GENERATION_PROMPT


# ── Pydantic models ───────────────────────────────────────────────────────────

class MatchedSignal(BaseModel):
    signal:     str
    evidence:   str
    source_url: str

class CompanyProfile(BaseModel):
    name:             str
    industry:         str
    estimated_size:   str
    stage:            str
    recent_triggers:  list[str] = []
    tech_signals:     list[str] = []

class EvidenceSnippet(BaseModel):
    text:       str
    source_url: str

class TargetResult(BaseModel):
    fit_score:         int
    fit_label:         str
    fit_summary:       str
    matched_signals:   list[MatchedSignal]
    gap_signals:       list[str]
    company_profile:   CompanyProfile
    evidence_snippets: list[EvidenceSnippet] = []

load_dotenv()

_client        = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL          = "claude-sonnet-4-20250514"
MAX_TOKENS     = 4096
MAX_ITERATIONS = 12
FIT_THRESHOLD  = 50


def _execute_tool(name: str, inputs: dict) -> str:
    if name == "scrape_page":
        return scrape_page(inputs["url"], inputs["goal"])
    elif name == "search_web":
        return search_web(inputs["query"], inputs["goal"])
    return f"Unknown tool: {name}"


def _generate_emails(
    sender_url:  str,
    value_prop:  str,
    target_url:  str,
    role:        str,
    seniority:   str,
    fit_result:  dict,
) -> dict:
    """Dedicated synthesis call. Only runs if fit_score >= FIT_THRESHOLD."""
    evidence_snippets  = fit_result.get("evidence_snippets", [])
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

    Returns one of two shapes:

    Poor fit (score < 50):
      { fit_result, emails_generated: False, reason }

    Good fit (score >= 50):
      { fit_result, email_a, email_b, claim_map, emails_generated: True }
    """
    if not target_url.startswith("http"):
        target_url = "https://" + target_url
    if not sender_url.startswith("http"):
        sender_url = "https://" + sender_url

    messages = [
        {
            "role": "user",
            "content": (
                f"Research this target company and evaluate how well it fits our ICP.\n\n"
                f"Target company: {target_url}\n"
                f"Recipient persona: {role} ({seniority})\n\n"
                f"Sender value proposition: {value_prop}\n\n"
                f"Sender ICP (use this as your evaluation framework):\n"
                f"{json.dumps(sender_icp, indent=2)}\n\n"
                f"Start with the target's homepage, then research whatever signals you need."
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
            system=TARGET_SYSTEM_PROMPT,
            tools=TARGET_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return {
                "fit_result":       {},
                "emails_generated": False,
                "reason":           "Agent ended without calling finish()",
                "pages_researched": pages_researched,
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
                    # Validate finish() output with Pydantic
                    try:
                        fit_result = TargetResult(**tool_inputs)
                    except ValidationError as e:
                        print(f"[target_agent] finish() validation error: {e}")
                        fit_result = TargetResult(
                            fit_score=tool_inputs.get("fit_score", 0),
                            fit_label=tool_inputs.get("fit_label", "Unknown"),
                            fit_summary=tool_inputs.get("fit_summary", ""),
                            matched_signals=[],
                            gap_signals=[],
                            company_profile=CompanyProfile(
                                name="", industry="", estimated_size="", stage=""
                            ),
                        )

                    fit_score = fit_result.fit_score

                    # Hard threshold — business rule in code, not in the agent
                    if fit_score < FIT_THRESHOLD:
                        return {
                            "fit_result":       fit_result.model_dump(),
                            "emails_generated": False,
                            "reason": (
                                f"Target scored {fit_score}/100 — below the "
                                f"{FIT_THRESHOLD} threshold for email generation. "
                                f"{fit_result.fit_summary}"
                            ),
                            "pages_researched": pages_researched,
                            "iterations":       iterations,
                        }

                    email_data = _generate_emails(
                        sender_url=sender_url,
                        value_prop=value_prop,
                        target_url=target_url,
                        role=role,
                        seniority=seniority,
                        fit_result=fit_result.model_dump(),
                    )

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
                        "fit_result":       fit_result.model_dump(),
                        "email_a":          email_data.get("email_a", {}),
                        "email_b":          email_data.get("email_b", {}),
                        "claim_map":        claim_map,
                        "emails_generated": True,
                        "pages_researched": pages_researched,
                        "iterations":       iterations,
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
        "fit_result":       {},
        "emails_generated": False,
        "reason":           f"Max iterations ({MAX_ITERATIONS}) reached",
        "pages_researched": pages_researched,
    }
