#!/usr/bin/env python3
"""Run the full optimization agent pipeline against the furniture workshop problem."""

from __future__ import annotations

import logging
import time

logging.disable(logging.CRITICAL)

from optim_agent.agent import create_optim_agent, _extract_text

FURNITURE_PROBLEM = """\
A furniture workshop produces tables and chairs. Each table yields a profit
of $55 and each chair yields a profit of $45. A table requires 3 hours of
carpentry and 1 hour of painting. A chair requires 2 hours of carpentry and
2 hours of painting. There are 90 carpentry hours and 62 painting hours
available per week. The workshop can only produce whole units. How many
tables and chairs should be produced to maximize weekly profit?
"""

print("=" * 72)
print("FURNITURE WORKSHOP — FULL PIPELINE RUN")
print("=" * 72)
print()
print("Problem:")
print(FURNITURE_PROBLEM)
print("Loading agent...")
t0 = time.time()
agent = create_optim_agent()
print(f"Agent loaded in {time.time() - t0:.1f}s")
print()
print("Running pipeline (extract → validate → critic → adjudicator → solve → report)...")
print("This will make several Gemini API calls — please wait.\n")

t1 = time.time()
result = agent.invoke({"messages": [{"role": "user", "content": FURNITURE_PROBLEM}]})
elapsed = time.time() - t1

# Print every message in the conversation for full transparency
print("=" * 72)
print("FULL MESSAGE TRACE")
print("=" * 72)
for i, msg in enumerate(result["messages"]):
    msg_type = getattr(msg, "type", "unknown")
    name = getattr(msg, "name", "")
    tool_calls = getattr(msg, "tool_calls", None)

    if msg_type == "human":
        print(f"\n--- [{i}] HUMAN ---")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(content[:200] + ("..." if len(content) > 200 else ""))

    elif msg_type == "ai":
        text = _extract_text(msg.content) if msg.content else ""
        print(f"\n--- [{i}] AI ---")
        if text.strip():
            print(text[:500] + ("..." if len(text) > 500 else ""))
        if tool_calls:
            for tc in tool_calls:
                tc_name = tc.get("name", "?")
                tc_args = tc.get("args", {})
                # Truncate long args for readability
                args_preview = {}
                for k, v in tc_args.items():
                    s = str(v)
                    args_preview[k] = s[:120] + "..." if len(s) > 120 else s
                print(f"  TOOL CALL: {tc_name}({args_preview})")

    elif msg_type == "tool":
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"\n--- [{i}] TOOL RESULT: {name} ---")
        print(content[:600] + ("..." if len(content) > 600 else ""))

print()
print("=" * 72)
print("FINAL ANSWER")
print("=" * 72)
# Extract the last AI message
for msg in reversed(result["messages"]):
    if hasattr(msg, "content") and getattr(msg, "type", "") == "ai" and msg.content:
        text = _extract_text(msg.content)
        if text.strip():
            print(text)
            break

print(f"\nTotal pipeline time: {elapsed:.1f}s")
