#!/usr/bin/env python3
"""
explain_routing.py — Walkthrough of the Optimization Agent's routing logic.

This script demonstrates how two middleware layers (OptimRoutingMiddleware and
OptimGuardMiddleware) steer the LLM through a fixed 6-step pipeline without
hard-coding control flow.  The LLM remains the decision-maker at every turn;
the middleware only provides *advisory hints* appended to the system prompt.

Run:
    python scripts/explain_routing.py
"""

from __future__ import annotations

import textwrap

from langchain_core.messages import AIMessage, ToolMessage

# ─── Imports from the project ────────────────────────────────────────────────
from optim_agent.routing import (
    OptimRoutingMiddleware,
    _PIPELINE_STEPS,
    _STEP_HINTS,
    _TOOL_TO_STEP,
    _extract_completed_tools,
    _get_current_step,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THE PIPELINE AT A GLANCE
# ═══════════════════════════════════════════════════════════════════════════════

def show_pipeline_overview():
    """Print the 6-step pipeline that every optimization problem follows."""
    print("=" * 72)
    print("SECTION 1: THE PIPELINE AT A GLANCE")
    print("=" * 72)
    print()
    print(
        "Every natural-language optimization problem passes through the same\n"
        "6-step pipeline.  Steps 1a and 1b run in parallel (dual extraction);\n"
        "the rest are sequential.\n"
    )
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  User's NL problem                                     │")
    print("  └────────────────────────┬────────────────────────────────┘")
    print("                           │")
    print("              ┌────────────┴────────────┐")
    print("  Step 1a     │  run_extractor_a        │  Gemini 2.5 Pro")
    print("              │  (parallel)             │")
    print("  Step 1b     │  run_extractor_b        │  Gemini 2.5 Flash")
    print("              └────────────┬────────────┘")
    print("                           │  two independent LPFormulation JSONs")
    print("                           ▼")
    print("  Step 2      ┌─────────────────────────┐")
    print("              │  validate_formulation    │  Deterministic checks")
    print("              └────────────┬────────────┘")
    print("                           │  errors / warnings")
    print("                           ▼")
    print("  Step 3      ┌─────────────────────────┐")
    print("              │  run_critic              │  Diff + Gemini Pro")
    print("              └────────────┬────────────┘    reconciliation")
    print("                           │  single reconciled formulation")
    print("                           ▼")
    print("  Step 4      ┌─────────────────────────┐")
    print("              │  run_adjudicator         │  Gemini Flash")
    print("              └────────────┬────────────┘    consistency gate")
    print("                           │  approved formulation")
    print("                           ▼")
    print("  Step 5      ┌─────────────────────────┐")
    print("              │  solve_formulation       │  PuLP / CBC")
    print("              └────────────┬────────────┘")
    print("                           │  SolverResult")
    print("                           ▼")
    print("  Step 6      ┌─────────────────────────┐")
    print("              │  generate_report         │  Markdown report")
    print("              └────────────┬────────────┘")
    print("                           │")
    print("                           ▼")
    print("              ┌─────────────────────────┐")
    print("              │  Present to user         │")
    print("              └─────────────────────────┘")
    print()
    print("Registered pipeline steps:")
    for step_num, tool_name, description in _PIPELINE_STEPS:
        print(f"    Step {step_num}:  {tool_name:<25s}  {description}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HOW OptimRoutingMiddleware WORKS
# ═══════════════════════════════════════════════════════════════════════════════

def show_routing_mechanism():
    """Explain the core mechanism: scan messages → determine step → inject hint."""
    print("=" * 72)
    print("SECTION 2: HOW OptimRoutingMiddleware WORKS")
    print("=" * 72)
    print()
    print(textwrap.dedent("""\
    The routing middleware implements AgentMiddleware.wrap_model_call().
    This means it intercepts EVERY call from the agent loop to the LLM.

    The algorithm is:

        1.  Scan the message history for AIMessages with tool_calls.
        2.  Collect the set of tool names that have already been invoked.
        3.  Map those tool names to step numbers (via _TOOL_TO_STEP).
        4.  Take the MAX step number — that's the current pipeline position.
        5.  Look up the advisory hint for that step (via _STEP_HINTS).
        6.  Append the hint to the system prompt using append_to_system_message().
        7.  Pass the modified request to the next handler (the LLM call).

    Crucially, the hint is ADVISORY.  The LLM can ignore it.  But in practice,
    Gemini follows the pipeline ordering reliably because the system prompt
    already describes the pipeline, and the hint reinforces it.
    """))

    print("Tool → Step mapping:")
    for tool_name, step in sorted(_TOOL_TO_STEP.items(), key=lambda x: x[1]):
        print(f"    {tool_name:<25s} → Step {step}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIMULATED WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_routing_progression():
    """Walk through a realistic message history and show what hint fires."""
    print("=" * 72)
    print("SECTION 3: SIMULATED WALKTHROUGH")
    print("=" * 72)
    print()
    print(
        "Below we simulate the message history at each stage of the pipeline\n"
        "and show which advisory hint the middleware would inject.\n"
    )

    # Each scenario: (description, list of tool names the LLM has called so far)
    scenarios = [
        (
            "Turn 0: Fresh start — no tools called yet",
            [],
        ),
        (
            "Turn 1: Both extractors have returned",
            ["run_extractor_a", "run_extractor_b"],
        ),
        (
            "Turn 2: Validation completed on both formulations",
            ["run_extractor_a", "run_extractor_b", "validate_formulation"],
        ),
        (
            "Turn 3: Critic has reconciled the two formulations",
            ["run_extractor_a", "run_extractor_b", "validate_formulation",
             "run_critic"],
        ),
        (
            "Turn 4: Adjudicator approved the formulation",
            ["run_extractor_a", "run_extractor_b", "validate_formulation",
             "run_critic", "run_adjudicator"],
        ),
        (
            "Turn 5: Solver has returned the optimal solution",
            ["run_extractor_a", "run_extractor_b", "validate_formulation",
             "run_critic", "run_adjudicator", "solve_formulation"],
        ),
        (
            "Turn 6: Report generated — pipeline complete",
            ["run_extractor_a", "run_extractor_b", "validate_formulation",
             "run_critic", "run_adjudicator", "solve_formulation",
             "generate_report"],
        ),
    ]

    for description, tool_names in scenarios:
        # Build a fake message history with AIMessages that have tool_calls
        messages = []
        for name in tool_names:
            messages.append(AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {}, "id": f"call_{name}"}],
            ))

        completed = _extract_completed_tools(messages)
        current_step = _get_current_step(completed)
        hint = _STEP_HINTS.get(current_step, "(no hint)")

        print(f"  {description}")
        print(f"    Completed tools : {sorted(completed) if completed else '(none)'}")
        print(f"    Current step    : {current_step}")
        print(f"    Injected hint   : {hint}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HOW OptimGuardMiddleware COMPLEMENTS ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def show_guardrails_interaction():
    """Explain how guardrails work alongside routing."""
    print("=" * 72)
    print("SECTION 4: HOW OptimGuardMiddleware COMPLEMENTS ROUTING")
    print("=" * 72)
    print()
    print(textwrap.dedent("""\
    The guardrail middleware runs AFTER the routing middleware in the
    middleware stack.  It provides three safety nets:

    ┌─────────────────────┬──────────────────────────────────────────────┐
    │  Guard              │  What it does                                │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │  Iteration cap (30) │  Counts wrap_model_call invocations.         │
    │                     │  If exceeded, short-circuits with an         │
    │                     │  AIMessage telling the LLM to stop.          │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │  Token budget (400k)│  Sums input_tokens from usage_metadata       │
    │                     │  on each response.  Stops the agent if       │
    │                     │  cumulative prompt tokens exceed the budget.  │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │  Tool call cap (20) │  Counts "billable" tool calls (excludes      │
    │                     │  free_tools like think, read_file).  When     │
    │                     │  exceeded, strips tool_calls from the         │
    │                     │  AIMessage but preserves any text content.    │
    └─────────────────────┴──────────────────────────────────────────────┘

    Additionally, the guardrail has a SYNTHESIS DIRECTIVE:

        After solve_formulation returns, it scans recent ToolMessages.
        If it sees a ToolMessage named "solve_formulation", it appends:

            "[NOTE] The solver has returned results. Call generate_report
             to produce the final report, then present it to the user."

        This nudge works WITH the routing hint (which says "PIPELINE STEP 6:
        Call generate_report...") to ensure the LLM doesn't keep calling
        tools after the problem is already solved.

    Free tools (think, read_file, write_file, write_todos) never count
    against the tool-call cap.  This means reflection and inter-agent
    file I/O don't eat into the analysis budget.
    """))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MIDDLEWARE EXECUTION ORDER
# ═══════════════════════════════════════════════════════════════════════════════

def show_middleware_stack():
    """Show the full middleware stack and execution order."""
    print("=" * 72)
    print("SECTION 5: MIDDLEWARE EXECUTION ORDER")
    print("=" * 72)
    print()
    print(textwrap.dedent("""\
    When create_optim_agent() is called, the middleware list is:

        middleware = [OptimRoutingMiddleware(), OptimGuardMiddleware()]

    These are passed to create_deep_agent(), which APPENDS them after its
    own built-in middleware stack.  The full stack (innermost → outermost):

        1.  TodoListMiddleware        (built-in: to-do tracking)
        2.  MemoryMiddleware          (built-in: loads AGENTS.md into prompt)
        3.  FilesystemMiddleware      (built-in: provides read/write_file tools)
        4.  SubAgentMiddleware        (built-in: provides task() tool)
        5.  SummarizationMiddleware   (built-in: context window management)
        6.  AnthropicPromptCaching    (built-in: prompt caching optimization)
        7.  PatchToolCallsMiddleware  (built-in: fixes malformed tool calls)
        8.  OptimRoutingMiddleware    ← OUR routing hints
        9.  OptimGuardMiddleware      ← OUR resource caps

    On each model call, the request flows DOWN the stack (1 → 9), then
    the response flows BACK UP (9 → 1).  This means:

        - Routing hint is injected LAST before the LLM sees the prompt.
        - Guardrails check the response FIRST after the LLM returns.
        - If guardrails short-circuit (returning an AIMessage with no
          tool_calls), the agent loop terminates gracefully.
    """))

    print("  Request flow:")
    print()
    print("    User message")
    print("        │")
    print("        ▼")
    print("    ┌─ TodoList ─── Memory ─── Filesystem ─── SubAgent ──┐")
    print("    │                                                     │")
    print("    │  ┌─ Summarization ─── PromptCaching ─── Patch ──┐  │")
    print("    │  │                                               │  │")
    print("    │  │  ┌─ OptimRouting ──── OptimGuard ──┐          │  │")
    print("    │  │  │                                  │          │  │")
    print("    │  │  │        ┌─────────────┐           │          │  │")
    print("    │  │  │        │  Gemini LLM │           │          │  │")
    print("    │  │  │        └─────────────┘           │          │  │")
    print("    │  │  │     hint injected ▲  ▼ caps checked         │  │")
    print("    │  │  └─────────────────────────────────┘          │  │")
    print("    │  └───────────────────────────────────────────────┘  │")
    print("    └────────────────────────────────────────────────────┘")
    print("        │")
    print("        ▼")
    print("    Tool execution or final answer")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WHY ADVISORY (NOT HARD-CODED) ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def show_design_rationale():
    """Explain why hints are advisory instead of enforced."""
    print("=" * 72)
    print("SECTION 6: WHY ADVISORY (NOT HARD-CODED) ROUTING")
    print("=" * 72)
    print()
    print(textwrap.dedent("""\
    The routing middleware does NOT enforce the pipeline.  It only suggests.
    This is a deliberate design choice inherited from the STRAP architecture:

    1.  FLEXIBILITY — If the LLM detects an extraction error mid-pipeline,
        it can re-run an extractor without being blocked by rigid control flow.

    2.  ERROR RECOVERY — If a step fails (e.g., JSON parse error), the LLM
        can retry or skip to a fallback rather than crashing the pipeline.

    3.  SIMPLICITY — No state machine, no DAG orchestrator, no conditional
        branching logic.  The entire routing is ~130 lines of Python.

    4.  LLM AS ORCHESTRATOR — The system prompt already describes the
        pipeline in detail.  The advisory hints are reinforcement, not the
        primary source of truth.  This pattern ("prompt + nudge") is more
        robust than ("code forces sequence") because the LLM can adapt.

    5.  COMPOSABILITY — Adding a new pipeline step is trivial: add one
        tuple to _PIPELINE_STEPS and one entry to _STEP_HINTS.  No new
        classes, no graph edges, no state transitions to define.

    The guardrails are the HARD limit.  If the LLM goes off-script and
    burns through 20 tool calls or 400k tokens, the guard terminates it.
    The routing is the SOFT limit — a GPS giving turn-by-turn directions
    that the driver can override.
    """))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COMPARISON WITH STRAP ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def show_strap_comparison():
    """Compare this routing with the original STRAP architecture."""
    print("=" * 72)
    print("SECTION 7: COMPARISON WITH STRAP (DISSOLVE) ROUTING")
    print("=" * 72)
    print()
    print(textwrap.dedent("""\
    The STRAP (DISSOLVE) system uses a different routing pattern because
    its problem is different: STRAP routes BETWEEN specialist subagents,
    while we route WITHIN a single-agent pipeline.

    ┌──────────────────────┬────────────────────────┬────────────────────────┐
    │  Aspect              │  STRAP / DISSOLVE      │  Optim Agent           │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Routing target      │  Which subagent to     │  Which pipeline step   │
    │                      │  delegate to           │  to execute next       │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Classification      │  LLM classifier +      │  Tool-call history     │
    │  method              │  keyword fallback      │  scan (deterministic)  │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Hint content        │  "Delegate to X via    │  "Call tool X next"    │
    │                      │  task(subagent_type=X)" │                        │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Multi-agent         │  Parallel pairs,       │  Step 1 parallel,      │
    │  patterns            │  sequential chains     │  rest sequential       │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Progress tracking   │  Completed subagent    │  Completed tool name   │
    │                      │  names from task()     │  from tool_calls       │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Guardrails          │  SubagentGuardMiddle-  │  OptimGuardMiddleware  │
    │                      │  ware (per-subagent)   │  (per-agent)           │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │  Shared pattern      │  append_to_system_     │  append_to_system_     │
    │                      │  message()             │  message()             │
    └──────────────────────┴────────────────────────┴────────────────────────┘

    Both architectures share the same core insight: inject advisory context
    into the system prompt at each turn, and let the LLM make the final
    decision.  The difference is WHAT gets injected:

    - STRAP:  "This query suits specialist X"     (routing TO an agent)
    - Optim:  "You've done steps 1-3, do step 4"  (routing WITHIN a pipeline)
    """))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    show_pipeline_overview()
    show_routing_mechanism()
    simulate_routing_progression()
    show_guardrails_interaction()
    show_middleware_stack()
    show_design_rationale()
    show_strap_comparison()

    print("=" * 72)
    print("END OF ROUTING WALKTHROUGH")
    print("=" * 72)
