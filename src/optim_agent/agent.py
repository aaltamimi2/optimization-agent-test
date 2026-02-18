"""Optimization agent: wires model + tools + middleware + system prompt.

Uses deepagents' create_deep_agent() to build a multi-tool orchestrator
that extracts LP/MILP formulations from natural language, validates,
reconciles, solves, and reports results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from deepagents.backends import FilesystemBackend
from deepagents.graph import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from .adjudicator import run_adjudicator
from .critic import run_critic
from .extraction import run_extractor_a, run_extractor_b
from .guardrails import OptimGuardMiddleware
from .reporter import generate_report
from .routing import OptimRoutingMiddleware
from .schemas import LPFormulation, ValidationResult
from .solver import solve_formulation
from .validator import validate_formulation as _validate

_PACKAGE_DIR = Path(__file__).parent


# ── Validation wrapper tool ──────────────────────────────────────────

@tool
def validate_formulation(formulation_json: str) -> str:
    """Validate an LP formulation for structural and mathematical correctness.

    Takes a JSON string representing an LPFormulation and returns validation
    results including any errors (fatal) and warnings (suspicious patterns
    like matching objective/constraint coefficients).
    """
    data = json.loads(formulation_json)
    formulation = LPFormulation(**data)
    result = _validate(formulation)
    return result.model_dump_json(indent=2)


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert optimization agent that converts natural-language
optimization problems into LP/MILP formulations, validates them, and solves
them using PuLP/CBC.

## Pipeline

Follow these steps IN ORDER for every problem:

1. **Extract (parallel)**: Call `run_extractor_a` AND `run_extractor_b`
   with the problem text. These use different Gemini models for diversity.
2. **Validate**: Call `validate_formulation` on each extraction to check
   for structural errors and coefficient-copy warnings.
3. **Critic**: Call `run_critic` with both extractions and the original
   problem to reconcile any disagreements between extractors.
4. **Adjudicator**: Call `run_adjudicator` with the reconciled formulation
   and original problem for a final consistency check.
5. **Solve**: Call `solve_formulation` with the approved formulation.
6. **Report**: Call `generate_report` with the formulation and solver
   result to produce a formatted report.

## Critical Rules

- **Coefficient accuracy is paramount**: Objective coefficients (profit/cost
  per unit) are DIFFERENT from constraint coefficients (resource usage per
  unit). Never confuse them.
- Always run BOTH extractors. Different models catch different errors.
- If validation warns about matching objective/constraint coefficients,
  investigate carefully before proceeding.
- Present the final report to the user in a clean, readable format.

## Problem Types

- **LP** (Linear Programming): Continuous variables, linear objective and
  constraints.
- **MILP** (Mixed Integer LP): Some variables restricted to integers.
- **Binary**: Variables restricted to 0/1.

Common vocabulary mappings:
- "maximize profit" → MAX objective
- "minimize cost" → MIN objective
- "available hours", "capacity", "budget" → constraint RHS
- "requires X hours", "uses X kg" → constraint coefficients
- "earns $X", "costs $X per unit" → objective coefficients
"""


def create_optim_agent(
    model_name: str = os.getenv("OPTIM_MODEL", "google_genai:gemini-2.5-pro"),
):
    """Create and return a compiled optimization deep agent.

    Registers all 7 pipeline tools + validate_formulation, applies
    routing and guardrail middleware.
    """
    model = init_chat_model(model_name)

    # Pipeline tools
    tools = [
        run_extractor_a,
        run_extractor_b,
        validate_formulation,
        run_critic,
        run_adjudicator,
        solve_formulation,
        generate_report,
    ]

    # Middleware: routing (advisory hints) + guardrails (resource caps)
    routing = OptimRoutingMiddleware()
    guardrails = OptimGuardMiddleware(
        max_iterations=30,
        token_budget=400_000,
        max_tool_calls=20,
        free_tools={"think", "read_file", "write_file", "write_todos"},
    )

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        memory=["./AGENTS.md"],
        backend=FilesystemBackend(root_dir=str(_PACKAGE_DIR)),
        middleware=[routing, guardrails],
        name="optim-agent",
    )
    return agent


# ── CLI helper ───────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Extract plain text from an AI message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def main():
    """Interactive CLI for the optimization agent."""
    import logging
    import readline  # noqa: F401
    import time

    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.spinner import Spinner as RichSpinner
    from rich.text import Text

    console = Console(stderr=True)
    logging.disable(logging.CRITICAL)

    console.print()
    console.print(
        Text.assemble(
            ("OPTIM-AGENT", "bold cyan"),
            (" v0.1.0", "dim"),
        )
    )
    console.print("[dim]LP/MILP solver from natural language[/]")
    console.print("[dim]Type [bold]quit[/bold] to exit.[/]\n")

    with Live(
        RichSpinner("dots", text=Text("Loading agent...", style="dim")),
        console=console,
        transient=True,
    ):
        agent = create_optim_agent()

    out = Console()
    history: list = []

    while True:
        try:
            user_input = out.input("[bold]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        history.append({"role": "user", "content": user_input})

        t0 = time.time()
        with Live(
            RichSpinner("dots", text=Text("Thinking...", style="dim")),
            console=console,
            transient=True,
        ):
            try:
                result = agent.invoke({"messages": list(history)})
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]\n")
                history.pop()
                continue
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}\n")
                history.pop()
                continue

        elapsed = time.time() - t0

        answer = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                break

        if answer:
            history.append({"role": "assistant", "content": answer})
            out.print()
            out.print(Markdown(answer))
            console.print(f"\n[dim]({elapsed:.1f}s)[/]\n")
        else:
            console.print("\n[dim]No response.[/]\n")


if __name__ == "__main__":
    main()
