"""Pipeline-order routing middleware for the optimization agent.

Before each model call, inspects which tools have been called and injects
an advisory hint about the next pipeline step. The hints are advisory — the
LLM remains in control and can override them.

Pipeline order:
  1. extractors (parallel) → 2. validate → 3. critic → 4. adjudicator
  → 5. solve → 6. report
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest

logger = logging.getLogger(__name__)

# Ordered pipeline steps: (step_number, tool_name, description)
_PIPELINE_STEPS = [
    (1, "run_extractor_a", "Extract formulation with Gemini Pro"),
    (1, "run_extractor_b", "Extract formulation with Gemini Flash"),
    (2, "validate_formulation", "Run deterministic validation checks"),
    (3, "run_critic", "Diff and reconcile the two extractions"),
    (4, "run_adjudicator", "Final consistency check"),
    (5, "solve_formulation", "Solve with PuLP/CBC"),
    (6, "generate_report", "Generate solution report"),
]

# Map tool names to step numbers
_TOOL_TO_STEP = {name: step for step, name, _ in _PIPELINE_STEPS}

# Map step numbers to descriptions of what to do next
_STEP_HINTS = {
    0: (
        "PIPELINE STEP 1: Call run_extractor_a AND run_extractor_b in "
        "parallel with the problem text to get two independent formulations."
    ),
    1: (
        "PIPELINE STEP 2: Call validate_formulation on each extracted "
        "formulation to check for structural errors."
    ),
    2: (
        "PIPELINE STEP 3: Call run_critic with both formulations and the "
        "original problem to reconcile any disagreements."
    ),
    3: (
        "PIPELINE STEP 4: Call run_adjudicator with the reconciled "
        "formulation and original problem for a final consistency check."
    ),
    4: (
        "PIPELINE STEP 5: Call solve_formulation with the approved "
        "formulation to get the optimal solution."
    ),
    5: (
        "PIPELINE STEP 6: Call generate_report with the formulation and "
        "solver result to produce the final report."
    ),
    6: (
        "PIPELINE COMPLETE: All steps done. Present the report to the user."
    ),
}


def _extract_completed_tools(messages: list) -> set[str]:
    """Extract tool names from completed tool calls in message history."""
    completed: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                if name:
                    completed.add(name)
    return completed


def _get_current_step(completed_tools: set[str]) -> int:
    """Determine the highest completed pipeline step."""
    max_step = 0
    for tool_name, step in _TOOL_TO_STEP.items():
        if tool_name in completed_tools:
            max_step = max(max_step, step)
    return max_step


class OptimRoutingMiddleware(AgentMiddleware):
    """Injects pipeline-step advisory hints into the system prompt.

    Before each model call, checks which pipeline tools have been called
    and suggests the next step.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        return await handler(request)

    def _inject_hint(self, request: ModelRequest) -> ModelRequest:
        completed = _extract_completed_tools(request.messages)
        current_step = _get_current_step(completed)
        hint = _STEP_HINTS.get(current_step)

        if hint and request.system_message is not None:
            advisory = f"\n\n[ADVISORY: {hint}]"
            new_system = append_to_system_message(
                request.system_message, advisory
            )
            logger.info(
                "OptimRouting: step=%d, hint=%s", current_step, hint[:60]
            )
            return request.override(system_message=new_system)

        return request
