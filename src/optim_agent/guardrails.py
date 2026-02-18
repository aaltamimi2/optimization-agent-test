"""Resource-cap guardrail middleware for the optimization agent.

Mirrors STRAP's SubagentGuardMiddleware pattern — caps iterations, token
budget, and tool calls. After solve_formulation returns, injects a synthesis
directive.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class OptimGuardMiddleware(AgentMiddleware):
    """Caps iterations (30), token budget (400k), tool calls (20).

    After solve_formulation returns, injects a synthesis directive telling
    the LLM to generate the report and present results.
    """

    def __init__(
        self,
        max_iterations: int = 30,
        token_budget: int = 400_000,
        max_tool_calls: int = 20,
        free_tools: set[str] | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._max_tool_calls = max_tool_calls
        self._free_tools = free_tools or {"think", "read_file", "write_file"}

        # Per-invocation counters
        self._iterations = 0
        self._total_prompt_tokens = 0
        self._total_tool_calls = 0

    def before_agent(self, state, runtime):
        self._iterations = 0
        self._total_prompt_tokens = 0
        self._total_tool_calls = 0

    async def abefore_agent(self, state, runtime):
        self._iterations = 0
        self._total_prompt_tokens = 0
        self._total_tool_calls = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        self._iterations += 1

        if self._iterations > self._max_iterations:
            logger.warning(
                "OptimGuard: iteration limit (%d) reached",
                self._max_iterations,
            )
            return AIMessage(
                content="[LIMIT] Max iterations reached. Present your findings now.",
            )

        # Check if solver has returned — inject synthesis directive
        request = self._inject_post_solve_directive(request)

        response = handler(request)

        # Track tokens
        self._track_tokens(response)
        if self._total_prompt_tokens > self._token_budget:
            logger.warning(
                "OptimGuard: token budget (%d) exceeded at %d",
                self._token_budget,
                self._total_prompt_tokens,
            )
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Present your findings now.",
            )

        return self._enforce_tool_call_limit(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        self._iterations += 1

        if self._iterations > self._max_iterations:
            return AIMessage(
                content="[LIMIT] Max iterations reached. Present your findings now.",
            )

        request = self._inject_post_solve_directive(request)
        response = await handler(request)

        self._track_tokens(response)
        if self._total_prompt_tokens > self._token_budget:
            return AIMessage(
                content="[LIMIT] Token budget exceeded. Present your findings now.",
            )

        return self._enforce_tool_call_limit(response)

    def _track_tokens(self, response) -> None:
        result = getattr(response, "result", None)
        if not result:
            return
        ai_msg = result[0]
        usage = getattr(ai_msg, "usage_metadata", None)
        if usage:
            self._total_prompt_tokens += usage.get("input_tokens", 0)

    def _enforce_tool_call_limit(self, response):
        result = getattr(response, "result", None)
        if not result:
            return response
        ai_msg = result[0]
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if tool_calls:
            billable = [
                tc for tc in tool_calls
                if tc.get("name") not in self._free_tools
            ]
            self._total_tool_calls += len(billable)
        if self._total_tool_calls >= self._max_tool_calls:
            logger.warning(
                "OptimGuard: tool call limit (%d) reached at %d",
                self._max_tool_calls,
                self._total_tool_calls,
            )
            existing_text = getattr(ai_msg, "content", "") or ""
            if isinstance(existing_text, list):
                parts = []
                for item in existing_text:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item["text"])
                existing_text = "\n".join(parts)
            suffix = (
                "\n\n[LIMIT] Tool call budget exhausted. Present your "
                "findings into a clear, complete answer NOW."
            )
            return AIMessage(content=existing_text + suffix)
        return response

    def _inject_post_solve_directive(self, request: ModelRequest) -> ModelRequest:
        """After solve_formulation returns, nudge toward report generation."""
        solver_seen = False
        for msg in reversed(request.messages):
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                if tool_name == "solve_formulation":
                    solver_seen = True
                    break
            elif isinstance(msg, AIMessage):
                break

        if solver_seen and request.system_message is not None:
            directive = (
                "\n\n[NOTE] The solver has returned results. Call "
                "generate_report to produce the final report, then present "
                "it to the user."
            )
            new_system = append_to_system_message(
                request.system_message, directive
            )
            return request.override(system_message=new_system)

        return request
