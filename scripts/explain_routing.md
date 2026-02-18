# Optimization Agent — Routing Logic Walkthrough

This document explains how two middleware layers — **`OptimRoutingMiddleware`** and **`OptimGuardMiddleware`** — steer the LLM through a fixed 6-step pipeline **without hard-coding control flow**. The LLM remains the decision-maker at every turn; the middleware only provides *advisory hints* appended to the system prompt.

---

## Table of Contents

1. [The Pipeline at a Glance](#1-the-pipeline-at-a-glance)
2. [How OptimRoutingMiddleware Works](#2-how-optimroutingmiddleware-works)
3. [Simulated Walkthrough](#3-simulated-walkthrough)
4. [How OptimGuardMiddleware Complements Routing](#4-how-optimguardmiddleware-complements-routing)
5. [Middleware Execution Order](#5-middleware-execution-order)
6. [Why Advisory (Not Hard-Coded) Routing](#6-why-advisory-not-hard-coded-routing)
7. [Comparison with STRAP (DISSOLVE) Routing](#7-comparison-with-strap-dissolve-routing)

---

## 1. The Pipeline at a Glance

Every natural-language optimization problem passes through the same 6-step pipeline. Steps 1a and 1b run in **parallel** (dual extraction); the rest are sequential.

```
  ┌─────────────────────────────────────────────────────────┐
  │  User's NL problem                                     │
  └────────────────────────┬────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
  Step 1a     │  run_extractor_a        │  Gemini 2.5 Pro
              │  (parallel)             │
  Step 1b     │  run_extractor_b        │  Gemini 2.5 Flash
              └────────────┬────────────┘
                           │  two independent LPFormulation JSONs
                           ▼
  Step 2      ┌─────────────────────────┐
              │  validate_formulation    │  Deterministic checks
              └────────────┬────────────┘
                           │  errors / warnings
                           ▼
  Step 3      ┌─────────────────────────┐
              │  run_critic              │  Diff + Gemini Pro reconciliation
              └────────────┬────────────┘
                           │  single reconciled formulation
                           ▼
  Step 4      ┌─────────────────────────┐
              │  run_adjudicator         │  Gemini Flash consistency gate
              └────────────┬────────────┘
                           │  approved formulation
                           ▼
  Step 5      ┌─────────────────────────┐
              │  solve_formulation       │  PuLP / CBC
              └────────────┬────────────┘
                           │  SolverResult
                           ▼
  Step 6      ┌─────────────────────────┐
              │  generate_report         │  Markdown report
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Present to user         │
              └─────────────────────────┘
```

### Registered Pipeline Steps

| Step | Tool Name | Description |
|------|-----------|-------------|
| 1 | `run_extractor_a` | Extract formulation with Gemini 2.5 Pro |
| 1 | `run_extractor_b` | Extract formulation with Gemini 2.5 Flash |
| 2 | `validate_formulation` | Run deterministic validation checks |
| 3 | `run_critic` | Diff and reconcile the two extractions |
| 4 | `run_adjudicator` | Final consistency check |
| 5 | `solve_formulation` | Solve with PuLP/CBC |
| 6 | `generate_report` | Generate solution report |

---

## 2. How OptimRoutingMiddleware Works

The routing middleware implements `AgentMiddleware.wrap_model_call()`. This means it **intercepts every call** from the agent loop to the LLM.

### Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│  On every LLM call:                                                 │
│                                                                     │
│  1. Scan message history for AIMessages with tool_calls             │
│  2. Collect the set of tool names already invoked                   │
│  3. Map tool names → step numbers (via _TOOL_TO_STEP)               │
│  4. Take MAX step number = current pipeline position                │
│  5. Look up advisory hint for that step (via _STEP_HINTS)           │
│  6. Append hint to system prompt via append_to_system_message()     │
│  7. Pass modified request to next handler (the actual LLM call)     │
└─────────────────────────────────────────────────────────────────────┘
```

Crucially, the hint is **advisory**. The LLM can ignore it. But in practice, Gemini follows the pipeline ordering reliably because the system prompt already describes the pipeline, and the hint reinforces it.

### Tool-to-Step Mapping

Defined in `routing.py` as `_TOOL_TO_STEP`:

```python
_TOOL_TO_STEP = {
    "run_extractor_a":      1,
    "run_extractor_b":      1,
    "validate_formulation":  2,
    "run_critic":            3,
    "run_adjudicator":       4,
    "solve_formulation":     5,
    "generate_report":       6,
}
```

Both extractors share **step 1** because they run in parallel — completing either one (or both) advances the pipeline to step 1.

### Core Implementation

```python
class OptimRoutingMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):
        request = self._inject_hint(request)
        return handler(request)

    def _inject_hint(self, request):
        completed = _extract_completed_tools(request.messages)
        current_step = _get_current_step(completed)
        hint = _STEP_HINTS.get(current_step)

        if hint and request.system_message is not None:
            advisory = f"\n\n[ADVISORY: {hint}]"
            new_system = append_to_system_message(
                request.system_message, advisory
            )
            return request.override(system_message=new_system)

        return request
```

**Key details:**
- `_extract_completed_tools()` scans **all** `AIMessage` objects in the conversation for `tool_calls` and collects tool names into a set.
- `_get_current_step()` takes the **maximum** step number among completed tools — this handles out-of-order or skipped steps gracefully.
- `append_to_system_message()` is a `deepagents` utility that adds a new content block to the system message without overwriting existing content.

---

## 3. Simulated Walkthrough

Below we trace what the middleware injects at each stage of the pipeline. These are the exact hints the LLM sees appended to its system prompt.

### Turn 0 — Fresh Start (no tools called yet)

| Field | Value |
|-------|-------|
| Completed tools | *(none)* |
| Current step | 0 |
| Injected hint | `PIPELINE STEP 1: Call run_extractor_a AND run_extractor_b in parallel with the problem text to get two independent formulations.` |

### Turn 1 — Both Extractors Have Returned

| Field | Value |
|-------|-------|
| Completed tools | `run_extractor_a`, `run_extractor_b` |
| Current step | 1 |
| Injected hint | `PIPELINE STEP 2: Call validate_formulation on each extracted formulation to check for structural errors.` |

### Turn 2 — Validation Completed

| Field | Value |
|-------|-------|
| Completed tools | `run_extractor_a`, `run_extractor_b`, `validate_formulation` |
| Current step | 2 |
| Injected hint | `PIPELINE STEP 3: Call run_critic with both formulations and the original problem to reconcile any disagreements.` |

### Turn 3 — Critic Has Reconciled

| Field | Value |
|-------|-------|
| Completed tools | `run_extractor_a`, `run_extractor_b`, `validate_formulation`, `run_critic` |
| Current step | 3 |
| Injected hint | `PIPELINE STEP 4: Call run_adjudicator with the reconciled formulation and original problem for a final consistency check.` |

### Turn 4 — Adjudicator Approved

| Field | Value |
|-------|-------|
| Completed tools | `...`, `run_critic`, `run_adjudicator` |
| Current step | 4 |
| Injected hint | `PIPELINE STEP 5: Call solve_formulation with the approved formulation to get the optimal solution.` |

### Turn 5 — Solver Has Returned

| Field | Value |
|-------|-------|
| Completed tools | `...`, `run_adjudicator`, `solve_formulation` |
| Current step | 5 |
| Injected hint | `PIPELINE STEP 6: Call generate_report with the formulation and solver result to produce the final report.` |

### Turn 6 — Report Generated (Pipeline Complete)

| Field | Value |
|-------|-------|
| Completed tools | `...`, `solve_formulation`, `generate_report` |
| Current step | 6 |
| Injected hint | `PIPELINE COMPLETE: All steps done. Present the report to the user.` |

---

## 4. How OptimGuardMiddleware Complements Routing

The guardrail middleware runs **after** the routing middleware in the middleware stack. It provides three safety nets:

### Resource Caps

| Guard | Default | Behavior When Exceeded |
|-------|---------|----------------------|
| **Iteration cap** | 30 | Short-circuits with `AIMessage("Present your findings now.")` — no tool calls, agent loop terminates |
| **Token budget** | 400,000 | Sums `input_tokens` from `usage_metadata` on each response. Same short-circuit behavior |
| **Tool call cap** | 20 | Strips `tool_calls` from the `AIMessage` but **preserves any text content** the LLM already generated |

### Free Tools

The following tools are **excluded** from the tool-call count:

- `think` — reflection/reasoning
- `read_file` — inter-agent file I/O
- `write_file` — inter-agent file I/O
- `write_todos` — task tracking

This means reflection and file operations don't eat into the analysis budget.

### Post-Solve Synthesis Directive

After `solve_formulation` returns, the guardrail scans recent `ToolMessage` objects. If it finds one named `"solve_formulation"`, it appends a directive to the system prompt:

```
[NOTE] The solver has returned results. Call generate_report to produce
the final report, then present it to the user.
```

This **reinforces** the routing hint (which says `"PIPELINE STEP 6: Call generate_report..."`) to ensure the LLM doesn't keep calling tools after the problem is already solved. The two middleware layers provide **overlapping nudges** toward the same action.

### Guardrail Execution Flow

```python
def wrap_model_call(self, request, handler):
    self._iterations += 1

    # 1. Check iteration cap BEFORE calling the LLM
    if self._iterations > self._max_iterations:
        return AIMessage(content="[LIMIT] Max iterations reached...")

    # 2. Inject post-solve directive if applicable
    request = self._inject_post_solve_directive(request)

    # 3. Call the LLM
    response = handler(request)

    # 4. Track tokens AFTER the LLM responds
    self._track_tokens(response)
    if self._total_prompt_tokens > self._token_budget:
        return AIMessage(content="[LIMIT] Token budget exceeded...")

    # 5. Enforce tool-call limit on the response
    return self._enforce_tool_call_limit(response)
```

---

## 5. Middleware Execution Order

When `create_optim_agent()` is called, it passes:

```python
middleware = [OptimRoutingMiddleware(), OptimGuardMiddleware()]
```

These are appended **after** the built-in `deepagents` middleware stack. The full stack (innermost to outermost):

| # | Middleware | Source | Purpose |
|---|-----------|--------|---------|
| 1 | `TodoListMiddleware` | built-in | To-do tracking |
| 2 | `MemoryMiddleware` | built-in | Loads `AGENTS.md` into the system prompt |
| 3 | `FilesystemMiddleware` | built-in | Provides `read_file`/`write_file` tools |
| 4 | `SubAgentMiddleware` | built-in | Provides `task()` tool for subagent spawning |
| 5 | `SummarizationMiddleware` | built-in | Context window management |
| 6 | `AnthropicPromptCachingMiddleware` | built-in | Prompt caching optimization |
| 7 | `PatchToolCallsMiddleware` | built-in | Fixes malformed tool calls |
| 8 | **`OptimRoutingMiddleware`** | **custom** | **Pipeline-step advisory hints** |
| 9 | **`OptimGuardMiddleware`** | **custom** | **Resource caps + synthesis directive** |

### Request/Response Flow

On each model call, the **request** flows down the stack (1 &rarr; 9), then the **response** flows back up (9 &rarr; 1):

```
    User message
        │
        ▼
    ┌─ TodoList ─── Memory ─── Filesystem ─── SubAgent ──┐
    │                                                     │
    │  ┌─ Summarization ─── PromptCaching ─── Patch ──┐  │
    │  │                                               │  │
    │  │  ┌─ OptimRouting ──── OptimGuard ──┐          │  │
    │  │  │                                  │          │  │
    │  │  │        ┌─────────────┐           │          │  │
    │  │  │        │  Gemini LLM │           │          │  │
    │  │  │        └─────────────┘           │          │  │
    │  │  │     hint injected ▲  ▼ caps checked         │  │
    │  │  └─────────────────────────────────┘          │  │
    │  └───────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────┘
        │
        ▼
    Tool execution or final answer
```

**Implications of this ordering:**

- The routing hint is injected **last** before the LLM sees the prompt &mdash; it has the final say on prompt content.
- Guardrails check the response **first** after the LLM returns &mdash; they can short-circuit before any other middleware processes the response.
- If guardrails short-circuit (returning an `AIMessage` with no `tool_calls`), the agent loop terminates gracefully.

---

## 6. Why Advisory (Not Hard-Coded) Routing

The routing middleware does **not** enforce the pipeline. It only suggests. This is a deliberate design choice inherited from the STRAP architecture.

### 1. Flexibility

If the LLM detects an extraction error mid-pipeline, it can re-run an extractor without being blocked by rigid control flow.

### 2. Error Recovery

If a step fails (e.g., JSON parse error from Gemini), the LLM can retry or skip to a fallback rather than crashing the entire pipeline.

### 3. Simplicity

No state machine, no DAG orchestrator, no conditional branching logic. The entire routing middleware is **~130 lines of Python**.

### 4. LLM as Orchestrator

The system prompt already describes the pipeline in detail. The advisory hints are **reinforcement**, not the primary source of truth. This pattern ("prompt + nudge") is more robust than ("code forces sequence") because the LLM can adapt to unexpected situations.

### 5. Composability

Adding a new pipeline step is trivial:
1. Add one tuple to `_PIPELINE_STEPS`
2. Add one entry to `_STEP_HINTS`

No new classes, no graph edges, no state transitions to define.

### Soft vs Hard Limits

| Layer | Type | Analogy |
|-------|------|---------|
| `OptimRoutingMiddleware` | **Soft** limit | GPS giving turn-by-turn directions the driver can override |
| `OptimGuardMiddleware` | **Hard** limit | Fuel gauge that forces a stop when the tank is empty |

The routing is the GPS. The guardrails are the fuel gauge. Together, they keep the agent on track without being brittle.

---

## 7. Comparison with STRAP (DISSOLVE) Routing

The STRAP (DISSOLVE) system uses a different routing pattern because its problem is different: STRAP routes **between specialist subagents**, while we route **within a single-agent pipeline**.

| Aspect | STRAP / DISSOLVE | Optim Agent |
|--------|-----------------|-------------|
| **Routing target** | Which subagent to delegate to | Which pipeline step to execute next |
| **Classification method** | LLM classifier + keyword fallback | Tool-call history scan (deterministic) |
| **Hint content** | `"Delegate to X via task(subagent_type=X)"` | `"Call tool X next"` |
| **Multi-agent patterns** | Parallel pairs, sequential chains | Step 1 parallel, rest sequential |
| **Progress tracking** | Completed subagent names from `task()` calls | Completed tool names from `tool_calls` |
| **Guardrails** | `SubagentGuardMiddleware` (per-subagent budgets) | `OptimGuardMiddleware` (per-agent budgets) |
| **Shared pattern** | `append_to_system_message()` | `append_to_system_message()` |

### Key Architectural Differences

**STRAP** needs an LLM classifier because the routing decision is **semantic** &mdash; "does this query need a safety analyst or a separation engineer?" This requires understanding the user's intent, which is inherently fuzzy.

**Optim Agent** uses a **deterministic** scan because the routing decision is **positional** &mdash; "which pipeline tools have already run?" This is a simple set lookup with no ambiguity.

### Shared Core Insight

Both architectures share the same fundamental approach:

> Inject advisory context into the system prompt at each turn, and let the LLM make the final decision.

The difference is **what** gets injected:

- **STRAP**: *"This query suits specialist X"* &mdash; routing **to** an agent
- **Optim**: *"You've done steps 1-3, do step 4"* &mdash; routing **within** a pipeline

---

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| [`src/optim_agent/routing.py`](../src/optim_agent/routing.py) | 132 | `OptimRoutingMiddleware` &mdash; pipeline-step advisory hints |
| [`src/optim_agent/guardrails.py`](../src/optim_agent/guardrails.py) | 180 | `OptimGuardMiddleware` &mdash; resource caps + synthesis directive |
| [`src/optim_agent/agent.py`](../src/optim_agent/agent.py) | 243 | `create_optim_agent()` &mdash; wires middleware into `create_deep_agent()` |
