"""Dual extractor tools — same prompt, two different Gemini models.

Both extractors are decorated as @tool so the orchestrator can call them
in parallel. The SCRATCHPAD requirement forces the LLM to reason about
variable meanings before writing JSON.
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import LPFormulation

# ── Shared extraction prompt ──────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert at converting natural-language optimization problems into
precise LP/MILP formulations in JSON.

CRITICAL: Objective coefficients and constraint coefficients are DIFFERENT
numbers from the problem text.  Objective coefficients measure VALUE per unit
(profit, cost, revenue).  Constraint coefficients measure RESOURCE USAGE per
unit (hours, kg, area).  NEVER copy objective coefficients into constraints
or vice versa.

## Instructions

1. **SCRATCHPAD** — Before writing any JSON, think step by step:
   - List every decision variable and what it represents.
   - Identify the objective: what is being maximised/minimised, and what is
     the per-unit value for each variable?
   - Identify each constraint: what resource is limited, what is each
     variable's per-unit consumption, and what is the total available?
   - Double-check: are the objective coefficients different from the
     constraint coefficients?  They almost always are.

2. **JSON OUTPUT** — After the scratchpad, output a single JSON object inside
   a ```json``` fenced code block.  The JSON must match this schema exactly:

```
{
  "problem_name": "<short title>",
  "variables": [
    {"name": "x1", "label": "<what x1 represents>", "var_type": "Integer|Continuous|Binary", "lb": 0, "ub": null}
  ],
  "objective": {
    "sense": "MAX" or "MIN",
    "terms": [{"var_name": "x1", "coefficient": <objective coeff>}],
    "label": "<what the objective measures>"
  },
  "constraints": [
    {
      "name": "c1",
      "terms": [{"var_name": "x1", "coefficient": <resource usage coeff>}],
      "sense": "<=" or ">=" or "=",
      "rhs": <right-hand side value>,
      "label": "<what resource this constraint limits>"
    }
  ],
  "extraction_source": "<model name>",
  "confidence_notes": "<any caveats>"
}
```

Remember: objective term coefficients represent VALUE (profit/cost per unit),
constraint term coefficients represent RESOURCE USAGE (hours/kg per unit).
These are almost always different numbers.
"""


def _extract_json(text: str) -> dict:
    """Extract JSON from a ```json``` fenced block in model response."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fallback: try parsing the whole thing
    return json.loads(text)


def _extract_text(content) -> str:
    """Handle Gemini's list-of-dicts or plain string content."""
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


def _run_extraction(model_name: str, problem_text: str) -> str:
    """Call a Gemini model and return a validated LPFormulation as JSON."""
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "human", "content": problem_text},
    ]
    response = model.invoke(messages)
    raw_text = _extract_text(response.content)
    data = _extract_json(raw_text)
    data["extraction_source"] = model_name
    # Validate via Pydantic
    formulation = LPFormulation(**data)
    return formulation.model_dump_json(indent=2)


@tool
def run_extractor_a(problem_text: str) -> str:
    """Extract LP/MILP formulation using Gemini 2.5 Pro.

    Takes the natural-language problem description and returns a validated
    JSON formulation. Uses a detailed scratchpad prompt to ensure coefficient
    accuracy.
    """
    return _run_extraction("gemini-2.5-pro", problem_text)


@tool
def run_extractor_b(problem_text: str) -> str:
    """Extract LP/MILP formulation using Gemini 2.5 Flash.

    Takes the natural-language problem description and returns a validated
    JSON formulation. Uses a detailed scratchpad prompt to ensure coefficient
    accuracy. Provides a second opinion to catch extraction errors.
    """
    return _run_extraction("gemini-2.5-flash", problem_text)
