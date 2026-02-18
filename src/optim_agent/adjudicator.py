"""Adjudicator: lightweight final consistency pass before solving.

Uses Gemini 2.5 Flash to verify the formulation is internally consistent
and ready for the solver.
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import LPFormulation

_ADJUDICATOR_SYSTEM_PROMPT = """\
You are a final-gate reviewer for LP/MILP formulations. Given:
1. The original natural-language problem
2. A JSON formulation about to be sent to the solver

Check ALL of the following:
- Can the objective value be computed manually from the variable values?
  (e.g., if x1=14, x2=24, does 55*14 + 45*24 = 1850?)
- Are all constraints evaluable with the given variable types?
- Are variable domains (Integer/Continuous/Binary) explicit and correct?
- Do the coefficients in the objective match the VALUE terms in the problem?
- Do the coefficients in the constraints match the RESOURCE USAGE terms?

Respond with JSON inside a ```json``` code block:
```json
{
  "approved": true/false,
  "issues": ["list of issues if any"],
  "formulation": <the formulation JSON, possibly with corrections>
}
```

If approved, return the formulation unchanged.
If not approved, fix the issues and return the corrected formulation.
"""


def _extract_text(content) -> str:
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


def _extract_json(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


@tool
def run_adjudicator(formulation_json: str, original_problem: str) -> str:
    """Final consistency check on an LP formulation before solving.

    Uses Gemini 2.5 Flash to verify the formulation is internally consistent.
    Returns the (possibly corrected) formulation JSON if approved, or an
    error message if issues remain.
    """
    user_msg = (
        f"## Original problem\n{original_problem}\n\n"
        f"## Formulation to review\n```json\n{formulation_json}\n```"
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    response = model.invoke([
        {"role": "system", "content": _ADJUDICATOR_SYSTEM_PROMPT},
        {"role": "human", "content": user_msg},
    ])

    raw = _extract_text(response.content)
    data = _extract_json(raw)

    approved = data.get("approved", False)
    issues = data.get("issues", [])
    form_data = data.get("formulation", json.loads(formulation_json))

    form_data["extraction_source"] = "adjudicator"
    if issues:
        form_data["confidence_notes"] = f"Adjudicator issues: {'; '.join(issues)}"

    # Validate
    formulation = LPFormulation(**form_data)
    result = {
        "approved": approved,
        "issues": issues,
        "formulation": json.loads(formulation.model_dump_json()),
    }
    return json.dumps(result, indent=2)
