"""Critic: diff two extractions + LLM reconciliation.

If extractors agree on everything, passes through unchanged.
Otherwise a Gemini 2.5 Pro call reconciles disagreements with explicit
reasoning — never averaging coefficients.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import LPFormulation


# ── Deterministic diff ───────────────────────────────────────────────

def diff_formulations(a: LPFormulation, b: LPFormulation) -> dict:
    """Field-by-field comparison of two formulations.

    Returns a dict with keys for each area of disagreement.
    Empty dict means full agreement.
    """
    diffs: dict[str, dict] = {}

    # Objective sense
    if a.objective.sense != b.objective.sense:
        diffs["objective_sense"] = {
            "a": a.objective.sense.value,
            "b": b.objective.sense.value,
        }

    # Objective coefficients per variable
    a_obj = {t.var_name: t.coefficient for t in a.objective.terms}
    b_obj = {t.var_name: t.coefficient for t in b.objective.terms}
    obj_diff = {}
    all_obj_vars = set(a_obj) | set(b_obj)
    for v in sorted(all_obj_vars):
        ca = a_obj.get(v)
        cb = b_obj.get(v)
        if ca != cb:
            obj_diff[v] = {"a": ca, "b": cb}
    if obj_diff:
        diffs["objective_coefficients"] = obj_diff

    # Constraint coefficients and RHS
    a_cons = {c.name: c for c in a.constraints}
    b_cons = {c.name: c for c in b.constraints}
    all_cnames = sorted(set(a_cons) | set(b_cons))
    cons_diff = {}
    for cname in all_cnames:
        ca = a_cons.get(cname)
        cb = b_cons.get(cname)
        if ca is None or cb is None:
            cons_diff[cname] = {
                "a": ca.model_dump() if ca else None,
                "b": cb.model_dump() if cb else None,
                "issue": "constraint exists in only one formulation",
            }
            continue
        cdiff = {}
        # Compare coefficients
        a_terms = {t.var_name: t.coefficient for t in ca.terms}
        b_terms = {t.var_name: t.coefficient for t in cb.terms}
        term_vars = set(a_terms) | set(b_terms)
        term_diff = {}
        for v in sorted(term_vars):
            ta = a_terms.get(v)
            tb = b_terms.get(v)
            if ta != tb:
                term_diff[v] = {"a": ta, "b": tb}
        if term_diff:
            cdiff["coefficients"] = term_diff
        # Compare RHS
        if ca.rhs != cb.rhs:
            cdiff["rhs"] = {"a": ca.rhs, "b": cb.rhs}
        # Compare sense
        if ca.sense != cb.sense:
            cdiff["sense"] = {"a": ca.sense.value, "b": cb.sense.value}
        if cdiff:
            cons_diff[cname] = cdiff
    if cons_diff:
        diffs["constraints"] = cons_diff

    # Variable types
    a_vtypes = {v.name: v.var_type.value for v in a.variables}
    b_vtypes = {v.name: v.var_type.value for v in b.variables}
    vtype_diff = {}
    for v in sorted(set(a_vtypes) | set(b_vtypes)):
        ta = a_vtypes.get(v)
        tb = b_vtypes.get(v)
        if ta != tb:
            vtype_diff[v] = {"a": ta, "b": tb}
    if vtype_diff:
        diffs["variable_types"] = vtype_diff

    return diffs


# ── Critic prompt ────────────────────────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """\
You are an expert LP/MILP formulation critic. You are given:
1. The original natural-language problem
2. Two JSON formulations (A and B) from different extractors
3. A deterministic diff showing exactly which fields disagree

Your job: produce a SINGLE corrected JSON formulation.

RULES:
- NEVER average coefficients. Pick the correct one with explicit reasoning.
- Objective coefficients measure VALUE per unit (profit, cost, revenue).
- Constraint coefficients measure RESOURCE USAGE per unit (hours, kg, area).
- These are almost always different numbers.
- If both extractors agree on a field, keep it.
- For each disagreement, state which extractor is correct and why.
- Output the corrected formulation inside a ```json``` code block.
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
    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


@tool
def run_critic(
    formulation_a_json: str,
    formulation_b_json: str,
    original_problem: str,
) -> str:
    """Compare two LP formulations and reconcile any disagreements.

    Takes two JSON formulations from different extractors and the original
    problem text. If they agree, returns the first one unchanged. If they
    disagree, uses Gemini 2.5 Pro to pick the correct values with reasoning.
    """
    a = LPFormulation(**json.loads(formulation_a_json))
    b = LPFormulation(**json.loads(formulation_b_json))

    diffs = diff_formulations(a, b)

    # If extractors agree, pass through
    if not diffs:
        a_data = a.model_dump()
        a_data["confidence_notes"] = "Both extractors agreed on all fields."
        return json.dumps(a_data, indent=2)

    # Build critic prompt
    user_msg = (
        f"## Original problem\n{original_problem}\n\n"
        f"## Formulation A\n```json\n{formulation_a_json}\n```\n\n"
        f"## Formulation B\n```json\n{formulation_b_json}\n```\n\n"
        f"## Diff (fields where A and B disagree)\n```json\n"
        f"{json.dumps(diffs, indent=2)}\n```\n\n"
        f"Produce the corrected formulation."
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
    response = model.invoke([
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "human", "content": user_msg},
    ])

    raw = _extract_text(response.content)
    data = _extract_json(raw)
    data["extraction_source"] = "critic"
    # Validate
    formulation = LPFormulation(**data)
    return formulation.model_dump_json(indent=2)
