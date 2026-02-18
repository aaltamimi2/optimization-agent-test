"""Report generator — formats solution matching presentation slide format.

Sections: Solution Summary, Derived Equations, Constraint Diagnostics,
Interpretation.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .schemas import (
    ConstraintSense,
    LPFormulation,
    Sense,
    SolverResult,
)


def _format_objective(f: LPFormulation) -> str:
    """Format the objective function as a human-readable equation."""
    sense = "Maximize" if f.objective.sense == Sense.MAX else "Minimize"
    terms = []
    for i, t in enumerate(f.objective.terms):
        coeff = t.coefficient
        if i == 0:
            terms.append(f"{coeff}{t.var_name}")
        elif coeff >= 0:
            terms.append(f"+ {coeff}{t.var_name}")
        else:
            terms.append(f"- {abs(coeff)}{t.var_name}")
    return f"{sense} Z = {' '.join(terms)}"


def _format_constraint(c) -> str:
    """Format a constraint as a human-readable equation."""
    terms = []
    for i, t in enumerate(c.terms):
        coeff = t.coefficient
        if i == 0:
            terms.append(f"{coeff}{t.var_name}")
        elif coeff >= 0:
            terms.append(f"+ {coeff}{t.var_name}")
        else:
            terms.append(f"- {abs(coeff)}{t.var_name}")
    return f"{' '.join(terms)} {c.sense.value} {c.rhs}"


def generate_report_text(
    formulation: LPFormulation,
    result: SolverResult,
) -> str:
    """Generate a formatted solution report."""
    lines: list[str] = []

    # Header
    title = formulation.problem_name or "Optimization Problem"
    lines.append(f"# {title} — Solution Report")
    lines.append("")

    # Solution Summary
    lines.append("## Solution Summary")
    lines.append(f"- **Status**: {result.status}")
    if result.objective_value is not None:
        obj_label = formulation.objective.label or "Objective"
        lines.append(f"- **{obj_label}**: {result.objective_value}")
    lines.append("- **Optimal values**:")
    for v in formulation.variables:
        val = result.variable_values.get(v.name)
        label = f" ({v.label})" if v.label else ""
        lines.append(f"  - {v.name}{label} = {val}")
    lines.append("")

    # Derived Equations
    lines.append("## Derived Equations")
    lines.append(f"- **Objective**: {_format_objective(formulation)}")
    lines.append("- **Constraints**:")
    for c in formulation.constraints:
        label = f" [{c.label}]" if c.label else ""
        lines.append(f"  - {_format_constraint(c)}{label}")
    lines.append("")

    # Constraint Diagnostics
    if result.constraint_diagnostics:
        lines.append("## Constraint Diagnostics")
        lines.append("| Constraint | LHS | RHS | Slack | Binding |")
        lines.append("|---|---|---|---|---|")
        for d in result.constraint_diagnostics:
            binding = "Yes" if d.binding else "No"
            lines.append(
                f"| {d.name} | {d.lhs_value:.1f} | {d.rhs:.1f} | "
                f"{d.slack:.1f} | {binding} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    if result.status == "Optimal":
        binding = [
            d.name for d in result.constraint_diagnostics if d.binding
        ]
        if binding:
            lines.append(
                f"The optimal solution is bounded by: {', '.join(binding)}. "
                f"These are the active constraints — relaxing any of them "
                f"could improve the objective."
            )
        non_binding = [
            d.name for d in result.constraint_diagnostics if not d.binding
        ]
        if non_binding:
            lines.append(
                f"Non-binding constraints ({', '.join(non_binding)}) have "
                f"unused capacity (slack > 0)."
            )
    else:
        lines.append(f"The solver returned status: {result.status}.")

    return "\n".join(lines)


@tool
def generate_report(formulation_json: str, solver_result_json: str) -> str:
    """Generate a formatted solution report from formulation and solver result.

    Takes the LP formulation and solver result as JSON strings, and produces
    a human-readable report with solution summary, equations, constraint
    diagnostics, and interpretation.
    """
    formulation = LPFormulation(**json.loads(formulation_json))
    result = SolverResult(**json.loads(solver_result_json))
    return generate_report_text(formulation, result)
