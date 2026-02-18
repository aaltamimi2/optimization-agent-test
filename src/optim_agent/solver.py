"""PuLP/CBC wrapper — maps LPFormulation to PuLP objects and solves."""

from __future__ import annotations

import json

import pulp
from langchain_core.tools import tool

from .schemas import (
    ConstraintDiagnostic,
    ConstraintSense,
    LPFormulation,
    Sense,
    SolverResult,
    VarType,
)


_PULP_VAR_TYPE = {
    VarType.CONTINUOUS: pulp.LpContinuous,
    VarType.INTEGER: pulp.LpInteger,
    VarType.BINARY: pulp.LpBinary,
}


def solve_lp(formulation: LPFormulation) -> SolverResult:
    """Solve an LP/MILP formulation using PuLP/CBC.

    Maps the Pydantic schema to PuLP objects, solves, and returns
    status + variable values + constraint diagnostics.
    """
    # Sense
    lp_sense = (
        pulp.LpMaximize if formulation.objective.sense == Sense.MAX
        else pulp.LpMinimize
    )
    prob = pulp.LpProblem(formulation.problem_name or "problem", lp_sense)

    # Decision variables
    lp_vars: dict[str, pulp.LpVariable] = {}
    for v in formulation.variables:
        lp_vars[v.name] = pulp.LpVariable(
            v.name,
            lowBound=v.lb,
            upBound=v.ub,
            cat=_PULP_VAR_TYPE[v.var_type],
        )

    # Objective
    prob += pulp.lpSum(
        t.coefficient * lp_vars[t.var_name]
        for t in formulation.objective.terms
    ), "objective"

    # Constraints
    for c in formulation.constraints:
        lhs = pulp.lpSum(
            t.coefficient * lp_vars[t.var_name] for t in c.terms
        )
        if c.sense == ConstraintSense.LE:
            prob += (lhs <= c.rhs, c.name)
        elif c.sense == ConstraintSense.GE:
            prob += (lhs >= c.rhs, c.name)
        else:
            prob += (lhs == c.rhs, c.name)

    # Solve (suppress CBC output)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective) if prob.status == pulp.constants.LpStatusOptimal else None

    # Variable values
    var_vals = {}
    if prob.status == pulp.constants.LpStatusOptimal:
        for name, var in lp_vars.items():
            var_vals[name] = var.varValue

    # Constraint diagnostics
    diagnostics = []
    if prob.status == pulp.constants.LpStatusOptimal:
        for c in formulation.constraints:
            lhs_val = sum(
                t.coefficient * lp_vars[t.var_name].varValue
                for t in c.terms
            )
            slack = c.rhs - lhs_val
            if c.sense == ConstraintSense.GE:
                slack = lhs_val - c.rhs
            elif c.sense == ConstraintSense.EQ:
                slack = abs(lhs_val - c.rhs)
            diagnostics.append(ConstraintDiagnostic(
                name=c.name,
                lhs_value=lhs_val,
                rhs=c.rhs,
                slack=slack,
                binding=abs(slack) < 1e-6,
            ))

    return SolverResult(
        status=status,
        objective_value=obj_val,
        variable_values=var_vals,
        constraint_diagnostics=diagnostics,
    )


@tool
def solve_formulation(formulation_json: str) -> str:
    """Solve an LP/MILP formulation using PuLP/CBC.

    Takes a JSON string representing an LPFormulation and returns the solver
    result including optimal variable values and constraint diagnostics.
    """
    data = json.loads(formulation_json)
    formulation = LPFormulation(**data)
    result = solve_lp(formulation)
    return result.model_dump_json(indent=2)
