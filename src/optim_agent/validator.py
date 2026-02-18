"""Deterministic structural and mathematical validation — no LLM calls."""

from __future__ import annotations

from .schemas import LPFormulation, ValidationResult


def validate_formulation(formulation: LPFormulation) -> ValidationResult:
    """Run deterministic checks on an LP formulation.

    Returns ValidationResult with errors (fatal) and warnings (suspicious).
    """
    errors: list[str] = []
    warnings: list[str] = []
    valid_names = {v.name for v in formulation.variables}

    # 1. Variables non-empty (already enforced by Pydantic min_length, but be safe)
    if not formulation.variables:
        errors.append("No variables defined.")

    # 2. Objective has terms
    if not formulation.objective.terms:
        errors.append("Objective function has no terms.")

    # 3. Variable references valid (also enforced by model_validator)
    for term in formulation.objective.terms:
        if term.var_name not in valid_names:
            errors.append(
                f"Objective references unknown variable '{term.var_name}'."
            )
    for c in formulation.constraints:
        for term in c.terms:
            if term.var_name not in valid_names:
                errors.append(
                    f"Constraint '{c.name}' references unknown variable "
                    f"'{term.var_name}'."
                )

    # 4. lb <= ub for all variables
    for v in formulation.variables:
        if v.ub is not None and v.lb > v.ub:
            errors.append(
                f"Variable '{v.name}': lb ({v.lb}) > ub ({v.ub})."
            )

    # 5. No duplicate constraint names
    cnames = [c.name for c in formulation.constraints]
    seen: set[str] = set()
    for cn in cnames:
        if cn in seen:
            errors.append(f"Duplicate constraint name: '{cn}'.")
        seen.add(cn)

    # 6. KEY CHECK: warn when a constraint coefficient equals an objective
    #    coefficient for the same variable (most common LLM copy-paste error)
    obj_coeffs = {t.var_name: t.coefficient for t in formulation.objective.terms}
    for c in formulation.constraints:
        for term in c.terms:
            obj_coeff = obj_coeffs.get(term.var_name)
            if obj_coeff is not None and term.coefficient == obj_coeff:
                warnings.append(
                    f"Constraint '{c.name}' has coefficient "
                    f"{term.coefficient} for '{term.var_name}', which equals "
                    f"the objective coefficient. This is often a copy-paste "
                    f"error — constraint coefficients usually differ from "
                    f"objective coefficients."
                )

    # 7. Variables in objective but not any constraint (or vice versa)
    vars_in_obj = {t.var_name for t in formulation.objective.terms}
    vars_in_cons = set()
    for c in formulation.constraints:
        for t in c.terms:
            vars_in_cons.add(t.var_name)

    only_obj = vars_in_obj - vars_in_cons
    only_con = vars_in_cons - vars_in_obj
    if only_obj:
        warnings.append(
            f"Variables in objective but no constraint: {only_obj}."
        )
    if only_con:
        warnings.append(
            f"Variables in constraints but not objective: {only_con}."
        )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
