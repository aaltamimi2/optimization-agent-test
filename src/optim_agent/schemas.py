"""Canonical data contract for LP/MILP formulations.

Separate ObjectiveTerm / ConstraintTerm types prevent the LLM from confusing
objective coefficients with constraint coefficients at the schema level.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── Enums ──────────────────────────────────────────────────────────────

class VarType(str, Enum):
    CONTINUOUS = "Continuous"
    INTEGER = "Integer"
    BINARY = "Binary"


class Sense(str, Enum):
    MAX = "MAX"
    MIN = "MIN"


class ConstraintSense(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "="


# ── Building blocks ───────────────────────────────────────────────────

class Variable(BaseModel):
    """A decision variable."""
    name: str = Field(..., description="Short internal name, e.g. 'x1'")
    label: str = Field("", description="Human-readable label, e.g. 'chairs produced'")
    var_type: VarType = Field(VarType.CONTINUOUS, description="Variable domain")
    lb: float = Field(0.0, description="Lower bound")
    ub: float | None = Field(None, description="Upper bound (None = unbounded)")


class ObjectiveTerm(BaseModel):
    """One term in the objective function (e.g. 55*x1).

    These coefficients represent the OBJECTIVE value per unit — e.g. profit,
    cost, or revenue per unit of each variable.
    """
    var_name: str = Field(..., description="References Variable.name")
    coefficient: float


class ConstraintTerm(BaseModel):
    """One term on the LHS of a constraint (e.g. 3*x1).

    These coefficients represent RESOURCE USAGE per unit — e.g. hours of
    labor, kg of material consumed per unit of each variable.
    """
    var_name: str = Field(..., description="References Variable.name")
    coefficient: float


# ── Composite structures ─────────────────────────────────────────────

class Objective(BaseModel):
    """Objective function: sense (MAX/MIN) + list of terms."""
    sense: Sense
    terms: list[ObjectiveTerm] = Field(..., min_length=1)
    label: str = Field("", description="Human-readable label, e.g. 'total profit'")


class Constraint(BaseModel):
    """A single constraint: LHS terms  sense  RHS."""
    name: str = Field(..., description="Unique constraint identifier")
    terms: list[ConstraintTerm] = Field(..., min_length=1)
    sense: ConstraintSense
    rhs: float
    label: str = Field("", description="Human-readable, e.g. 'carpentry hours limit'")


# ── Top-level formulation ────────────────────────────────────────────

class LPFormulation(BaseModel):
    """Complete LP/MILP formulation extracted from natural language."""
    problem_name: str = Field("", description="Short problem title")
    variables: list[Variable] = Field(..., min_length=1)
    objective: Objective
    constraints: list[Constraint] = Field(..., min_length=1)
    extraction_source: str = Field("", description="Which extractor produced this")
    confidence_notes: str = Field("", description="Any caveats about extraction quality")

    @model_validator(mode="after")
    def _check_var_references(self) -> "LPFormulation":
        """Ensure every var_name in objective/constraint terms exists in variables."""
        valid_names = {v.name for v in self.variables}

        for i, term in enumerate(self.objective.terms):
            if term.var_name not in valid_names:
                raise ValueError(
                    f"Objective term {i} references unknown variable "
                    f"'{term.var_name}'. Valid: {valid_names}"
                )

        for c in self.constraints:
            for j, term in enumerate(c.terms):
                if term.var_name not in valid_names:
                    raise ValueError(
                        f"Constraint '{c.name}' term {j} references unknown "
                        f"variable '{term.var_name}'. Valid: {valid_names}"
                    )

        return self


# ── Solver result ────────────────────────────────────────────────────

class ConstraintDiagnostic(BaseModel):
    """Diagnostic info for one constraint after solving."""
    name: str
    lhs_value: float
    rhs: float
    slack: float
    binding: bool


class SolverResult(BaseModel):
    """Result from the PuLP/CBC solver."""
    status: str
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    constraint_diagnostics: list[ConstraintDiagnostic] = Field(default_factory=list)


# ── Validation result ────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Result of deterministic validation checks."""
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
