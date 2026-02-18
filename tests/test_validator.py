"""Tests for deterministic validator — no API needed."""

import pytest

from optim_agent.schemas import (
    Constraint,
    ConstraintSense,
    ConstraintTerm,
    LPFormulation,
    Objective,
    ObjectiveTerm,
    Sense,
    Variable,
    VarType,
)
from optim_agent.validator import validate_formulation


def _make_formulation(**overrides):
    defaults = dict(
        problem_name="test",
        variables=[
            Variable(name="x1", label="tables", var_type=VarType.INTEGER, lb=0),
            Variable(name="x2", label="chairs", var_type=VarType.INTEGER, lb=0),
        ],
        objective=Objective(
            sense=Sense.MAX,
            terms=[
                ObjectiveTerm(var_name="x1", coefficient=55),
                ObjectiveTerm(var_name="x2", coefficient=45),
            ],
            label="total profit",
        ),
        constraints=[
            Constraint(
                name="carpentry",
                terms=[
                    ConstraintTerm(var_name="x1", coefficient=3),
                    ConstraintTerm(var_name="x2", coefficient=2),
                ],
                sense=ConstraintSense.LE,
                rhs=90,
                label="carpentry hours",
            ),
            Constraint(
                name="painting",
                terms=[
                    ConstraintTerm(var_name="x1", coefficient=1),
                    ConstraintTerm(var_name="x2", coefficient=2),
                ],
                sense=ConstraintSense.LE,
                rhs=62,
                label="painting hours",
            ),
        ],
    )
    defaults.update(overrides)
    return LPFormulation(**defaults)


class TestValidator:
    def test_valid_formulation_passes(self):
        f = _make_formulation()
        result = validate_formulation(f)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_coefficient_equality_warning(self):
        """Warns when constraint coeff == objective coeff for same variable."""
        f = _make_formulation(
            constraints=[
                Constraint(
                    name="bad_constraint",
                    terms=[
                        # 55 matches the objective coeff for x1 — likely copy-paste
                        ConstraintTerm(var_name="x1", coefficient=55),
                        ConstraintTerm(var_name="x2", coefficient=2),
                    ],
                    sense=ConstraintSense.LE,
                    rhs=90,
                ),
            ]
        )
        result = validate_formulation(f)
        assert result.passed is True  # warnings don't fail
        assert any("copy-paste" in w for w in result.warnings)

    def test_lb_greater_than_ub_error(self):
        f = _make_formulation(
            variables=[
                Variable(name="x1", var_type=VarType.INTEGER, lb=100, ub=10),
                Variable(name="x2", var_type=VarType.INTEGER, lb=0),
            ]
        )
        result = validate_formulation(f)
        assert result.passed is False
        assert any("lb" in e and "ub" in e for e in result.errors)

    def test_duplicate_constraint_names(self):
        f = _make_formulation(
            constraints=[
                Constraint(
                    name="same",
                    terms=[ConstraintTerm(var_name="x1", coefficient=3)],
                    sense=ConstraintSense.LE,
                    rhs=90,
                ),
                Constraint(
                    name="same",
                    terms=[ConstraintTerm(var_name="x2", coefficient=2)],
                    sense=ConstraintSense.LE,
                    rhs=60,
                ),
            ]
        )
        result = validate_formulation(f)
        assert result.passed is False
        assert any("Duplicate" in e for e in result.errors)

    def test_variable_only_in_objective_warning(self):
        """Warns if a variable appears in objective but not constraints."""
        f = _make_formulation(
            variables=[
                Variable(name="x1", var_type=VarType.INTEGER, lb=0),
                Variable(name="x2", var_type=VarType.INTEGER, lb=0),
                Variable(name="x3", var_type=VarType.INTEGER, lb=0),
            ],
            objective=Objective(
                sense=Sense.MAX,
                terms=[
                    ObjectiveTerm(var_name="x1", coefficient=55),
                    ObjectiveTerm(var_name="x2", coefficient=45),
                    ObjectiveTerm(var_name="x3", coefficient=30),
                ],
            ),
            constraints=[
                Constraint(
                    name="c1",
                    terms=[
                        ConstraintTerm(var_name="x1", coefficient=3),
                        ConstraintTerm(var_name="x2", coefficient=2),
                    ],
                    sense=ConstraintSense.LE,
                    rhs=90,
                ),
            ],
        )
        result = validate_formulation(f)
        assert result.passed is True
        assert any("x3" in w for w in result.warnings)

    def test_no_warnings_on_clean_formulation(self):
        """A clean furniture problem should have zero warnings."""
        f = _make_formulation()
        result = validate_formulation(f)
        assert len(result.warnings) == 0
