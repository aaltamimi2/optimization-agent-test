"""Tests for Pydantic schemas — deterministic, no API needed."""

import pytest
from pydantic import ValidationError

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


def _make_formulation(**overrides):
    """Helper to build a valid LPFormulation with optional overrides."""
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
        ],
    )
    defaults.update(overrides)
    return LPFormulation(**defaults)


class TestLPFormulation:
    def test_valid_formulation(self):
        f = _make_formulation()
        assert f.problem_name == "test"
        assert len(f.variables) == 2
        assert len(f.objective.terms) == 2
        assert len(f.constraints) == 1

    def test_objective_term_var_reference_invalid(self):
        """model_validator catches unknown var_name in objective terms."""
        with pytest.raises(ValidationError, match="unknown variable"):
            _make_formulation(
                objective=Objective(
                    sense=Sense.MAX,
                    terms=[
                        ObjectiveTerm(var_name="x1", coefficient=55),
                        ObjectiveTerm(var_name="x99", coefficient=45),
                    ],
                )
            )

    def test_constraint_term_var_reference_invalid(self):
        """model_validator catches unknown var_name in constraint terms."""
        with pytest.raises(ValidationError, match="unknown variable"):
            _make_formulation(
                constraints=[
                    Constraint(
                        name="bad",
                        terms=[ConstraintTerm(var_name="x_bad", coefficient=3)],
                        sense=ConstraintSense.LE,
                        rhs=100,
                    ),
                ]
            )

    def test_empty_variables_rejected(self):
        """min_length=1 on variables rejects empty list."""
        with pytest.raises(ValidationError):
            LPFormulation(
                problem_name="empty",
                variables=[],
                objective=Objective(
                    sense=Sense.MAX,
                    terms=[ObjectiveTerm(var_name="x1", coefficient=1)],
                ),
                constraints=[
                    Constraint(
                        name="c1",
                        terms=[ConstraintTerm(var_name="x1", coefficient=1)],
                        sense=ConstraintSense.LE,
                        rhs=10,
                    )
                ],
            )

    def test_empty_objective_terms_rejected(self):
        """min_length=1 on objective terms rejects empty list."""
        with pytest.raises(ValidationError):
            _make_formulation(
                objective=Objective(sense=Sense.MAX, terms=[])
            )

    def test_var_type_enum(self):
        f = _make_formulation()
        assert f.variables[0].var_type == VarType.INTEGER

    def test_constraint_sense_values(self):
        assert ConstraintSense.LE.value == "<="
        assert ConstraintSense.GE.value == ">="
        assert ConstraintSense.EQ.value == "="

    def test_serialization_roundtrip(self):
        f = _make_formulation()
        json_str = f.model_dump_json()
        f2 = LPFormulation.model_validate_json(json_str)
        assert f2.problem_name == f.problem_name
        assert len(f2.variables) == len(f.variables)
        assert f2.objective.terms[0].coefficient == 55
