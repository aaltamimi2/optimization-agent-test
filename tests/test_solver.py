"""Tests for the PuLP/CBC solver — requires pulp, no API needed."""

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
from optim_agent.solver import solve_lp


def _furniture_formulation():
    """The canonical furniture workshop problem."""
    return LPFormulation(
        problem_name="furniture_workshop",
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
            label="total weekly profit",
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


class TestSolver:
    def test_furniture_optimal(self):
        """Furniture problem: x1=14, x2=24, profit=1850."""
        f = _furniture_formulation()
        result = solve_lp(f)

        assert result.status == "Optimal"
        assert result.objective_value == pytest.approx(1850.0)
        assert result.variable_values["x1"] == pytest.approx(14.0)
        assert result.variable_values["x2"] == pytest.approx(24.0)

    def test_furniture_constraint_diagnostics(self):
        """Check constraint diagnostics: both should be binding."""
        f = _furniture_formulation()
        result = solve_lp(f)

        assert len(result.constraint_diagnostics) == 2

        diag_by_name = {d.name: d for d in result.constraint_diagnostics}

        # Carpentry: 3*14 + 2*24 = 42 + 48 = 90 (binding)
        carp = diag_by_name["carpentry"]
        assert carp.lhs_value == pytest.approx(90.0)
        assert carp.binding is True
        assert carp.slack == pytest.approx(0.0)

        # Painting: 1*14 + 2*24 = 14 + 48 = 62 (binding)
        paint = diag_by_name["painting"]
        assert paint.lhs_value == pytest.approx(62.0)
        assert paint.binding is True

    def test_infeasible_problem(self):
        """A problem with contradictory constraints should be infeasible."""
        f = LPFormulation(
            problem_name="infeasible",
            variables=[
                Variable(name="x1", var_type=VarType.CONTINUOUS, lb=0),
            ],
            objective=Objective(
                sense=Sense.MAX,
                terms=[ObjectiveTerm(var_name="x1", coefficient=1)],
            ),
            constraints=[
                Constraint(
                    name="c1",
                    terms=[ConstraintTerm(var_name="x1", coefficient=1)],
                    sense=ConstraintSense.LE,
                    rhs=5,
                ),
                Constraint(
                    name="c2",
                    terms=[ConstraintTerm(var_name="x1", coefficient=1)],
                    sense=ConstraintSense.GE,
                    rhs=10,
                ),
            ],
        )
        result = solve_lp(f)
        assert result.status == "Infeasible"

    def test_minimization(self):
        """Test a simple minimization problem."""
        f = LPFormulation(
            problem_name="min_test",
            variables=[
                Variable(name="x1", var_type=VarType.CONTINUOUS, lb=0),
                Variable(name="x2", var_type=VarType.CONTINUOUS, lb=0),
            ],
            objective=Objective(
                sense=Sense.MIN,
                terms=[
                    ObjectiveTerm(var_name="x1", coefficient=2),
                    ObjectiveTerm(var_name="x2", coefficient=3),
                ],
            ),
            constraints=[
                Constraint(
                    name="c1",
                    terms=[
                        ConstraintTerm(var_name="x1", coefficient=1),
                        ConstraintTerm(var_name="x2", coefficient=1),
                    ],
                    sense=ConstraintSense.GE,
                    rhs=10,
                ),
            ],
        )
        result = solve_lp(f)
        assert result.status == "Optimal"
        # Minimum cost: x1=10, x2=0 → cost=20
        assert result.objective_value == pytest.approx(20.0)

    def test_binary_variables(self):
        """Test binary variable handling."""
        f = LPFormulation(
            problem_name="binary_test",
            variables=[
                Variable(name="y1", var_type=VarType.BINARY, lb=0, ub=1),
                Variable(name="y2", var_type=VarType.BINARY, lb=0, ub=1),
            ],
            objective=Objective(
                sense=Sense.MAX,
                terms=[
                    ObjectiveTerm(var_name="y1", coefficient=10),
                    ObjectiveTerm(var_name="y2", coefficient=20),
                ],
            ),
            constraints=[
                Constraint(
                    name="budget",
                    terms=[
                        ConstraintTerm(var_name="y1", coefficient=5),
                        ConstraintTerm(var_name="y2", coefficient=8),
                    ],
                    sense=ConstraintSense.LE,
                    rhs=10,
                ),
            ],
        )
        result = solve_lp(f)
        assert result.status == "Optimal"
        # y1=0, y2=1 → profit=20 (or y1=1, y2=0 → profit=10)
        # Actually: y2=1, y1=0 uses 8<=10, profit=20
        #   or y1=1, y2=0 uses 5<=10, profit=10
        # Best: y1=0, y2=1 → 20
        assert result.objective_value == pytest.approx(20.0)
