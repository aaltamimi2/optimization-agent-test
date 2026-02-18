"""Tests for extraction — requires GOOGLE_API_KEY.

These tests verify coefficient accuracy against the furniture workshop problem.
Run with: pytest tests/test_extraction.py -k furniture
"""

import json
import os

import pytest

from optim_agent.schemas import LPFormulation

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)

FURNITURE_PROBLEM = """\
A furniture workshop produces tables and chairs. Each table yields a profit
of $55 and each chair yields a profit of $45. A table requires 3 hours of
carpentry and 1 hour of painting. A chair requires 2 hours of carpentry and
2 hours of painting. There are 90 carpentry hours and 62 painting hours
available per week. The workshop can only produce whole units. How many
tables and chairs should be produced to maximize weekly profit?
"""


class TestFurnitureExtraction:
    def test_extractor_a_coefficients(self):
        """Gemini Pro: objective coeffs are [55, 45], NOT [3, 2]."""
        from optim_agent.extraction import run_extractor_a

        result_json = run_extractor_a.invoke({"problem_text": FURNITURE_PROBLEM})
        f = LPFormulation(**json.loads(result_json))

        # Objective coefficients should be profit values
        obj_coeffs = {t.var_name: t.coefficient for t in f.objective.terms}
        assert set(obj_coeffs.values()) == {55.0, 45.0}, (
            f"Objective coefficients should be {{55, 45}}, got {obj_coeffs}"
        )

        # Carpentry constraint coefficients should be resource usage
        carpentry = None
        for c in f.constraints:
            c_coeffs = {t.var_name: t.coefficient for t in c.terms}
            if set(c_coeffs.values()) == {3.0, 2.0}:
                carpentry = c
                break
        assert carpentry is not None, (
            "Expected a constraint with coefficients {3, 2} (carpentry hours)"
        )

        # Variables should be INTEGER
        for v in f.variables:
            assert v.var_type.value == "Integer", (
                f"Variable {v.name} should be Integer, got {v.var_type.value}"
            )

    def test_extractor_b_coefficients(self):
        """Gemini Flash: objective coeffs are [55, 45], NOT [3, 2]."""
        from optim_agent.extraction import run_extractor_b

        result_json = run_extractor_b.invoke({"problem_text": FURNITURE_PROBLEM})
        f = LPFormulation(**json.loads(result_json))

        obj_coeffs = {t.var_name: t.coefficient for t in f.objective.terms}
        assert set(obj_coeffs.values()) == {55.0, 45.0}, (
            f"Objective coefficients should be {{55, 45}}, got {obj_coeffs}"
        )

    def test_furniture_solver_produces_1850(self):
        """Full pipeline: extraction → solve → profit=1850."""
        from optim_agent.extraction import run_extractor_a
        from optim_agent.solver import solve_lp

        result_json = run_extractor_a.invoke({"problem_text": FURNITURE_PROBLEM})
        f = LPFormulation(**json.loads(result_json))
        result = solve_lp(f)

        assert result.status == "Optimal"
        assert result.objective_value == pytest.approx(1850.0), (
            f"Expected profit=1850, got {result.objective_value}"
        )
