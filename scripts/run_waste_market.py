#!/usr/bin/env python3
"""Run the full optimization agent pipeline against the waste market coordination problem.

Based on Sampat et al. (2019) "Coordinated management of organic waste and
derived products", Case Study C, Scenario I (System with Transformation).

Expected result: social welfare = 31,920 USD.

Usage:
    python scripts/run_waste_market.py              # Full agent pipeline
    python scripts/run_waste_market.py --verify-lp  # Direct solve only (no API calls)
"""

from __future__ import annotations

import json
import logging
import sys
import time

logging.disable(logging.CRITICAL)

from optim_agent.agent import create_optim_agent, _extract_text

WASTE_MARKET_PROBLEM = """\
An Independent System Operator (ISO) coordinates a regional organic waste market
with four nodes, three products, and six flow-balance requirements. The ISO
solves a dispatch problem to maximize total social welfare: the sum of consumer
bid values minus supplier bid costs minus transport costs minus technology costs.
All quantities are in tonnes and all prices are in USD per tonne. Fractional
quantities are permitted.

Node n1 is a waste supplier. It offers up to 10,000 tonnes of raw organic waste
(product p1) at a bid cost of 2 USD per tonne. All waste that the supplier
delivers must equal the amount transported away from n1.

Node n2 is a technology hub. It accepts raw waste (p1) from n1 via transport and
processes it. The technology can handle at most 8,000 tonnes of p1, at a
processing cost of 20 USD per tonne processed. For every tonne of p1 processed,
the technology produces exactly 0.01 tonnes of high-value derived product (p2)
and exactly 0.99 tonnes of low-value byproduct (p3). All p1 arriving at n2 must
be processed; all p2 produced at n2 must be transported to n3; all p3 produced
at n2 must be transported to n4.

Node n3 is a consumer of high-value product p2. It bids 3,500 USD per tonne and
can accept at most 1,000 tonnes. All p2 arriving at n3 must be consumed by this
buyer.

Node n4 is a consumer of low-value byproduct p3. It bids 1 USD per tonne and
can accept at most 10,000 tonnes. All p3 arriving at n4 must be consumed by this
buyer.

Transport links connect the network: shipping p1 from n1 to n2 costs 5 USD per
tonne, shipping p2 from n2 to n3 costs 5 USD per tonne, and shipping p3 from n2
to n4 costs 5 USD per tonne. Each transport link can carry up to 10,000 tonnes.

Decision variables are: s1 (tonnes supplied at n1), f1 (tonnes shipped n1 to
n2), xi1 (tonnes processed at n2), f2 (tonnes shipped n2 to n3), f3 (tonnes
shipped n2 to n4), d1 (tonnes consumed at n3), d2 (tonnes consumed at n4).

What is the dispatch that maximizes total social welfare?
"""

# ── Known ground truth (Table S3, Scenario I) ─────────────────────────
EXPECTED_WELFARE = 31_920.0
EXPECTED_VARS = {
    "s1":  8000.0,
    "f1":  8000.0,
    "xi1": 8000.0,
    "f2":  80.0,
    "f3":  7920.0,
    "d1":  80.0,
    "d2":  7920.0,
}
TOLERANCE = 1.0


def verify_solution(result_messages) -> bool:
    """Scan message trace for solver result and verify against known solution."""
    for msg in result_messages:
        if getattr(msg, "type", "") != "tool":
            continue
        if getattr(msg, "name", "") != "solve_formulation":
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        obj = data.get("objective_value")
        var_vals = data.get("variable_values", {})

        print("\n" + "=" * 72)
        print("VERIFICATION AGAINST KNOWN SOLUTION (Table S3, Scenario I)")
        print("=" * 72)

        all_pass = True

        # Check objective
        if obj is not None:
            diff = abs(obj - EXPECTED_WELFARE)
            status = "PASS" if diff <= TOLERANCE else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"Social welfare: {obj:.1f} USD  (expected {EXPECTED_WELFARE:.1f})  [{status}]")
        else:
            print("Social welfare: NOT FOUND  [FAIL]")
            all_pass = False

        # Check variables
        print("\nVariable values:")
        for var, expected in EXPECTED_VARS.items():
            actual = var_vals.get(var)
            if actual is not None:
                diff = abs(actual - expected)
                status = "PASS" if diff <= TOLERANCE else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  {var:>4} = {actual:>10.1f}  (expected {expected:>10.1f})  [{status}]")
            else:
                print(f"  {var:>4} = {'NOT FOUND':>10}  [FAIL]")
                all_pass = False

        overall = "ALL CORRECT" if all_pass else "MISMATCH — CHECK EXTRACTION"
        print(f"\nOverall: {overall}")
        return all_pass

    print("VERIFICATION: No solve_formulation tool result found in trace")
    return False


def direct_solve():
    """Construct the known-good LP and solve directly (no API calls)."""
    from optim_agent.schemas import (
        Constraint,
        ConstraintTerm,
        LPFormulation,
        Objective,
        ObjectiveTerm,
        Sense,
        ConstraintSense,
        Variable,
        VarType,
    )
    from optim_agent.solver import solve_lp

    formulation = LPFormulation(
        problem_name="waste_market_case_c",
        variables=[
            Variable(name="s1",  label="supply at n1",         var_type=VarType.CONTINUOUS, lb=0, ub=10000),
            Variable(name="f1",  label="transport n1->n2",     var_type=VarType.CONTINUOUS, lb=0, ub=10000),
            Variable(name="xi1", label="processing at n2",     var_type=VarType.CONTINUOUS, lb=0, ub=8000),
            Variable(name="f2",  label="transport n2->n3",     var_type=VarType.CONTINUOUS, lb=0, ub=1000),
            Variable(name="f3",  label="transport n2->n4",     var_type=VarType.CONTINUOUS, lb=0, ub=10000),
            Variable(name="d1",  label="demand at n3",         var_type=VarType.CONTINUOUS, lb=0, ub=1000),
            Variable(name="d2",  label="demand at n4",         var_type=VarType.CONTINUOUS, lb=0, ub=10000),
        ],
        objective=Objective(
            sense=Sense.MAX,
            label="total social welfare",
            terms=[
                ObjectiveTerm(var_name="d1",  coefficient=3500),
                ObjectiveTerm(var_name="d2",  coefficient=1),
                ObjectiveTerm(var_name="s1",  coefficient=-2),
                ObjectiveTerm(var_name="f1",  coefficient=-5),
                ObjectiveTerm(var_name="f2",  coefficient=-5),
                ObjectiveTerm(var_name="f3",  coefficient=-5),
                ObjectiveTerm(var_name="xi1", coefficient=-20),
            ],
        ),
        constraints=[
            Constraint(
                name="n1_p1_balance", label="supply equals transport out at n1",
                terms=[ConstraintTerm(var_name="s1", coefficient=1), ConstraintTerm(var_name="f1", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
            Constraint(
                name="n2_p1_balance", label="transport in equals processing at n2",
                terms=[ConstraintTerm(var_name="f1", coefficient=1), ConstraintTerm(var_name="xi1", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
            Constraint(
                name="n2_p2_balance", label="p2 production equals transport out at n2",
                terms=[ConstraintTerm(var_name="xi1", coefficient=0.01), ConstraintTerm(var_name="f2", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
            Constraint(
                name="n2_p3_balance", label="p3 production equals transport out at n2",
                terms=[ConstraintTerm(var_name="xi1", coefficient=0.99), ConstraintTerm(var_name="f3", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
            Constraint(
                name="n3_p2_balance", label="transport in equals consumption at n3",
                terms=[ConstraintTerm(var_name="f2", coefficient=1), ConstraintTerm(var_name="d1", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
            Constraint(
                name="n4_p3_balance", label="transport in equals consumption at n4",
                terms=[ConstraintTerm(var_name="f3", coefficient=1), ConstraintTerm(var_name="d2", coefficient=-1)],
                sense=ConstraintSense.EQ, rhs=0,
            ),
        ],
    )

    print("=" * 72)
    print("DIRECT LP SOLVE (bypassing extraction pipeline)")
    print("=" * 72)
    print()

    result = solve_lp(formulation)
    print(f"Status: {result.status}")
    print(f"Objective value: {result.objective_value}")
    print()
    print("Variable values:")
    for var, val in sorted(result.variable_values.items()):
        expected = EXPECTED_VARS.get(var, "?")
        status = "PASS" if isinstance(expected, float) and abs(val - expected) <= TOLERANCE else "FAIL"
        print(f"  {var:>4} = {val:>10.1f}  (expected {expected:>10})  [{status}]")
    print()
    print("Constraint diagnostics:")
    for diag in result.constraint_diagnostics:
        print(f"  {diag.name:>16}: LHS={diag.lhs_value:>10.2f}  RHS={diag.rhs:>6.1f}  slack={diag.slack:.6f}  binding={diag.binding}")

    welfare_ok = result.objective_value is not None and abs(result.objective_value - EXPECTED_WELFARE) <= TOLERANCE
    print(f"\nWelfare check: {result.objective_value} == {EXPECTED_WELFARE}? {'PASS' if welfare_ok else 'FAIL'}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--verify-lp" in sys.argv:
        direct_solve()
        sys.exit(0)

    print("=" * 72)
    print("WASTE MARKET COORDINATION — FULL PIPELINE RUN")
    print("(Sampat et al. 2019, Case Study C, Scenario I)")
    print("=" * 72)
    print()
    print("Problem:")
    print(WASTE_MARKET_PROBLEM)
    print("Loading agent...")
    t0 = time.time()
    agent = create_optim_agent()
    print(f"Agent loaded in {time.time() - t0:.1f}s")
    print()
    print("Running pipeline (extract → validate → critic → adjudicator → solve → report)...")
    print(f"Expected: social welfare = {EXPECTED_WELFARE:,.0f} USD")
    print("This will make several Gemini API calls — please wait.\n")

    t1 = time.time()
    result = agent.invoke({"messages": [{"role": "user", "content": WASTE_MARKET_PROBLEM}]})
    elapsed = time.time() - t1

    # ── Full message trace ────────────────────────────────────────────
    print("=" * 72)
    print("FULL MESSAGE TRACE")
    print("=" * 72)
    for i, msg in enumerate(result["messages"]):
        msg_type = getattr(msg, "type", "unknown")
        name = getattr(msg, "name", "")
        tool_calls = getattr(msg, "tool_calls", None)

        if msg_type == "human":
            print(f"\n--- [{i}] HUMAN ---")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(content[:200] + ("..." if len(content) > 200 else ""))

        elif msg_type == "ai":
            text = _extract_text(msg.content) if msg.content else ""
            print(f"\n--- [{i}] AI ---")
            if text.strip():
                print(text[:500] + ("..." if len(text) > 500 else ""))
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name", "?")
                    tc_args = tc.get("args", {})
                    args_preview = {}
                    for k, v in tc_args.items():
                        s = str(v)
                        args_preview[k] = s[:120] + "..." if len(s) > 120 else s
                    print(f"  TOOL CALL: {tc_name}({args_preview})")

        elif msg_type == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"\n--- [{i}] TOOL RESULT: {name} ---")
            print(content[:600] + ("..." if len(content) > 600 else ""))

    # ── Verification ──────────────────────────────────────────────────
    verify_solution(result["messages"])

    # ── Final answer ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and getattr(msg, "type", "") == "ai" and msg.content:
            text = _extract_text(msg.content)
            if text.strip():
                print(text)
                break

    print(f"\nTotal pipeline time: {elapsed:.1f}s")
