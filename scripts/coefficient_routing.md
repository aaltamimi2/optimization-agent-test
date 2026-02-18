# How Coefficients Get Routed to the Right Place

The central challenge of this system: given a paragraph like

> *"Each table yields a profit of $55... A table requires 3 hours of carpentry..."*

how does `55` end up in the objective function and `3` end up in the constraint — and not the other way around?

This document walks through every mechanism that makes that happen, from the prompt that first reads the text to the final arithmetic check before solving.

---

## Table of Contents

1. [The Problem: Why LLMs Swap Coefficients](#1-the-problem-why-llms-swap-coefficients)
2. [Layer 1: The Extraction Prompt (SCRATCHPAD)](#2-layer-1-the-extraction-prompt-scratchpad)
3. [Layer 2: Schema-Level Type Separation](#3-layer-2-schema-level-type-separation)
4. [Layer 3: Label Fields as Audit Trail](#4-layer-3-label-fields-as-audit-trail)
5. [Layer 4: Deterministic Validator](#5-layer-4-deterministic-validator)
6. [Layer 5: Dual Extraction + Critic Diff](#6-layer-5-dual-extraction--critic-diff)
7. [Layer 6: Adjudicator Arithmetic Check](#7-layer-6-adjudicator-arithmetic-check)
8. [End-to-End Example: Furniture Workshop](#8-end-to-end-example-furniture-workshop)
9. [What Each Layer Catches](#9-what-each-layer-catches)

---

## 1. The Problem: Why LLMs Swap Coefficients

Consider the furniture workshop problem:

> *A furniture workshop produces tables and chairs. Each table yields a profit of **$55** and each chair yields a profit of **$45**. A table requires **3 hours** of carpentry and **1 hour** of painting. A chair requires **2 hours** of carpentry and **2 hours** of painting.*

There are six numbers in this paragraph: `55, 45, 3, 1, 2, 2`. An LLM needs to place each one correctly:

```
Correct:                          Wrong (common LLM error):
  Objective: 55*x1 + 45*x2         Objective: 3*x1 + 2*x2    ← swapped!
  Carpentry: 3*x1 + 2*x2           Carpentry: 55*x1 + 45*x2  ← swapped!
  Painting:  1*x1 + 2*x2           Painting:  1*x1 + 2*x2
```

This happens because all six numbers appear near the same variables (`tables`, `chairs`) in the text. Without careful prompting, the LLM treats them as interchangeable. The system uses **six stacked layers** to prevent this.

---

## 2. Layer 1: The Extraction Prompt (SCRATCHPAD)

**File**: `extraction.py`, lines 20-72

The extraction prompt is where the coefficient-routing decision actually happens. It uses three techniques:

### Technique A: Explicit semantic labels in the prompt

The prompt opens with a `CRITICAL` instruction that defines what each type of coefficient means:

```
CRITICAL: Objective coefficients and constraint coefficients are DIFFERENT
numbers from the problem text.  Objective coefficients measure VALUE per unit
(profit, cost, revenue).  Constraint coefficients measure RESOURCE USAGE per
unit (hours, kg, area).  NEVER copy objective coefficients into constraints
or vice versa.
```

This tells the LLM the **semantic category** of each slot before it starts reading the problem. It's not just "fill in a number" — it's "fill in the number that represents VALUE" vs "fill in the number that represents RESOURCE USAGE."

### Technique B: Forced SCRATCHPAD reasoning

Before writing any JSON, the LLM must complete a scratchpad:

```
1. SCRATCHPAD — Before writing any JSON, think step by step:
   - List every decision variable and what it represents.
   - Identify the objective: what is being maximised/minimised, and what is
     the per-unit value for each variable?
   - Identify each constraint: what resource is limited, what is each
     variable's per-unit consumption, and what is the total available?
   - Double-check: are the objective coefficients different from the
     constraint coefficients?  They almost always are.
```

This is a chain-of-thought forcing mechanism. The LLM can't jump straight to JSON — it must first articulate:

- *"x1 = tables, x2 = chairs"*
- *"Objective: profit. Tables earn $55 each, chairs earn $45 each."*
- *"Carpentry constraint: tables use 3 hours, chairs use 2 hours."*
- *"Are 55 and 3 different? Yes. OK, proceeding."*

The scratchpad step 4 ("double-check") is specifically designed to catch the swap error before JSON is written.

### Technique C: Semantic annotations in the JSON schema

The JSON template in the prompt labels each coefficient slot:

```json
"objective": {
    "terms": [{"var_name": "x1", "coefficient": "<objective coeff>"}]
}
"constraints": [{
    "terms": [{"var_name": "x1", "coefficient": "<resource usage coeff>"}]
}]
```

And the prompt ends with a final reminder:

```
Remember: objective term coefficients represent VALUE (profit/cost per unit),
constraint term coefficients represent RESOURCE USAGE (hours/kg per unit).
These are almost always different numbers.
```

This creates a **three-point reinforcement** within a single prompt: the CRITICAL header, the scratchpad, and the closing reminder.

---

## 3. Layer 2: Schema-Level Type Separation

**File**: `schemas.py`, lines 45-62

The Pydantic schema uses **two separate types** for objective and constraint coefficients:

```python
class ObjectiveTerm(BaseModel):
    """These coefficients represent the OBJECTIVE value per unit —
    e.g. profit, cost, or revenue per unit of each variable."""
    var_name: str
    coefficient: float

class ConstraintTerm(BaseModel):
    """These coefficients represent RESOURCE USAGE per unit —
    e.g. hours of labor, kg of material consumed per unit."""
    var_name: str
    coefficient: float
```

These two classes have **identical fields** (`var_name`, `coefficient`). So why have two types?

Because the LLM sees the **class name** and **docstring** when generating JSON. When it's filling in an `ObjectiveTerm`, the context says "OBJECTIVE value per unit." When it's filling in a `ConstraintTerm`, the context says "RESOURCE USAGE per unit." The type name itself acts as a routing signal.

The objective uses `list[ObjectiveTerm]` and constraints use `list[ConstraintTerm]`:

```python
class Objective(BaseModel):
    terms: list[ObjectiveTerm]    # ← "these are VALUE coefficients"

class Constraint(BaseModel):
    terms: list[ConstraintTerm]   # ← "these are USAGE coefficients"
```

If we had used a single generic `Term` type for both, the LLM would have no type-level signal about which kind of number belongs where.

---

## 4. Layer 3: Label Fields as Audit Trail

**File**: `schemas.py`, lines 39, 71, 80

Every schema element has a `label` field that links back to the natural-language source:

```python
class Variable(BaseModel):
    name: str       # "x1"
    label: str      # "tables produced"    ← WHERE did this come from?

class Objective(BaseModel):
    terms: list[ObjectiveTerm]
    label: str      # "total weekly profit" ← WHAT does the objective measure?

class Constraint(BaseModel):
    terms: list[ConstraintTerm]
    label: str      # "carpentry hours"     ← WHAT resource is constrained?
```

These labels serve two purposes:

1. **During extraction**: The LLM must write `"label": "carpentry hours"` next to the constraint coefficients. This forces it to think about what the constraint *represents*, which reinforces using the correct coefficients. If the LLM wrote `"label": "carpentry hours"` but put `coefficient: 55`, the semantic mismatch is obvious.

2. **During review**: The critic and adjudicator can read the labels to sanity-check. If a constraint is labeled "carpentry hours" but has coefficient 55, something is wrong — $55 is not an amount of hours.

---

## 5. Layer 4: Deterministic Validator

**File**: `validator.py`, lines 54-67

The validator runs **no LLM calls** — it's pure Python checks. The key check for coefficient accuracy:

```python
# Build a map: variable_name → objective coefficient
obj_coeffs = {t.var_name: t.coefficient for t in formulation.objective.terms}

for constraint in formulation.constraints:
    for term in constraint.terms:
        obj_coeff = obj_coeffs.get(term.var_name)
        if obj_coeff is not None and term.coefficient == obj_coeff:
            warnings.append(
                f"Constraint '{constraint.name}' has coefficient "
                f"{term.coefficient} for '{term.var_name}', which equals "
                f"the objective coefficient. This is often a copy-paste error."
            )
```

**What this catches**: If the LLM copied `55` from the objective into the carpentry constraint (both for `x1`), this check fires:

```
WARNING: Constraint 'carpentry' has coefficient 55 for 'x1', which equals
the objective coefficient. This is often a copy-paste error.
```

**Why this works**: In real-world optimization problems, objective coefficients (profits, costs) and constraint coefficients (resource usage rates) are almost never the same number. A table doesn't "earn $3 profit" AND "use 3 hours of carpentry" — that would be a coincidence. So coefficient equality between objective and constraint for the same variable is a strong signal of a copy-paste error.

**Limitation**: This is a heuristic. In rare problems where a coefficient genuinely appears in both the objective and a constraint (e.g., `cost = 5` per unit and `5 units of resource per item`), this would produce a false-positive warning. That's why it's a *warning*, not an error — it doesn't block the pipeline, it just flags it for the critic to examine.

---

## 6. Layer 5: Dual Extraction + Critic Diff

**Files**: `extraction.py` (dual calls), `critic.py` (diff + reconciliation)

### Why two extractors?

Two different Gemini models (Pro and Flash) extract the same problem independently. If they both make the same mistake, we can't catch it. But if they make **different** mistakes, the disagreement is visible and the critic can fix it.

### How the diff works

The critic first runs a **deterministic, field-by-field diff** (`critic.py`, lines 20-99). For the furniture problem, imagine extractor A got it right but extractor B swapped the carpentry coefficients:

```
Extractor A (correct):           Extractor B (wrong):
  Objective: 55*x1 + 45*x2        Objective: 55*x1 + 45*x2     ✓ agree
  Carpentry: 3*x1 + 2*x2          Carpentry: 55*x1 + 45*x2     ✗ disagree!
  Painting:  1*x1 + 2*x2          Painting:  1*x1 + 2*x2       ✓ agree
```

The diff output would be:

```json
{
    "constraints": {
        "carpentry": {
            "coefficients": {
                "x1": {"a": 3, "b": 55},
                "x2": {"a": 2, "b": 45}
            }
        }
    }
}
```

### How the critic resolves disagreements

The diff (not the full formulations) is passed to a Gemini 2.5 Pro critic call with this key instruction:

```
RULES:
- NEVER average coefficients. Pick the correct one with explicit reasoning.
- Objective coefficients measure VALUE per unit (profit, cost, revenue).
- Constraint coefficients measure RESOURCE USAGE per unit (hours, kg, area).
```

The critic sees exactly which fields disagree and must pick one. For the carpentry example:

> *"The constraint labeled 'carpentry hours' should use hours-per-unit, not dollars-per-unit. The problem says 'A table requires 3 hours of carpentry.' Extractor A's coefficient of 3 is correct. Extractor B copied the objective coefficient (55) into the constraint."*

The **"NEVER average"** rule is critical. Without it, an LLM might "compromise" by averaging 3 and 55 to get 29, which is wrong in both directions. The critic must choose A's value or B's value with reasoning.

### When extractors agree

If both extractors produce identical formulations (as happened in our furniture workshop test run), the diff is empty and the critic passes through unchanged with a note: `"Both extractors agreed on all fields."` No LLM call is needed.

---

## 7. Layer 6: Adjudicator Arithmetic Check

**File**: `adjudicator.py`, lines 17-41

The adjudicator is the final gate. It asks Gemini Flash to verify the formulation with **manual arithmetic**:

```
Check ALL of the following:
- Can the objective value be computed manually from the variable values?
  (e.g., if x1=14, x2=24, does 55*14 + 45*24 = 1850?)
- Are all constraints evaluable with the given variable types?
- Do the coefficients in the objective match the VALUE terms in the problem?
- Do the coefficients in the constraints match the RESOURCE USAGE terms?
```

This catches errors that slipped through all previous layers. For example, if somehow the formulation has `objective: 3*x1 + 2*x2` and `carpentry: 55*x1 + 45*x2` (a complete swap), the adjudicator would compute:

> *"If x1=14, x2=24: objective = 3(14) + 2(24) = 90. But the problem says maximum profit should be $1,850. 90 is way too low — the objective coefficients are wrong. They should be 55 and 45 (the profit values), not 3 and 2 (the carpentry hours)."*

The arithmetic check is powerful because it works **backwards** from expected magnitudes. Profit of $90 for a workshop doesn't make sense; profit of $1,850 does.

---

## 8. End-to-End Example: Furniture Workshop

Here's how the number `55` (profit per table) flows through all six layers:

```
    PROBLEM TEXT
    "Each table yields a profit of $55"
    "A table requires 3 hours of carpentry"
                    │
                    ▼
    LAYER 1: EXTRACTION PROMPT
    Scratchpad: "x1 = tables. Profit per table = $55.
                 Carpentry per table = 3 hours.
                 Are 55 and 3 different? Yes."
    → ObjectiveTerm(var_name="x1", coefficient=55)
    → ConstraintTerm(var_name="x1", coefficient=3)
                    │
                    ▼
    LAYER 2: SCHEMA TYPES
    55 stored in ObjectiveTerm  (type says: "VALUE per unit")
    3  stored in ConstraintTerm (type says: "RESOURCE USAGE per unit")
    Pydantic validates: both reference known variable "x1" ✓
                    │
                    ▼
    LAYER 3: LABELS
    Objective label: "total weekly profit"     (55 is a profit → ✓)
    Constraint label: "carpentry hours"        (3 is hours → ✓)
    Labels match coefficient semantics.
                    │
                    ▼
    LAYER 4: VALIDATOR
    Check: does constraint coeff (3) == objective coeff (55) for x1?
    3 ≠ 55 → no warning. Clean pass. ✓
                    │
                    ▼
    LAYER 5: DUAL EXTRACTION + CRITIC
    Extractor A (Pro):  objective x1=55, carpentry x1=3
    Extractor B (Flash): objective x1=55, carpentry x1=3
    Diff: empty → both agree → pass through unchanged. ✓
                    │
                    ▼
    LAYER 6: ADJUDICATOR
    "Does 55*14 + 45*24 = 1850? → 770 + 1080 = 1850. ✓"
    "Does 3*14 + 2*24 = 90?  → 42 + 48 = 90.  ✓"
    Approved.
                    │
                    ▼
    SOLVER
    PuLP/CBC: x1=14, x2=24, profit=1850 ✓
```

---

## 9. What Each Layer Catches

| Layer | What it catches | Type | Can it fix? |
|-------|----------------|------|-------------|
| **Extraction prompt** (SCRATCHPAD) | Prevents the swap from happening in the first place by forcing the LLM to articulate what each number means before writing JSON | Preventive | N/A — prevents the error |
| **Schema types** (`ObjectiveTerm` vs `ConstraintTerm`) | Makes the LLM write coefficients into semantically-labeled containers, reducing confusion | Preventive | N/A — structural signal |
| **Label fields** | Creates a human-readable audit trail; semantic mismatch (e.g., "carpentry hours" with coefficient 55) is visible | Detective | No — informational |
| **Validator** (coefficient equality check) | Catches the #1 error: copying objective coefficients into constraints | Detective | No — warns only, doesn't auto-fix |
| **Dual extraction + Critic** | Catches errors where one model gets it right and the other doesn't | Detective + Corrective | **Yes** — critic picks the correct value with reasoning |
| **Adjudicator** (arithmetic check) | Catches errors that slipped through all previous layers by verifying the math makes sense | Detective + Corrective | **Yes** — can fix and return corrected formulation |

### Defense in depth

No single layer is foolproof:

- The **prompt** can be ignored by the LLM.
- The **schema types** are just labels — they don't enforce numeric ranges.
- The **validator** only catches exact equality, not near-misses.
- The **dual extraction** fails when both models make the same mistake.
- The **adjudicator** is another LLM call that could itself make errors.

But stacked together, the probability of a coefficient swap surviving all six layers is very low. Each layer catches a different failure mode, and the layers are **independent** — a prompt failure doesn't disable the validator, and a validator miss doesn't disable the critic.
