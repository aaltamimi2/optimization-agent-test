# Optimization Agent Memory

## Coefficient Extraction Rules

### Objective vs Constraint Coefficients
- **Objective coefficients** = VALUE per unit (profit, revenue, cost per item)
- **Constraint coefficients** = RESOURCE USAGE per unit (hours, kg, area per item)
- These are ALWAYS different numbers from the problem text
- The #1 extraction error is copying objective coefficients into constraints

### Vocabulary Mappings
| NL phrase | Maps to |
|---|---|
| "earns $X per unit" | Objective coefficient |
| "profit of $X each" | Objective coefficient |
| "costs $X to make" | Objective coefficient (MIN) |
| "requires X hours" | Constraint coefficient |
| "uses X kg of material" | Constraint coefficient |
| "takes X sq ft" | Constraint coefficient |
| "available hours: N" | Constraint RHS |
| "budget of $N" | Constraint RHS |
| "at most N" | Constraint RHS with <= |
| "at least N" | Constraint RHS with >= |

### Variable Type Hints
- "whole units", "integer", "count" → Integer
- "fraction allowed", "continuous" → Continuous
- "yes/no decision" → Binary

### Furniture Workshop Reference Problem
- x1 = tables, x2 = chairs
- Objective: MAX 55*x1 + 45*x2 (profit per unit)
- Carpentry: 3*x1 + 2*x2 <= 90 (hours per unit)
- Painting: 1*x1 + 2*x2 <= 62 (hours per unit)
- Solution: x1=14, x2=24, profit=1850
