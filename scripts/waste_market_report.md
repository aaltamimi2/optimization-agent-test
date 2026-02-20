# Waste Market Coordination — Agent Pipeline Report

**Paper**: Sampat, Hu, Sharara, Aguirre-Villegas, Ruiz-Mercado, Larson & Zavala (2019). "Coordinated management of organic waste and derived products." *Computers and Chemical Engineering* 128, 352-363.

**Case Study**: SI Section 3.3, Case C (System with Transformation), Scenario I

**Date**: 2026-02-19

---

## 1. Natural Language Query

The following query was submitted to the optim-agent pipeline. It encodes Case Study C from the paper's Supplementary Information — a 4-node waste market with product transformation:

> An Independent System Operator (ISO) coordinates a regional organic waste market with four nodes, three products, and six flow-balance requirements. The ISO solves a dispatch problem to maximize total social welfare: the sum of consumer bid values minus supplier bid costs minus transport costs minus technology costs. All quantities are in tonnes and all prices are in USD per tonne. Fractional quantities are permitted.
>
> **Node n1** is a waste supplier. It offers up to 10,000 tonnes of raw organic waste (product p1) at a bid cost of 2 USD per tonne. All waste that the supplier delivers must equal the amount transported away from n1.
>
> **Node n2** is a technology hub. It accepts raw waste (p1) from n1 via transport and processes it. The technology can handle at most 8,000 tonnes of p1, at a processing cost of 20 USD per tonne processed. For every tonne of p1 processed, the technology produces exactly 0.01 tonnes of high-value derived product (p2) and exactly 0.99 tonnes of low-value byproduct (p3). All p1 arriving at n2 must be processed; all p2 produced at n2 must be transported to n3; all p3 produced at n2 must be transported to n4.
>
> **Node n3** is a consumer of high-value product p2. It bids 3,500 USD per tonne and can accept at most 1,000 tonnes. All p2 arriving at n3 must be consumed by this buyer.
>
> **Node n4** is a consumer of low-value byproduct p3. It bids 1 USD per tonne and can accept at most 10,000 tonnes. All p3 arriving at n4 must be consumed by this buyer.
>
> Transport links connect the network: shipping p1 from n1 to n2 costs 5 USD per tonne, shipping p2 from n2 to n3 costs 5 USD per tonne, and shipping p3 from n2 to n4 costs 5 USD per tonne. Each transport link can carry up to 10,000 tonnes.
>
> Decision variables are: s1 (tonnes supplied at n1), f1 (tonnes shipped n1 to n2), xi1 (tonnes processed at n2), f2 (tonnes shipped n2 to n3), f3 (tonnes shipped n2 to n4), d1 (tonnes consumed at n3), d2 (tonnes consumed at n4).
>
> What is the dispatch that maximizes total social welfare?

---

## 2. Extracted Coefficients

Both extractors (Gemini 2.5 Pro and Gemini 2.5 Flash) independently produced identical formulations. Below are the extracted coefficients.

### 2.1 Decision Variables

| Variable | Description | Type | Lower Bound | Upper Bound |
|----------|-------------|------|-------------|-------------|
| `s1` | Tonnes of raw waste (p1) supplied at n1 | Continuous | 0 | 10,000 |
| `f1` | Tonnes of p1 shipped from n1 to n2 | Continuous | 0 | 10,000 |
| `xi1` | Tonnes of p1 processed at n2 | Continuous | 0 | 8,000 |
| `f2` | Tonnes of p2 shipped from n2 to n3 | Continuous | 0 | 10,000 |
| `f3` | Tonnes of p3 shipped from n2 to n4 | Continuous | 0 | 10,000 |
| `d1` | Tonnes of p2 consumed at n3 | Continuous | 0 | 1,000 |
| `d2` | Tonnes of p3 consumed at n4 | Continuous | 0 | 10,000 |

### 2.2 Objective Function

**Maximize** total social welfare (consumer value - supplier cost - transport cost - technology cost):

```
Z = 3500*d1 + 1*d2 - 2*s1 - 5*f1 - 5*f2 - 5*f3 - 20*xi1
```

| Variable | Coefficient | Source in Paper |
|----------|-------------|----------------|
| `d1` | +3500 | Consumer bid for p2 (alpha_1^d = 3500 USD/t) |
| `d2` | +1 | Consumer bid for p3 (alpha_2^d = 1 USD/t) |
| `s1` | -2 | Supplier bid cost (alpha_1^s = 2 USD/t) |
| `f1` | -5 | Transport cost n1->n2 (alpha_1^f = 5 USD/t) |
| `f2` | -5 | Transport cost n2->n3 (alpha_2^f = 5 USD/t) |
| `f3` | -5 | Transport cost n2->n4 (alpha_3^f = 5 USD/t) |
| `xi1` | -20 | Technology processing cost (alpha_1^xi = 20 USD/t) |

### 2.3 Constraints — Conservation Laws (Equality)

These correspond to the paper's Eq. (1b): mass balance at each node for each product.

| # | Constraint | Equation | Paper Mapping |
|---|-----------|----------|---------------|
| 1 | Flow balance at n1 (p1) | `s1 - f1 = 0` | Supply in = transport out |
| 2 | Flow balance at n2 (p1) | `f1 - xi1 = 0` | Transport in = processing |
| 3 | Production balance at n2 (p2) | `0.01*xi1 - f2 = 0` | Yield gamma(p2)=0.01 |
| 4 | Production balance at n2 (p3) | `0.99*xi1 - f3 = 0` | Yield gamma(p3)=0.99 |
| 5 | Flow balance at n3 (p2) | `f2 - d1 = 0` | Transport in = consumption |
| 6 | Flow balance at n4 (p3) | `f3 - d2 = 0` | Transport in = consumption |

### 2.4 Constraints — Capacity Bounds

| # | Constraint | Equation | Slack at Optimum |
|---|-----------|----------|------------------|
| 7 | Supply limit at n1 | `s1 <= 10000` | 2,000 |
| 8 | Processing capacity at n2 | `xi1 <= 8000` | 0 (binding) |
| 9 | Demand limit at n3 (p2) | `d1 <= 1000` | 920 |
| 10 | Demand limit at n4 (p3) | `d2 <= 10000` | 2,080 |
| 11 | Transport capacity n1->n2 | `f1 <= 10000` | 2,000 |
| 12 | Transport capacity n2->n3 | `f2 <= 10000` | 9,920 |
| 13 | Transport capacity n2->n4 | `f3 <= 10000` | 2,080 |

---

## 3. Pipeline Trace

![Pipeline Trace](visualize_waste_market_trace.png)

The 6-step pipeline executed in ~231 seconds:

| Step | Tool | Engine | Result |
|------|------|--------|--------|
| 1a | `run_extractor_a` | Gemini 2.5 Pro | 7 vars, 13 constraints extracted |
| 1b | `run_extractor_b` | Gemini 2.5 Flash | 7 vars, 13 constraints extracted (parallel) |
| 2 | `validate_formulation` | Deterministic | passed=True, 0 errors, 1 warning |
| 3 | `run_critic` | Gemini 2.5 Pro | Both extractors agreed — no reconciliation needed |
| 4 | `run_adjudicator` | Gemini 2.5 Flash | approved=True, 0 issues |
| 5 | `solve_formulation` | PuLP/CBC | Status: Optimal |
| 6 | `generate_report` | Deterministic | Solution report generated |

**Validator warning (expected false positive):** Variable `d2` has objective coefficient 1.0 (the bid price) equal to its constraint coefficient 1.0 (flow balance structure). The validator flagged this as a potential copy-paste error. The adjudicator correctly dismissed it after verifying the 1 USD/tonne bid price against the problem text.

---

## 4. Results Comparison with Paper

### 4.1 Optimal Allocations

| Variable | Description | Agent Result | Paper (Table S3) | Match |
|----------|-------------|-------------|-------------------|-------|
| `s1` | Supply at n1 | 8,000 t | 8,000 t | PASS |
| `f1` | Transport n1->n2 | 8,000 t | 8,000 t | PASS |
| `xi1` | Processing at n2 | 8,000 t | 8,000 t | PASS |
| `f2` | Transport n2->n3 | 80 t | 80 t | PASS |
| `f3` | Transport n2->n4 | 7,920 t | 7,920 t | PASS |
| `d1` | Demand at n3 (p2) | 80 t | 80 t | PASS |
| `d2` | Demand at n4 (p3) | 7,920 t | 7,920 t | PASS |

### 4.2 Social Welfare

| Metric | Agent | Paper | Match |
|--------|-------|-------|-------|
| **Social Welfare** | **$31,920 USD** | **$31,920 USD** | **PASS** |

Welfare breakdown:
```
Consumer value:    3500 * 80  +  1 * 7920     = 280,000 + 7,920  = $287,920
Supplier cost:     2 * 8000                    =                    $16,000
Transport cost:    5 * 8000 + 5 * 80 + 5 * 7920 = 40,000 + 400 + 39,600 = $80,000
Technology cost:   20 * 8000                   =                    $160,000
                                                                    --------
Net welfare:       287,920 - 16,000 - 80,000 - 160,000 =           $31,920
```

### 4.3 Stakeholder Profits (from Paper)

| Player | Profit (USD) | Formula |
|--------|-------------|---------|
| Supplier (i1) | 0 | (pi_i - alpha_s) * s1 = (2 - 2) * 8000 = 0 |
| Consumer 1 (j1) | 0 | (alpha_d1 - pi_j1) * d1 = (3500 - 3500) * 80 = 0 |
| Consumer 2 (j2) | 0 | (alpha_d2 - pi_j2) * d2 = (1 - 1) * 7920 = 0 |
| Technology (t1) | 31,920 | (pi_t - alpha_xi) * xi1 = (23.99 - 20) * 8000 = 31,920 |

### 4.4 Clearing Prices (from Paper, Table S3)

| Location, Product | Price (USD/tonne) | Interpretation |
|-------------------|-------------------|----------------|
| pi(n1, p1) | 2.00 | Raw waste at supplier = supplier bid |
| pi(n2, p1) | 7.00 | Raw waste at tech hub = supplier bid + transport |
| pi(n2, p2) | 3,495.00 | High-value product at hub |
| pi(n2, p3) | -4.00 | Byproduct at hub (negative = disposal cost) |
| pi(n3, p2) | 3,500.00 | High-value product at consumer = consumer bid |
| pi(n4, p3) | 1.00 | Byproduct at consumer = consumer bid |

**Transformation price verification:**
```
pi_t = pi(n2,p2) * gamma(p2) + pi(n2,p1) * gamma(p1) + pi(n2,p3) * gamma(p3)
     = 3495 * 0.01  +  7 * (-1)  +  (-4) * 0.99
     = 34.95 - 7.00 - 3.96
     = 23.99 USD
```
Since pi_t (23.99) > alpha_xi (20), the technology provider earns a profit of (23.99 - 20) * 8000 = **$31,920**.

---

## 5. Key Observations

1. **Bottleneck is technology capacity**: The processing limit (8,000 t) is the sole binding capacity constraint. Supply (10,000 t available, 8,000 used) and demand (consumer n3 can take 1,000 t but only receives 80 t) are not limiting.

2. **High-value product drives the economics**: Product p2 at 3,500 USD/t accounts for 97.3% of consumer value ($280,000 of $287,920), despite representing only 1% of processed mass (80 of 8,000 tonnes).

3. **Technology provider captures all profit**: All other players (supplier, consumers, transporters) have zero profit — the clearing prices exactly equal their bids. The entire surplus goes to the technology provider because it creates the value chain.

4. **Negative clearing price at n2 for p3**: The byproduct p3 has a price of -4 USD/tonne at the technology hub, meaning there is a cost to disposing of the byproduct. This is offset by the high value of p2.

5. **Agent handled all framework features**: The pipeline correctly extracted yield factors (0.01, 0.99), equality conservation constraints, negative objective coefficients for costs, and a 7-variable LP — significantly more complex than the 2-variable furniture workshop reference problem.

---

## 6. Files

| File | Description |
|------|-------------|
| `scripts/run_waste_market.py` | Runner script with query, pipeline invocation, and verification |
| `scripts/visualize_waste_market_trace.py` | Trace visualization generator |
| `scripts/visualize_waste_market_trace.png` | Pipeline trace image |
| `scripts/waste_market_report.md` | This report |
| `pdf/1-s2.0-S009813541930119X-main.pdf` | Main paper |
| `pdf/1-s2.0-S009813541930119X-mmc1-SI.pdf` | Supplementary Information |
