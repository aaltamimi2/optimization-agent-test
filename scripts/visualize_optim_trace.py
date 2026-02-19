"""
Visualize optim-agent pipeline trace:
extract_a || extract_b -> validate -> critic -> adjudicator -> solve -> report
for the Furniture Workshop problem (MAX profit, tables + chairs).

Style matches DISSOLVE trace15 (hand-crafted card layout).
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans", "Arial", "DejaVu Sans",
]
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import textwrap
import os


# ── Color palette ────────────────────────────────────────────────────
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"

C_USER       = "#6366F1"   # Indigo — user query / orchestrator
C_ROUTER     = "#22C55E"   # Green — routing middleware
C_EXTRACT    = "#14B8A6"   # Teal — dual extraction (step 1)
C_VALIDATE   = "#F97316"   # Orange — validation (step 2)
C_CRITIC     = "#8B5CF6"   # Violet — critic (step 3)
C_ADJUD      = "#0EA5E9"   # Sky — adjudicator (step 4)
C_SOLVER     = "#64748B"   # Slate — solver (step 5)
C_REPORT     = "#10B981"   # Emerald — report (step 6)
C_META       = "#D97706"   # Amber — coefficient routing / metadata

C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_OK         = "#15803D"   # Dark green for pass/check marks


# ── Layout constants ─────────────────────────────────────────────────
FIG_W  = 7.5
FIG_H  = 23.0

LEFT     = 0.010
RIGHT    = 0.990
WIDTH    = RIGHT - LEFT
MID      = (LEFT + RIGHT) / 2
PAD      = 0.015
ACCENT_W = 0.006
PILL_H   = 0.016
GAP      = 0.006


# ── Helpers ──────────────────────────────────────────────────────────

def _box(ax, x, y_top, w, h, accent_color, radius=0.005):
    y_bot = y_top - h
    box = FancyBboxPatch(
        (x, y_bot), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(box)
    bar = Rectangle(
        (x + 0.002, y_bot + radius), ACCENT_W, h - 2 * radius,
        facecolor=accent_color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(bar)
    return y_bot


def _divider(ax, x1, x2, y, color="#D1D5DB", lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            transform=ax.transAxes, zorder=3)


def _pill(ax, x, y_center, text, fg, bg, border, fs=6.5, mono=True):
    tw = len(text) * (0.0075 if mono else 0.0065) + 0.014
    pill = FancyBboxPatch(
        (x, y_center - 0.008), tw, 0.016,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        facecolor=bg, edgecolor=border, linewidth=0.6,
        transform=ax.transAxes, clip_on=False, zorder=4,
    )
    ax.add_patch(pill)
    ax.text(
        x + tw / 2, y_center, text,
        ha="center", va="center", fontsize=fs, color=fg,
        fontfamily="monospace" if mono else "sans-serif",
        transform=ax.transAxes, zorder=5,
    )
    return tw


def _wrap(text, width=95):
    lines = text.split("\n")
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width) if len(line) > width else [line])
    return "\n".join(out)


# ── Main figure ──────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    y = 0.990
    cl = LEFT + ACCENT_W + PAD
    cr = RIGHT - PAD

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    query = (
        "A furniture workshop produces tables and chairs. Each table yields "
        "a profit of $55 and each chair yields a profit of $45. A table "
        "requires 3 hours of carpentry and 1 hour of painting. A chair "
        "requires 2 hours of carpentry and 2 hours of painting. There are "
        "90 carpentry hours and 62 painting hours available per week. The "
        "workshop can only produce whole units. How many tables and chairs "
        "should be produced to maximize weekly profit?"
    )
    box_h = 0.042
    _box(ax, LEFT, y, WIDTH, box_h, C_USER)
    ax.text(cl, y - 0.007, "User Query",
            fontsize=9, fontweight="bold", color=C_USER,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl, y - 0.017, _wrap(query, 115),
            fontsize=7.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.3)
    y -= box_h + GAP

    # ================================================================
    # 2. PIPELINE ROUTING
    # ================================================================
    box_h = 0.035
    _box(ax, LEFT, y, WIDTH, box_h, C_ROUTER)
    ax.text(cl, y - 0.007, "OptimRoutingMiddleware — Advisory Step Hints",
            fontsize=9, fontweight="bold", color="#15803D",
            transform=ax.transAxes, va="top", zorder=5)
    route_text = (
        "Step flow:  extract (parallel)  ->  validate  ->  "
        "critic  ->  adjudicator  ->  solve  ->  report"
    )
    ax.text(cl, y - 0.018, route_text,
            fontsize=7, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.4)
    ax.text(cr, y - 0.018,
            "Steps 0-6  |  Gemini 2.5 Pro / Flash  |  PuLP/CBC",
            fontsize=6.5, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    y -= box_h + GAP

    # ================================================================
    # 3. STEP 1: DUAL EXTRACTION (parallel columns)
    # ================================================================
    col_gap_inner = 0.010
    col_w = (WIDTH - col_gap_inner) / 2
    c1_x = LEFT
    c2_x = LEFT + col_w + col_gap_inner
    par_top = y
    col_h = 0.098

    # Fork arrows
    arrow_kw = dict(arrowstyle="-|>,head_length=0.4,head_width=0.25",
                    color=C_ROUTER, lw=2.0,
                    connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(c1_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP * 3 + 0.003),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)
    ax.annotate("", xy=(c2_x + col_w / 2, par_top + 0.002),
                xytext=(MID, par_top + GAP * 3 + 0.003),
                arrowprops=arrow_kw, transform=ax.transAxes, zorder=6)

    # --- Column 1: Extractor A (Gemini Pro) ---
    _box(ax, c1_x, par_top, col_w, col_h, C_EXTRACT, radius=0.004)
    a_cl = c1_x + ACCENT_W + 0.008
    a_cr = c1_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "run_extractor_a",
            fontsize=8.5, fontweight="bold", color=C_EXTRACT,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "~3.2s  |  ~12K tok",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.014

    _pill(ax, a_cl, cy, "run_extractor_a", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= PILL_H + 0.003
    _divider(ax, a_cl, a_cr, cy)
    cy -= 0.007

    ax.text(a_cl, cy, "Extracted LP Formulation:",
            fontsize=8, fontweight="bold", color="#0D9488",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    formulation_lines = [
        "Objective:  MAX  55*x1 + 45*x2",
        "Carpentry:  3*x1 + 2*x2 <= 90",
        "Painting:   1*x1 + 2*x2 <= 62",
        "Domain:     x1, x2  Integer, >= 0",
    ]
    for line in formulation_lines:
        ax.text(a_cl, cy, line, fontsize=6.5, color=C_BODY,
                fontfamily="monospace",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    ax.text(a_cl, cy, "Model: Gemini 2.5 Pro",
            fontsize=6, color="#64748B", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    # --- Column 2: Extractor B (Gemini Flash) ---
    _box(ax, c2_x, par_top, col_w, col_h, C_EXTRACT, radius=0.004)
    a_cl = c2_x + ACCENT_W + 0.008
    a_cr = c2_x + col_w - 0.006

    cy = par_top - 0.008
    ax.text(a_cl, cy, "run_extractor_b",
            fontsize=8.5, fontweight="bold", color=C_EXTRACT,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(a_cr, cy, "~1.8s  |  ~8K tok",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.014

    _pill(ax, a_cl, cy, "run_extractor_b", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6)
    cy -= PILL_H + 0.003
    _divider(ax, a_cl, a_cr, cy)
    cy -= 0.007

    ax.text(a_cl, cy, "Extracted LP Formulation:",
            fontsize=8, fontweight="bold", color="#0D9488",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    for line in formulation_lines:
        ax.text(a_cl, cy, line, fontsize=6.5, color=C_BODY,
                fontfamily="monospace",
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    ax.text(a_cl, cy, "Both extractors agreed",
            fontsize=6.5, color=C_OK, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.009
    ax.text(a_cl, cy, "Model: Gemini 2.5 Flash",
            fontsize=6, color="#64748B", style="italic",
            transform=ax.transAxes, va="top", zorder=5)

    y = par_top - col_h - GAP

    # ================================================================
    # 4. STEP 2: VALIDATION
    # ================================================================
    box_h = 0.062
    _box(ax, LEFT, y, WIDTH, box_h, C_VALIDATE)
    ax.text(cl, y - 0.007, "Step 2 — Formulation Validation",
            fontsize=9, fontweight="bold", color=C_VALIDATE,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.007, "deterministic  |  no LLM",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "validate_formulation", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.003
    _divider(ax, cl, cr, cy)
    cy -= 0.007

    ax.text(cl, cy, "7 structural checks run:",
            fontsize=8, fontweight="bold", color="#C2410C",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.42, cy, "passed=True  |  errors=0  |  warnings=0",
            fontsize=7.5, color=C_OK, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    checks = [
        ("Variables non-empty, objective has terms", C_OK),
        ("All variable references valid in objective + constraints", C_OK),
        ("Bounds valid (lb <= ub), no duplicate constraint names", C_OK),
        ("Constraint coeff != objective coeff (no copy-paste error)", C_OK),
    ]
    for text, color in checks:
        ax.text(cl + 0.008, cy, "OK", fontsize=6.5, color=color,
                fontweight="bold", fontfamily="monospace",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.035, cy, text, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ================================================================
    # 5. STEP 3: CRITIC
    # ================================================================
    box_h = 0.058
    _box(ax, LEFT, y, WIDTH, box_h, C_CRITIC)
    ax.text(cl, y - 0.007, "Step 3 — Critic Review",
            fontsize=9, fontweight="bold", color=C_CRITIC,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.007, "Gemini 2.5 Pro  |  fast path (no LLM call)",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "run_critic", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.003
    _divider(ax, cl, cr, cy)
    cy -= 0.007

    ax.text(cl, cy, "diff_formulations(a, b) returned empty  ->  no LLM call needed",
            fontsize=7, color="#64748B", style="italic",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    ax.text(cl, cy, "Both extractors agreed on all fields",
            fontsize=8, color=C_OK, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    diff_checks = [
        "Objective sense: MAX = MAX",
        "Objective coeffs: {x1: 55, x2: 45} = {x1: 55, x2: 45}",
        "Constraint coeffs + RHS: identical",
        "Variable types: Integer = Integer",
    ]
    for i, text in enumerate(diff_checks):
        ax.text(cl + 0.008, cy, "OK", fontsize=6.5, color=C_OK,
                fontweight="bold", fontfamily="monospace",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.035, cy, text, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    y -= box_h + GAP

    # ================================================================
    # 6. STEP 4: ADJUDICATOR
    # ================================================================
    box_h = 0.070
    _box(ax, LEFT, y, WIDTH, box_h, C_ADJUD)
    ax.text(cl, y - 0.007, "Step 4 — Adjudicator",
            fontsize=9, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.007, "Gemini 2.5 Flash  |  ~2.1s",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "run_adjudicator", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.003
    _divider(ax, cl, cr, cy)
    cy -= 0.007

    ax.text(cl, cy, "Arithmetic Verification:",
            fontsize=8, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.011
    ax.text(cl + 0.008, cy, "55 x 14  +  45 x 24  =  770 + 1080  =  1850",
            fontsize=7, color=C_BODY, fontfamily="monospace",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.52, cy, "OK", fontsize=7, color=C_OK,
            fontweight="bold", fontfamily="monospace",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    ax.text(cl, cy, "Coefficient Semantic Check:",
            fontsize=8, fontweight="bold", color="#0369A1",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010
    sem_checks = [
        ("Objective coeffs = VALUE per unit (profit: $55, $45)", C_OK),
        ("Constraint coeffs = RESOURCE USAGE per unit (hrs: 3,2,1,2)", C_OK),
    ]
    for text, color in sem_checks:
        ax.text(cl + 0.008, cy, "OK", fontsize=6.5, color=color,
                fontweight="bold", fontfamily="monospace",
                transform=ax.transAxes, va="top", zorder=5)
        ax.text(cl + 0.035, cy, text, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="top", zorder=5)
        cy -= 0.009

    ax.text(cl, cy, "approved=true  |  issues=[]",
            fontsize=7.5, color=C_OK, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)

    y -= box_h + GAP

    # ================================================================
    # 7. STEP 5: SOLVER
    # ================================================================
    box_h = 0.078
    _box(ax, LEFT, y, WIDTH, box_h, C_SOLVER)
    ax.text(cl, y - 0.007, "Step 5 — Solver (PuLP/CBC)",
            fontsize=9, fontweight="bold", color="#334155",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.007, "deterministic  |  no LLM",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "solve_formulation", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.003
    _divider(ax, cl, cr, cy)
    cy -= 0.007

    ax.text(cl, cy, "Status: Optimal",
            fontsize=8, fontweight="bold", color=C_OK,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.20, cy, "Objective Value: $1,850",
            fontsize=8, fontweight="bold", color=C_OK,
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.012

    ax.text(cl, cy, "x1 = 14  (tables)",
            fontsize=7.5, color=C_BODY, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cl + 0.22, cy, "x2 = 24  (chairs)",
            fontsize=7.5, color=C_BODY, fontweight="bold",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.013

    # Constraint diagnostics table
    ax.text(cl, cy, "Constraint Diagnostics:",
            fontsize=8, fontweight="bold", color="#334155",
            transform=ax.transAxes, va="top", zorder=5)
    cy -= 0.010

    tab_cols = [cl, cl + 0.18, cl + 0.30, cl + 0.42, cl + 0.54]
    headers = ["Constraint", "LHS", "RHS", "Slack", "Binding"]
    for cx, hdr in zip(tab_cols, headers):
        ax.text(cx, cy, hdr, fontsize=7, fontweight="bold",
                color="#475569", transform=ax.transAxes, va="top", zorder=5)
    _divider(ax, cl, cl + 0.64, cy - 0.008, color="#94A3B8", lw=0.8)

    diag_rows = [
        ("Carpentry", "90.0", "90", "0.0", "Yes"),
        ("Painting",  "62.0", "62", "0.0", "Yes"),
    ]
    for i, (name, lhs, rhs, slack, binding) in enumerate(diag_rows):
        ry = cy - 0.015 - i * 0.011
        vals = [name, lhs, rhs, slack, binding]
        for cx, val in zip(tab_cols, vals):
            ax.text(cx, ry, val, fontsize=7,
                    color=C_META if binding == "Yes" else C_BODY,
                    fontweight="bold" if binding == "Yes" else "normal",
                    transform=ax.transAxes, va="top", zorder=5)

    y -= box_h + GAP

    # ================================================================
    # 8. STEP 6: REPORT GENERATION
    # ================================================================
    box_h = 0.080
    _box(ax, LEFT, y, WIDTH, box_h, C_REPORT)
    ax.text(cl, y - 0.007, "Step 6 — Report Generation",
            fontsize=9, fontweight="bold", color=C_REPORT,
            transform=ax.transAxes, va="top", zorder=5)
    ax.text(cr, y - 0.007, "deterministic  |  no LLM",
            fontsize=7, color=C_BODY, ha="right",
            transform=ax.transAxes, va="top", zorder=5)

    cy = y - 0.022
    _pill(ax, cl, cy, "generate_report", C_TOOL_TEXT, C_TOOL_BG, C_TOOL_TEXT, fs=6.5)
    cy -= PILL_H + 0.003
    _divider(ax, cl, cr, cy)
    cy -= 0.007

    report_text = (
        "FURNITURE WORKSHOP  --  Solution Report\n"
        "========================================\n"
        "Status: Optimal   |   Weekly Profit: $1,850\n"
        "\n"
        "Decision Variables:\n"
        "  Tables (x1):  14 units\n"
        "  Chairs (x2):  24 units\n"
        "\n"
        "Derived Equations:\n"
        "  Maximize Z = 55*x1 + 45*x2\n"
        "  Carpentry:  3*x1 + 2*x2 <= 90  (BINDING)\n"
        "  Painting:   1*x1 + 2*x2 <= 62  (BINDING)"
    )
    ax.text(cl + 0.005, cy, report_text,
            fontsize=6, color=C_BODY, fontfamily="monospace",
            transform=ax.transAxes, va="top", zorder=5,
            linespacing=1.15)

    y -= box_h + GAP * 2

    # ================================================================
    # 9. EXECUTION TIMELINE (Gantt chart)
    # ================================================================
    ax.text(MID, y, "Execution Timeline",
            ha="center", fontsize=9, fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)
    y -= 0.004

    bar_left = LEFT + 0.01
    bar_w = WIDTH - 0.02
    bar_h = 0.014
    total_s = 9.0

    def _bar_seg(ax, y_bar, t_start, t_dur, color, label):
        x0 = bar_left + (t_start / total_s) * bar_w
        w = (t_dur / total_s) * bar_w
        ax.add_patch(FancyBboxPatch(
            (x0, y_bar), w, bar_h,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3,
        ))
        if label and w > 0.03:
            ax.text(x0 + w / 2, y_bar + bar_h / 2, label,
                    ha="center", va="center", fontsize=5.5, color="white",
                    fontweight="bold", transform=ax.transAxes, zorder=4)

    bg_kw = dict(boxstyle="round,pad=0,rounding_size=0.003",
                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)

    row_labels = ["LLM Calls", "Deterministic", "Middleware"]
    row_colors_bar = [C_EXTRACT, C_SOLVER, C_ROUTER]
    rows_y = []

    for i in range(3):
        by = y - (i + 1) * (bar_h + 0.004)
        rows_y.append(by)
        ax.add_patch(FancyBboxPatch((bar_left, by), bar_w, bar_h,
                                    transform=ax.transAxes, zorder=2, **bg_kw))
        ax.text(bar_left - 0.005, by + bar_h / 2, row_labels[i],
                ha="right", va="center", fontsize=6, color=row_colors_bar[i],
                fontweight="bold", transform=ax.transAxes)

    # LLM row: extractor_a (0-3.2s), extractor_b (0-1.8s overlay), adjudicator (5.5-7.6s)
    _bar_seg(ax, rows_y[0], 0.0, 3.2, C_EXTRACT, "extractor_a (3.2s)")
    # Draw extractor_b as overlapping lighter bar
    x0_b = bar_left + (0.0 / total_s) * bar_w
    w_b = (1.8 / total_s) * bar_w
    ax.add_patch(FancyBboxPatch(
        (x0_b, rows_y[0] + 0.002), w_b, bar_h - 0.004,
        boxstyle="round,pad=0,rounding_size=0.002",
        facecolor="#5EEAD4", edgecolor="none", alpha=0.7,
        transform=ax.transAxes, zorder=4,
    ))
    ax.text(x0_b + w_b / 2, rows_y[0] + bar_h / 2 + 0.001, "ext_b",
            ha="center", va="center", fontsize=4.5, color="#0F766E",
            fontweight="bold", transform=ax.transAxes, zorder=5)
    _bar_seg(ax, rows_y[0], 5.5, 2.1, C_ADJUD, "adjudicator (2.1s)")

    # Deterministic row: validator (3.4-3.5s), critic (3.6-3.7s), solver (7.8-8.0s), reporter (8.1-8.3s)
    _bar_seg(ax, rows_y[1], 3.4, 0.1, C_VALIDATE, "")
    _bar_seg(ax, rows_y[1], 3.6, 0.1, C_CRITIC, "")
    _bar_seg(ax, rows_y[1], 7.8, 0.2, C_SOLVER, "")
    _bar_seg(ax, rows_y[1], 8.1, 0.2, C_REPORT, "")
    # Labels below since bars are thin
    for t_mid, lbl in [(3.45, "val"), (3.65, "crit"), (7.9, "solve"), (8.2, "report")]:
        tx = bar_left + (t_mid / total_s) * bar_w
        ax.text(tx, rows_y[1] - 0.004, lbl,
                ha="center", fontsize=4.5, color=C_BODY,
                transform=ax.transAxes, zorder=5)

    # Middleware row: routing hints (0-8.5s) + guardrails (0-8.5s)
    _bar_seg(ax, rows_y[2], 0.0, 8.5, C_ROUTER, "routing hints + guardrails (continuous)")

    # Time axis
    ty = rows_y[2] - 0.012
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        tx = bar_left + (t / total_s) * bar_w
        ax.text(tx, ty, f"{t}s",
                ha="center", fontsize=6.5, color=C_BODY,
                transform=ax.transAxes)

    # Legend
    ly = ty - 0.014
    items = [
        (C_EXTRACT, "Extraction"), ("#5EEAD4", "Ext-B (parallel)"),
        (C_ADJUD, "Adjudicator"), (C_VALIDATE, "Validator"),
        (C_CRITIC, "Critic"), (C_SOLVER, "Solver"),
        (C_REPORT, "Reporter"), (C_ROUTER, "Middleware"),
    ]
    lx = bar_left + 0.005
    sp = (bar_w - 0.01) / len(items)
    for color, label in items:
        ax.plot([lx, lx + 0.012], [ly, ly], color=color, lw=4.0,
                transform=ax.transAxes, solid_capstyle="round",
                zorder=3, alpha=0.85)
        ax.text(lx + 0.016, ly, label, fontsize=5.5, color=C_BODY,
                va="center", transform=ax.transAxes)
        lx += sp

    y = ly - 0.020

    # ================================================================
    # 10. BOTTOM PANELS (three columns)
    # ================================================================
    panel_h = 0.082
    panel_gap = 0.010
    pw = (WIDTH - 2 * panel_gap) / 3
    p1x = LEFT
    p2x = LEFT + pw + panel_gap
    p3x = LEFT + 2 * (pw + panel_gap)
    pcl = ACCENT_W + 0.008

    # --- Panel 1: Trace Metadata ---
    _box(ax, p1x, y, pw, panel_h, C_USER, radius=0.004)
    ax.text(p1x + pw / 2, y - 0.007, "Trace Metadata",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    metrics = [
        ("Run Time",       "~8.5s"),
        ("Tool Calls",     "7 (total)"),
        ("LLM Calls",      "3 (ext-a, ext-b, adjud)"),
        ("Deterministic",  "4 (val, crit, solve, rpt)"),
        ("Parallel Steps", "1 (dual extraction)"),
        ("Orchestrator",   "Gemini 2.5 Pro"),
    ]
    my = y - 0.019
    for label, value in metrics:
        ax.text(p1x + pcl, my, label, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p1x + pw - 0.006, my, value, fontsize=6.5,
                fontweight="bold", color=C_TITLE, ha="right",
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.010

    # --- Panel 2: Key Results ---
    _box(ax, p2x, y, pw, panel_h, C_REPORT, radius=0.004)
    ax.text(p2x + pw / 2, y - 0.007, "Key Results",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    results_text = (
        "Optimal Solution:\n"
        "  x1 = 14 tables\n"
        "  x2 = 24 chairs\n"
        "  Profit = $1,850/week\n"
        "\n"
        "Both constraints binding\n"
        "  Carpentry: 90/90 hrs\n"
        "  Painting:  62/62 hrs"
    )
    ax.text(p2x + pcl, y - 0.018, results_text,
            fontsize=6.5, color=C_BODY, transform=ax.transAxes,
            va="top", zorder=5, linespacing=1.15)

    # --- Panel 3: Coefficient Routing ---
    _box(ax, p3x, y, pw, panel_h, C_META, radius=0.004)
    ax.text(p3x + pw / 2, y - 0.007, "Coefficient Routing",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C_TITLE, transform=ax.transAxes)
    layers = [
        ("L1", "Extraction scratchpad"),
        ("L2", "Schema type separation"),
        ("L3", "Label audit trail"),
        ("L4", "Deterministic validator"),
        ("L5", "Dual extract + critic"),
        ("L6", "Adjudicator arithmetic"),
    ]
    my = y - 0.020
    for tag, desc in layers:
        ax.text(p3x + pcl, my, tag, fontsize=6.5, color=C_META,
                fontweight="bold", fontfamily="monospace",
                transform=ax.transAxes, va="center", zorder=5)
        ax.text(p3x + pcl + 0.035, my, desc, fontsize=6.5, color=C_BODY,
                transform=ax.transAxes, va="center", zorder=5)
        my -= 0.010

    y -= panel_h + 0.005

    # ── Footer ──
    ax.text(MID, y,
            "optim-agent v0.1.0  |  Furniture Workshop  |  "
            "extract -> validate -> critic -> adjudicator -> solve -> report",
            ha="center", fontsize=7, color=C_BODY,
            transform=ax.transAxes)

    # ── Save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "visualize_optim_trace.png")
    fig.savefig(out_path, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
