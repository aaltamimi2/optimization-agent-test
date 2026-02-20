"""
Visualize optim-agent pipeline trace for the Waste Market Coordination problem:
extract_a || extract_b -> validate -> critic -> adjudicator -> solve -> report

Based on Sampat et al. (2019), Case Study C, Scenario I.
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


# ── Colors ───────────────────────────────────────────────────────────
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F5F5F5"
C_BOX_BORDER = "#E5E7EB"
C_USER       = "#6366F1"
C_ROUTER     = "#22C55E"
C_EXTRACT    = "#14B8A6"
C_VALIDATE   = "#F97316"
C_CRITIC     = "#8B5CF6"
C_ADJUD      = "#0EA5E9"
C_SOLVER     = "#64748B"
C_REPORT     = "#10B981"
C_META       = "#D97706"
C_PAPER      = "#DC2626"
C_TITLE      = "#1E293B"
C_BODY       = "#374151"
C_TOOL_BG    = "#FEF3C7"
C_TOOL_TEXT  = "#92400E"
C_OK         = "#15803D"
C_WARN       = "#D97706"
C_DIM        = "#64748B"

# ── Layout ───────────────────────────────────────────────────────────
FIG_W = 7.5
FIG_H = 46.0

LEFT   = 0.010
RIGHT  = 0.990
WIDTH  = RIGHT - LEFT
MID    = (LEFT + RIGHT) / 2
PAD    = 0.015
ACCENT = 0.006
GAP    = 0.004


# ── Helpers ──────────────────────────────────────────────────────────

def _box(ax, x, y_top, w, h, accent, radius=0.003):
    yb = y_top - h
    ax.add_patch(FancyBboxPatch(
        (x, yb), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, linewidth=0.8,
        transform=ax.transAxes, clip_on=False, zorder=2))
    ax.add_patch(Rectangle(
        (x + 0.002, yb + radius), ACCENT, h - 2 * radius,
        facecolor=accent, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3))


def _div(ax, x1, x2, y, color="#D1D5DB"):
    ax.plot([x1, x2], [y, y], color=color, lw=0.6,
            transform=ax.transAxes, zorder=3)


def _pill(ax, x, yc, text, fs=5.5):
    tw = len(text) * 0.006 + 0.010
    ax.add_patch(FancyBboxPatch(
        (x, yc - 0.005), tw, 0.010,
        boxstyle="round,pad=0.001,rounding_size=0.002",
        facecolor=C_TOOL_BG, edgecolor=C_TOOL_TEXT, linewidth=0.5,
        transform=ax.transAxes, clip_on=False, zorder=4))
    ax.text(x + tw / 2, yc, text, ha="center", va="center",
            fontsize=fs, color=C_TOOL_TEXT, fontfamily="monospace",
            transform=ax.transAxes, zorder=5)


def _t(ax, x, y, text, **kw):
    """Shorthand text placement."""
    defaults = dict(fontsize=6, color=C_BODY, va="top",
                    transform=ax.transAxes, zorder=5)
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


# ── Build figure ─────────────────────────────────────────────────────

def create_trace_figure():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(C_BG)

    cl = LEFT + ACCENT + PAD          # content left
    cr = RIGHT - PAD                   # content right
    y = 0.995                          # cursor

    # Spacing constants (axes units, 1 unit = FIG_H inches)
    S_TITLE  = 0.012   # space consumed by a section title line
    S_PILL   = 0.012   # pill + gap above/below
    S_DIV    = 0.004   # gap after divider
    S_LINE   = 0.006   # one text line
    S_PAD    = 0.006   # top/bottom padding inside box
    S_GAP    = GAP     # between boxes

    # ================================================================
    # TITLE
    # ================================================================
    _t(ax, MID, y, "Waste Market Coordination — Pipeline Trace",
       fontsize=11, fontweight="bold", color=C_TITLE, ha="center")
    y -= 0.007
    _t(ax, MID, y, "Sampat et al. (2019), Case Study C, Scenario I  |  optim-agent v0.1.0",
       fontsize=7, color=C_DIM, ha="center")
    y -= 0.009

    # ================================================================
    # 1. USER QUERY
    # ================================================================
    _query_raw = (
        "An Independent System Operator (ISO) coordinates a regional organic waste market "
        "with four nodes, three products, and six flow-balance requirements. The ISO "
        "solves a dispatch problem to maximize total social welfare: the sum of consumer "
        "bid values minus supplier bid costs minus transport costs minus technology costs. "
        "All quantities are in tonnes and all prices are in USD per tonne. Fractional "
        "quantities are permitted."
        "\n\n"
        "Node n1 is a waste supplier. It offers up to 10,000 tonnes of raw organic waste "
        "(product p1) at a bid cost of 2 USD per tonne. All waste that the supplier "
        "delivers must equal the amount transported away from n1."
        "\n\n"
        "Node n2 is a technology hub. It accepts raw waste (p1) from n1 via transport and "
        "processes it. The technology can handle at most 8,000 tonnes of p1, at a "
        "processing cost of 20 USD per tonne processed. For every tonne of p1 processed, "
        "the technology produces exactly 0.01 tonnes of high-value derived product (p2) "
        "and exactly 0.99 tonnes of low-value byproduct (p3). All p1 arriving at n2 must "
        "be processed; all p2 produced at n2 must be transported to n3; all p3 produced "
        "at n2 must be transported to n4."
        "\n\n"
        "Node n3 is a consumer of high-value product p2. It bids 3,500 USD per tonne and "
        "can accept at most 1,000 tonnes. All p2 arriving at n3 must be consumed by this buyer."
        "\n\n"
        "Node n4 is a consumer of low-value byproduct p3. It bids 1 USD per tonne and "
        "can accept at most 10,000 tonnes. All p3 arriving at n4 must be consumed by this buyer."
        "\n\n"
        "Transport links connect the network: shipping p1 from n1 to n2 costs 5 USD per "
        "tonne, shipping p2 from n2 to n3 costs 5 USD per tonne, and shipping p3 from n2 "
        "to n4 costs 5 USD per tonne. Each transport link can carry up to 10,000 tonnes."
        "\n\n"
        "Decision variables are: s1 (tonnes supplied at n1), f1 (tonnes shipped n1 to "
        "n2), xi1 (tonnes processed at n2), f2 (tonnes shipped n2 to n3), f3 (tonnes "
        "shipped n2 to n4), d1 (tonnes consumed at n3), d2 (tonnes consumed at n4)."
        "\n\n"
        "What is the dispatch that maximizes total social welfare?"
    )
    # Re-wrap each paragraph to fill box width (~155 chars at fontsize 5)
    query = "\n".join(
        textwrap.fill(para, 155) if para.strip() else ""
        for para in _query_raw.split("\n\n")
    )
    n_lines = query.count("\n") + 1
    bh = S_PAD + S_LINE + 0.001 + n_lines * 0.0021 + S_PAD
    _box(ax, LEFT, y, WIDTH, bh, C_USER)
    _t(ax, cl, y - S_PAD, "User Query  (Waste Market Dispatch)",
       fontsize=8, fontweight="bold", color=C_USER)
    _t(ax, cl, y - S_PAD - S_LINE - 0.001,
       query, fontsize=5, linespacing=1.3)
    y -= bh + S_GAP

    # ================================================================
    # 2. ROUTING
    # ================================================================
    bh = 0.015
    _box(ax, LEFT, y, WIDTH, bh, C_ROUTER)
    _t(ax, cl, y - 0.004, "OptimRoutingMiddleware",
       fontsize=7, fontweight="bold", color="#15803D")
    _t(ax, cl + 0.18, y - 0.004,
       "extract (||) -> validate -> critic -> adjudicator -> solve -> report",
       fontsize=5.5, color=C_DIM)
    y -= bh + S_GAP

    # ================================================================
    # 3. DUAL EXTRACTION
    # ================================================================
    cg = 0.008
    cw = (WIDTH - cg) / 2
    x1 = LEFT; x2 = LEFT + cw + cg
    col_h = 0.095

    # arrows
    akw = dict(arrowstyle="-|>,head_length=0.3,head_width=0.2",
               color=C_ROUTER, lw=1.5, connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(x1 + cw/2, y + 0.001),
                xytext=(MID, y + 0.008),
                arrowprops=akw, transform=ax.transAxes, zorder=6)
    ax.annotate("", xy=(x2 + cw/2, y + 0.001),
                xytext=(MID, y + 0.008),
                arrowprops=akw, transform=ax.transAxes, zorder=6)

    ext_lines = [
        "MAX 3500*d1 + 1*d2 - 2*s1",
        "    -5*f1 -5*f2 -5*f3 -20*xi1",
        "s1-f1=0       (n1/p1)",
        "f1-xi1=0      (n2/p1)",
        "0.01*xi1-f2=0 (n2/p2)",
        "0.99*xi1-f3=0 (n2/p3)",
        "f2-d1=0       (n3/p2)",
        "f3-d2=0       (n4/p3)",
        "+7 capacity bounds",
    ]

    for cx, name, model in [(x1, "run_extractor_a", "Gemini 2.5 Pro"),
                             (x2, "run_extractor_b", "Gemini 2.5 Flash")]:
        _box(ax, cx, y, cw, col_h, C_EXTRACT)
        il = cx + ACCENT + 0.006
        ir = cx + cw - 0.005
        cy = y - S_PAD
        _t(ax, il, cy, name, fontsize=7, fontweight="bold", color=C_EXTRACT)
        _t(ax, ir, cy, model, fontsize=5.5, color=C_DIM, ha="right")
        cy -= S_TITLE
        _pill(ax, il, cy, name, fs=5)
        cy -= S_PILL
        _div(ax, il, ir, cy)
        cy -= S_DIV
        _t(ax, il, cy, "Extracted LP (7 vars, 13 constr):",
           fontsize=5.5, fontweight="bold", color="#0D9488")
        cy -= S_LINE + 0.001
        for ln in ext_lines:
            _t(ax, il, cy, ln, fontsize=5, fontfamily="monospace")
            cy -= 0.005

    # agreement note
    _t(ax, x2 + ACCENT + 0.006, cy,
       "Both extractors agreed", fontsize=5.5, color=C_OK, fontweight="bold")

    y -= col_h + S_GAP

    # ================================================================
    # 4. VALIDATION
    # ================================================================
    bh = 0.082
    _box(ax, LEFT, y, WIDTH, bh, C_VALIDATE)
    cy = y - S_PAD
    _t(ax, cl, cy, "Step 2 — Formulation Validation",
       fontsize=8, fontweight="bold", color=C_VALIDATE)
    _t(ax, cr, cy, "deterministic", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE
    _pill(ax, cl, cy, "validate_formulation", fs=5)
    cy -= S_PILL
    _div(ax, cl, cr, cy)
    cy -= S_DIV + 0.002
    _t(ax, cl, cy, "passed=True  |  errors=0  |  warnings=1",
       fontsize=6, color=C_WARN, fontweight="bold")
    cy -= S_LINE + 0.002
    for tag, txt, col in [
        ("OK",   "Variables non-empty (7), objective has 7 terms", C_OK),
        ("OK",   "All variable refs valid in objective + constraints", C_OK),
        ("OK",   "Bounds valid (lb <= ub), no duplicate names", C_OK),
        ("WARN", "d2: obj coeff 1.0 == constr coeff 1.0 (false positive)", C_WARN),
    ]:
        _t(ax, cl + 0.004, cy, tag, fontsize=5.5, color=col,
           fontweight="bold", fontfamily="monospace")
        _t(ax, cl + 0.032, cy, txt, fontsize=5.5)
        cy -= S_LINE + 0.001
    y -= bh + S_GAP

    # ================================================================
    # 5. CRITIC
    # ================================================================
    bh = 0.074
    _box(ax, LEFT, y, WIDTH, bh, C_CRITIC)
    cy = y - S_PAD
    _t(ax, cl, cy, "Step 3 — Critic Review",
       fontsize=8, fontweight="bold", color=C_CRITIC)
    _t(ax, cr, cy, "Gemini 2.5 Pro", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE
    _pill(ax, cl, cy, "run_critic", fs=5)
    cy -= S_PILL
    _div(ax, cl, cr, cy)
    cy -= S_DIV + 0.002
    _t(ax, cl, cy, "Both extractors agreed — reconciliation confirmed",
       fontsize=6, color=C_OK, fontweight="bold")
    cy -= S_LINE + 0.002
    for txt in [
        "Obj: MAX | coeffs {d1:3500, d2:1, s1:-2, f1:-5, f2:-5, f3:-5, xi1:-20}",
        "6 equality + 7 capacity constraints: identical",
        "Variable types: all Continuous | all bounds match",
    ]:
        _t(ax, cl + 0.004, cy, "OK", fontsize=5, color=C_OK,
           fontweight="bold", fontfamily="monospace")
        _t(ax, cl + 0.025, cy, txt, fontsize=5)
        cy -= S_LINE + 0.001
    y -= bh + S_GAP

    # ================================================================
    # 6. ADJUDICATOR
    # ================================================================
    bh = 0.084
    _box(ax, LEFT, y, WIDTH, bh, C_ADJUD)
    cy = y - S_PAD
    _t(ax, cl, cy, "Step 4 — Adjudicator",
       fontsize=8, fontweight="bold", color="#0369A1")
    _t(ax, cr, cy, "Gemini 2.5 Flash", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE
    _pill(ax, cl, cy, "run_adjudicator", fs=5)
    cy -= S_PILL
    _div(ax, cl, cr, cy)
    cy -= S_DIV + 0.002
    _t(ax, cl, cy, "Coefficient Semantic Check:",
       fontsize=6, fontweight="bold", color="#0369A1")
    cy -= S_LINE + 0.002
    for txt in [
        "Obj coeffs = VALUE per unit (bids: 3500, 1, -2, -5, -20)",
        "Constr coeffs = FLOW (1, -1) + yields (0.01, 0.99)",
        "d2 warning dismissed: 1 USD/t is correct bid price",
    ]:
        _t(ax, cl + 0.004, cy, "OK", fontsize=5, color=C_OK,
           fontweight="bold", fontfamily="monospace")
        _t(ax, cl + 0.025, cy, txt, fontsize=5)
        cy -= S_LINE + 0.001
    cy -= 0.002
    _t(ax, cl, cy, "approved=true  |  issues=[]",
       fontsize=6, color=C_OK, fontweight="bold")
    y -= bh + S_GAP

    # ================================================================
    # 7. SOLVER
    # ================================================================
    bh = 0.098
    _box(ax, LEFT, y, WIDTH, bh, C_SOLVER)
    cy = y - S_PAD
    _t(ax, cl, cy, "Step 5 — Solver (PuLP/CBC)",
       fontsize=8, fontweight="bold", color="#334155")
    _t(ax, cr, cy, "deterministic", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE
    _pill(ax, cl, cy, "solve_formulation", fs=5)
    cy -= S_PILL
    _div(ax, cl, cr, cy)
    cy -= S_DIV + 0.002

    _t(ax, cl, cy, "Status: Optimal", fontsize=7, fontweight="bold", color=C_OK)
    _t(ax, cl + 0.14, cy, "Social Welfare: $31,920 USD",
       fontsize=7, fontweight="bold", color=C_OK)
    cy -= 0.008

    _t(ax, cl, cy, "Decision Variables:", fontsize=6, fontweight="bold", color="#334155")
    cy -= S_LINE + 0.002

    for left, right in [
        ("s1  = 8,000 t  (supply)",       "f3  = 7,920 t  (n2->n4)"),
        ("f1  = 8,000 t  (n1->n2)",       "d1  = 80 t     (demand n3)"),
        ("xi1 = 8,000 t  (processed)",     "d2  = 7,920 t  (demand n4)"),
        ("f2  = 80 t     (n2->n3)",        ""),
    ]:
        _t(ax, cl + 0.004, cy, left, fontsize=5, fontfamily="monospace")
        if right:
            _t(ax, cl + 0.36, cy, right, fontsize=5, fontfamily="monospace")
        cy -= 0.005

    cy -= 0.003
    _t(ax, cl, cy, "Binding (7):", fontsize=5.5, fontweight="bold", color=C_META)
    _t(ax, cl + 0.36, cy, "Non-Binding (6):", fontsize=5.5, fontweight="bold", color="#94A3B8")
    cy -= S_LINE + 0.001
    _t(ax, cl + 0.004, cy,
       "6 balance eqs + processing cap (xi1=8000)",
       fontsize=5, fontfamily="monospace", color=C_META)
    _t(ax, cl + 0.365, cy,
       "supply, demand n3/n4, 3 transport caps",
       fontsize=5, fontfamily="monospace", color="#94A3B8")

    y -= bh + S_GAP

    # ================================================================
    # 8. REPORT
    # ================================================================
    bh = 0.042
    _box(ax, LEFT, y, WIDTH, bh, C_REPORT)
    cy = y - S_PAD
    _t(ax, cl, cy, "Step 6 — Report Generation",
       fontsize=8, fontweight="bold", color=C_REPORT)
    _t(ax, cr, cy, "deterministic", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE
    _pill(ax, cl, cy, "generate_report", fs=5)
    cy -= S_PILL - 0.002
    _t(ax, cl + 0.003, cy,
       "Optimal | Welfare $31,920 | "
       "MAX 3500*d1+1*d2-2*s1-5*f1-5*f2-5*f3-20*xi1 | "
       "Bottleneck: xi1=8000",
       fontsize=5, fontfamily="monospace")

    y -= bh + S_GAP * 2

    # ================================================================
    # 9. PAPER COMPARISON
    # ================================================================
    bh = 0.082
    _box(ax, LEFT, y, WIDTH, bh, C_PAPER)
    cy = y - S_PAD
    _t(ax, cl, cy, "Verification Against Paper — Table S3, Scenario I",
       fontsize=8, fontweight="bold", color=C_PAPER)
    _t(ax, cr, cy, "Sampat et al. (2019)", fontsize=5.5, color=C_DIM, ha="right")
    cy -= S_TITLE + 0.002

    # Table header
    cols = [cl, cl+0.09, cl+0.18, cl+0.27, cl+0.38, cl+0.50, cl+0.64]
    for cx, h in zip(cols, ["", "s1", "f1", "xi1", "f2/d1", "f3/d2", "Welfare"]):
        _t(ax, cx, cy, h, fontsize=5.5, fontweight="bold", color="#475569", fontfamily="monospace")
    _div(ax, cl, cl + 0.78, cy - 0.004, color="#94A3B8")
    cy -= 0.008

    for label, color, fw in [("Paper", C_PAPER, "normal"),
                              ("Agent", C_OK, "bold"),
                              ("Match", C_OK, "bold")]:
        vals = {"Paper": ["8000","8000","8000","80","7920","$31,920"],
                "Agent": ["8000","8000","8000","80","7920","$31,920"],
                "Match": ["PASS","PASS","PASS","PASS","PASS","PASS"]}[label]
        _t(ax, cols[0], cy, label, fontsize=5.5, color=color,
           fontweight=fw, fontfamily="monospace")
        for cx, v in zip(cols[1:], vals):
            _t(ax, cx, cy, v, fontsize=5.5, color=color,
               fontweight=fw, fontfamily="monospace")
        cy -= 0.007

    cy -= 0.003
    _t(ax, cl, cy, "Clearing Prices (paper):", fontsize=5.5, fontweight="bold", color="#334155")
    cy -= S_LINE + 0.001
    _t(ax, cl + 0.003, cy,
       "pi(n1,p1)=2  pi(n2,p1)=7  pi(n2,p2)=3495  "
       "pi(n2,p3)=-4  pi(n3,p2)=3500  pi(n4,p3)=1",
       fontsize=4.5, fontfamily="monospace")
    cy -= S_LINE
    _t(ax, cl + 0.003, cy,
       "Transformation: pi_t = 3495*0.01 - 7 + (-4)*0.99 = 23.99 "
       "(> bid 20 -> profit $31,920)",
       fontsize=4.5, fontfamily="monospace", color=C_META)

    y -= bh + S_GAP * 2

    # ================================================================
    # 10. TIMELINE
    # ================================================================
    _t(ax, MID, y, "Execution Timeline",
       fontsize=8, fontweight="bold", color=C_TITLE, ha="center")
    y -= 0.006

    bl = LEFT + 0.06; bw = WIDTH - 0.07; bh_bar = 0.007
    total = 240.0

    def seg(yb, t0, dur, c, lab=""):
        x0 = bl + (t0/total)*bw
        w = max((dur/total)*bw, 0.002)
        ax.add_patch(FancyBboxPatch(
            (x0, yb), w, bh_bar,
            boxstyle="round,pad=0,rounding_size=0.001",
            facecolor=c, edgecolor="none", alpha=0.85,
            transform=ax.transAxes, zorder=3))
        if lab and w > 0.035:
            ax.text(x0+w/2, yb+bh_bar/2, lab, ha="center", va="center",
                    fontsize=4, color="white", fontweight="bold",
                    transform=ax.transAxes, zorder=4)

    bg = dict(boxstyle="round,pad=0,rounding_size=0.001",
              facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.5)
    ry = []
    for i, (lab, c) in enumerate([("LLM", C_EXTRACT), ("Det.", C_SOLVER), ("MW", C_ROUTER)]):
        by = y - (i+1) * (bh_bar + 0.003)
        ry.append(by)
        ax.add_patch(FancyBboxPatch((bl, by), bw, bh_bar,
                                    transform=ax.transAxes, zorder=2, **bg))
        ax.text(bl - 0.004, by + bh_bar/2, lab, ha="right", va="center",
                fontsize=5, color=c, fontweight="bold", transform=ax.transAxes)

    seg(ry[0], 0, 60, C_EXTRACT, "extractors")
    seg(ry[0], 80, 40, C_CRITIC, "critic")
    seg(ry[0], 130, 40, C_ADJUD, "adjudicator")
    seg(ry[1], 65, 5, C_VALIDATE)
    seg(ry[1], 175, 5, C_SOLVER)
    seg(ry[1], 185, 10, C_REPORT)
    seg(ry[2], 0, 231, C_ROUTER, "routing + guardrails (~231s)")

    ty = ry[2] - 0.006
    for t in [0, 60, 120, 180, 240]:
        _t(ax, bl + (t/total)*bw, ty, f"{t}s", fontsize=5, ha="center")

    y = ty - 0.009

    # ================================================================
    # 11. BOTTOM PANELS
    # ================================================================
    ph = 0.046; pg = 0.006
    pw = (WIDTH - 2*pg) / 3
    px = [LEFT, LEFT+pw+pg, LEFT+2*(pw+pg)]
    pcl = ACCENT + 0.005

    # Panel 1: Metadata
    _box(ax, px[0], y, pw, ph, C_USER)
    _t(ax, px[0]+pw/2, y-0.004, "Trace Metadata",
       fontsize=6.5, fontweight="bold", color=C_TITLE, ha="center")
    my = y - 0.013
    for lab, val in [("Run Time","~231s"), ("Tool Calls","7"),
                     ("LLM Calls","4"), ("Det. Calls","3"),
                     ("Variables","7 Cont."), ("Constraints","13")]:
        _t(ax, px[0]+pcl, my, lab, fontsize=5, va="center")
        _t(ax, px[0]+pw-0.004, my, val, fontsize=5, fontweight="bold",
           color=C_TITLE, ha="right", va="center")
        my -= 0.005

    # Panel 2: Results
    _box(ax, px[1], y, pw, ph, C_REPORT)
    _t(ax, px[1]+pw/2, y-0.004, "Key Results",
       fontsize=6.5, fontweight="bold", color=C_TITLE, ha="center")
    my = y - 0.013
    for ln in ["Welfare = $31,920", "Bottleneck: tech cap",
               "xi1=8000 (max 8000)", "80t p2, 7920t p3",
               "Tech profit: $31,920", "Others: $0"]:
        _t(ax, px[1]+pcl, my, ln, fontsize=5, va="center")
        my -= 0.005

    # Panel 3: Coeff Routing
    _box(ax, px[2], y, pw, ph, C_META)
    _t(ax, px[2]+pw/2, y-0.004, "Coefficient Routing",
       fontsize=6.5, fontweight="bold", color=C_TITLE, ha="center")
    my = y - 0.013
    for tag, desc in [("L1","Scratchpad"), ("L2","Schema sep."),
                      ("L3","Label audit"), ("L4","Det. validator"),
                      ("L5","Dual+critic"), ("L6","Adjud. arith.")]:
        _t(ax, px[2]+pcl, my, tag, fontsize=5, color=C_META,
           fontweight="bold", fontfamily="monospace", va="center")
        _t(ax, px[2]+pcl+0.018, my, desc, fontsize=5, va="center")
        my -= 0.005

    y -= ph + 0.004

    # Footer
    _t(ax, MID, y,
       "optim-agent v0.1.0 | Waste Market (Sampat et al. 2019) | "
       "extract->validate->critic->adjudicator->solve->report",
       fontsize=5.5, color="#94A3B8", ha="center")

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "visualize_waste_market_trace.png")
    fig.savefig(out, dpi=300, facecolor=C_BG,
                bbox_inches="tight", pad_inches=0.08)
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    create_trace_figure()
