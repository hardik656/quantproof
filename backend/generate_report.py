"""
QuantProof PDF Report Generator v1.2
Synced to validator v6.2

v1.2 changes:
  - n_trials depth shown in report header (Standard / Institutional)
  - DSR check row highlighted with n_trials context note
  - Footer shows validation depth used
  - Depth badge on cover page
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Wedge, Polygon
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker
import io, datetime, math

# Brand
C_BLACK  = colors.HexColor("#0A0A0A")
C_WHITE  = colors.white
C_ACCENT = colors.HexColor("#00C896")
C_ORANGE = colors.HexColor("#FF6B35")
C_DANGER = colors.HexColor("#FF3B30")
C_WARN   = colors.HexColor("#FF9F0A")
C_OK     = colors.HexColor("#30D158")
C_DARK   = colors.HexColor("#1C1C1E")
C_MID    = colors.HexColor("#3A3A3C")
C_LITE   = colors.HexColor("#8E8E93")
C_PAPER  = colors.HexColor("#F2F2F7")
C_PAPER2 = colors.HexColor("#E5E5EA")
C_INST   = colors.HexColor("#7B61FF")   # purple for Institutional depth badge

PAGE_W, PAGE_H = A4
MARGIN  = 18 * mm
INNER_W = PAGE_W - 2 * MARGIN


def _s(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=10, leading=14,
                    textColor=C_DARK, spaceAfter=0, spaceBefore=0)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


def make_styles():
    return {
        "h_section":  _s("h_section",  fontName="Helvetica-Bold", fontSize=12,
                          textColor=C_BLACK, spaceBefore=12, spaceAfter=5),
        "body":       _s("body",        leading=15, alignment=TA_JUSTIFY,
                          textColor=C_DARK, spaceAfter=4),
        "body_sm":    _s("body_sm",     fontSize=9, leading=13, textColor=C_MID),
        "label":      _s("label",       fontName="Helvetica-Bold", fontSize=8,
                          textColor=C_LITE, leading=10),
        "big_num":    _s("big_num",     fontName="Helvetica-Bold", fontSize=22,
                          leading=26, textColor=C_BLACK),
        "grade":      _s("grade",       fontName="Helvetica-Bold", fontSize=18,
                          leading=22, textColor=C_BLACK),
        "insight":    _s("insight",     fontSize=9, leading=13,
                          fontName="Helvetica-Oblique", textColor=C_MID, leftIndent=8),
        "fix_txt":    _s("fix_txt",     fontSize=9, leading=13,
                          textColor=C_ORANGE, leftIndent=8),
        "footer":     _s("footer",      fontSize=7.5, leading=11,
                          textColor=C_LITE, alignment=TA_CENTER),
        "crash_name": _s("crash_name",  fontName="Helvetica-Bold", fontSize=10,
                          textColor=C_BLACK),
        "verdict":    _s("verdict",     fontSize=9, leading=13,
                          fontName="Helvetica-Oblique", textColor=C_MID),
        "num_badge":  _s("num_badge",   fontName="Helvetica-Bold", fontSize=13,
                          textColor=C_ACCENT, alignment=TA_CENTER),
        "action_h":   _s("action_h",    fontName="Helvetica-Bold", fontSize=10,
                          textColor=C_BLACK),
        "action_b":   _s("action_b",    fontSize=9, leading=13, textColor=C_DARK,
                          alignment=TA_JUSTIFY),
        "watermark":  _s("watermark",   fontName="Helvetica-Bold", fontSize=8,
                          textColor=C_ACCENT, alignment=TA_RIGHT),
        "depth_note": _s("depth_note",  fontSize=8, leading=12,
                          textColor=C_INST, fontName="Helvetica-Oblique"),
    }


def _depth_label(n_trials: int) -> tuple:
    """Return (short_label, description, color) for a given n_trials value."""
    if n_trials <= 1:
        return "QUICK (n=1)", "Rapid scan — less rigorous", C_LITE
    elif n_trials <= 10:
        return f"STANDARD (n={n_trials})", "Standard free-tier depth", C_ACCENT
    elif n_trials <= 20:
        return f"INSTITUTIONAL (n={n_trials})", "Institutional depth — fund-grade stress test", C_INST
    else:
        return f"DEEP INSTITUTIONAL (n={n_trials})", f"Deep stress test — {n_trials}x Monte Carlo simulations", C_INST


# Score gauge
def score_gauge(score: float, grade_short: str, width=130, height=80) -> Drawing:
    d  = Drawing(width, height)
    cx = width / 2
    cy = 10
    R  = 54
    arcs = [(0, 40, C_DANGER), (40, 70, C_WARN), (70, 88, C_OK), (88, 100, C_ACCENT)]
    for s0, s1, col in arcs:
        steps = max(3, int((s1 - s0) * 1.8))
        for i in range(steps):
            a0 = math.radians(180 - (s0 + (s1-s0)*i/steps) * 1.8)
            a1 = math.radians(180 - (s0 + (s1-s0)*(i+1)/steps) * 1.8)
            x0 = cx + R * math.cos(a0); y0 = cy + R * math.sin(a0)
            x1 = cx + R * math.cos(a1); y1 = cy + R * math.sin(a1)
            ln = Line(x0, y0, x1, y1)
            ln.strokeColor = col; ln.strokeWidth = 9
            d.add(ln)
    na = math.radians(180 - score * 1.8)
    nx = cx + (R - 6) * math.cos(na); ny = cy + (R - 6) * math.sin(na)
    needle = Line(cx, cy, nx, ny)
    needle.strokeColor = C_BLACK; needle.strokeWidth = 2.2
    d.add(needle)
    hub = Wedge(cx, cy, 5, 0, 360)
    hub.fillColor = C_BLACK; hub.strokeColor = None
    d.add(hub)
    d.add(String(cx, cy + 22, f"{score:.0f}", fontName="Helvetica-Bold", fontSize=24,
                 fillColor=C_BLACK, textAnchor="middle"))
    d.add(String(cx, cy + 11, grade_short, fontName="Helvetica", fontSize=8,
                 fillColor=C_LITE, textAnchor="middle"))
    return d


def score_bar(score: float, width=50, height=7) -> Drawing:
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=C_PAPER2, strokeColor=None))
    col = C_DANGER if score < 40 else C_WARN if score < 70 else C_OK if score < 88 else C_ACCENT
    d.add(Rect(0, 0, score / 100 * width, height, fillColor=col, strokeColor=None))
    return d


def equity_curve_chart(equity: list, width=None, height=110) -> Drawing:
    w = width or INNER_W * 0.72
    if not equity or len(equity) < 3:
        d = Drawing(w, height)
        d.add(String(w / 2, height / 2, "No equity curve data",
                     fontName="Helvetica", fontSize=8,
                     fillColor=C_LITE, textAnchor="middle"))
        return d

    d   = Drawing(w, height)
    n   = len(equity)
    pad = dict(l=8, r=6, t=10, b=18)
    aw  = w - pad["l"] - pad["r"]
    ah  = height - pad["t"] - pad["b"]
    mn  = min(equity)
    mx  = max(equity)
    rng = mx - mn or 1.0

    d.add(Rect(pad["l"], pad["b"], aw, ah,
               fillColor=colors.HexColor("#F8F8FA"), strokeColor=C_PAPER2, strokeWidth=0.5))

    if mn <= 100 <= mx:
        y100 = pad["b"] + (100 - mn) / rng * ah
        ln = Line(pad["l"], y100, pad["l"] + aw, y100)
        ln.strokeColor = C_PAPER2; ln.strokeWidth = 0.8; ln.strokeDashArray = [3, 3]
        d.add(ln)

    pts = [(pad["l"] + i / (n - 1) * aw,
            pad["b"] + (equity[i] - mn) / rng * ah) for i in range(n)]

    fill_pts = [pad["l"], pad["b"]]
    for px, py in pts:
        fill_pts += [px, py]
    fill_pts += [pad["l"] + aw, pad["b"]]
    poly = Polygon(fill_pts)
    end_col = C_OK if equity[-1] >= equity[0] else C_DANGER
    poly.fillColor  = colors.Color(end_col.red, end_col.green, end_col.blue, alpha=0.10)
    poly.strokeColor = None
    d.add(poly)

    for i in range(n - 1):
        seg = Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        seg.strokeColor = end_col
        seg.strokeWidth = 1.4
        d.add(seg)

    for val, label in [(mn, f"{mn:.0f}"), (100, "100"), (mx, f"{mx:.0f}")]:
        if mn == mx:
            break
        yy = pad["b"] + (val - mn) / rng * ah
        d.add(String(pad["l"] - 2, yy - 3, label,
                     fontName="Helvetica", fontSize=6,
                     fillColor=C_LITE, textAnchor="end"))

    for idx, label in [(0, "T1"), (n // 2, f"T{n//2}"), (n - 1, f"T{n}")]:
        xx = pad["l"] + idx / max(n - 1, 1) * aw
        d.add(String(xx, pad["b"] - 10, label,
                     fontName="Helvetica", fontSize=6,
                     fillColor=C_LITE, textAnchor="middle"))

    final_x = pad["l"] + aw + 2
    final_y = pts[-1][1] - 3
    d.add(String(final_x, final_y, f"{equity[-1]:.1f}",
                 fontName="Helvetica-Bold", fontSize=7,
                 fillColor=end_col, textAnchor="start"))

    return d


def crash_bar_chart(crashes: list, width=None, height=110) -> Drawing:
    w = width or INNER_W
    if not crashes:
        return Drawing(w, height)

    d   = Drawing(w, height)
    n   = len(crashes)
    pad = dict(l=32, r=10, t=8, b=28)
    aw  = w - pad["l"] - pad["r"]
    ah  = height - pad["t"] - pad["b"]
    bw  = aw / n
    bw2 = bw * 0.32
    gap = bw * 0.08

    all_vals = [c.get("market_drop", 0) for c in crashes] + \
               [c.get("strategy_drop", 0) * 100 for c in crashes]
    mn = min(all_vals + [0])
    mx = max(all_vals + [0])
    rng = mx - mn or 1.0

    def y_of(v):
        return pad["b"] + (v - mn) / rng * ah

    y0 = y_of(0)
    zl = Line(pad["l"], y0, pad["l"] + aw, y0)
    zl.strokeColor = C_MID; zl.strokeWidth = 0.6
    d.add(zl)

    for tick_v in [mn, mn / 2, 0]:
        yt = y_of(tick_v)
        tl = Line(pad["l"] - 3, yt, pad["l"], yt)
        tl.strokeColor = C_LITE; tl.strokeWidth = 0.5
        d.add(tl)
        d.add(String(pad["l"] - 4, yt - 3, f"{tick_v:.0f}%",
                     fontName="Helvetica", fontSize=6,
                     fillColor=C_LITE, textAnchor="end"))

    for i, crash in enumerate(crashes):
        cx   = pad["l"] + i * bw + bw * 0.1
        mktd = crash.get("market_drop", 0)
        strd = crash.get("strategy_drop", 0) * 100
        surv = crash.get("survived", False)

        mkt_y0 = y_of(min(mktd, 0))
        mkt_h  = abs(y_of(mktd) - y0)
        mr = Rect(cx, mkt_y0, bw2, mkt_h)
        mr.fillColor  = colors.Color(C_DANGER.red, C_DANGER.green, C_DANGER.blue, alpha=0.7)
        mr.strokeColor = None
        d.add(mr)

        str_col = C_OK if surv else C_WARN
        str_y   = y_of(min(strd, 0))
        str_h   = abs(y_of(strd) - y0)
        sr = Rect(cx + bw2 + gap, str_y, bw2, max(str_h, 1))
        sr.fillColor  = colors.Color(str_col.red, str_col.green, str_col.blue, alpha=0.75)
        sr.strokeColor = None
        d.add(sr)

        year_raw = crash.get("year", "")
        label    = year_raw.split("-")[0].split(" ")[-1][:4]
        d.add(String(cx + bw2 + gap / 2, pad["b"] - 14, label,
                     fontName="Helvetica", fontSize=6.5,
                     fillColor=C_DARK, textAnchor="middle"))

        lbl_y = y_of(strd) + (3 if strd >= 0 else -10)
        d.add(String(cx + bw2 + gap / 2, lbl_y, f"{strd:+.0f}%",
                     fontName="Helvetica-Bold", fontSize=6,
                     fillColor=str_col, textAnchor="middle"))

    lx = pad["l"]
    ly = pad["b"] - 24
    mr2 = Rect(lx, ly + 2, 8, 6)
    mr2.fillColor = colors.Color(C_DANGER.red, C_DANGER.green, C_DANGER.blue, alpha=0.7)
    mr2.strokeColor = None
    d.add(mr2)
    d.add(String(lx + 10, ly + 2, "Market",
                 fontName="Helvetica", fontSize=6.5, fillColor=C_LITE))
    sr2 = Rect(lx + 50, ly + 2, 8, 6)
    sr2.fillColor = colors.Color(C_OK.red, C_OK.green, C_OK.blue, alpha=0.75)
    sr2.strokeColor = None
    d.add(sr2)
    d.add(String(lx + 62, ly + 2, "Your Strategy",
                 fontName="Helvetica", fontSize=6.5, fillColor=C_LITE))

    return d


def category_radar_chart(checks: list, width=120, height=120) -> Drawing:
    cats    = ["Overfitting", "Risk", "Regime", "Execution", "Compliance"]
    scores  = {}
    for cat in cats:
        cc = [c for c in checks if
              (c.get("category") if isinstance(c, dict) else getattr(c, "category", "")) == cat]
        if cc:
            scores[cat] = sum(
                (c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0))
                for c in cc
            ) / len(cc)
        else:
            scores[cat] = 0

    d  = Drawing(width, height)
    cx = width / 2
    cy = height / 2 - 6
    R  = min(width, height) / 2 - 18
    n  = len(cats)

    def pt(i, r_frac):
        ang = math.pi / 2 + 2 * math.pi * i / n
        return (cx + r_frac * R * math.cos(ang),
                cy + r_frac * R * math.sin(ang))

    for frac in [0.25, 0.5, 0.75, 1.0]:
        ring_pts = []
        for i in range(n):
            px, py = pt(i, frac)
            ring_pts += [px, py]
        ring = Polygon(ring_pts)
        ring.fillColor   = None
        ring.strokeColor = C_PAPER2
        ring.strokeWidth = 0.5
        d.add(ring)

    for i in range(n):
        px, py = pt(i, 1.0)
        spoke = Line(cx, cy, px, py)
        spoke.strokeColor = C_PAPER2; spoke.strokeWidth = 0.5
        d.add(spoke)

    score_pts = []
    for i, cat in enumerate(cats):
        frac = scores[cat] / 100
        px, py = pt(i, frac)
        score_pts += [px, py]
    spoly = Polygon(score_pts)
    spoly.fillColor  = colors.Color(C_ACCENT.red, C_ACCENT.green, C_ACCENT.blue, alpha=0.18)
    spoly.strokeColor = C_ACCENT
    spoly.strokeWidth = 1.4
    d.add(spoly)

    for i, cat in enumerate(cats):
        frac = scores[cat] / 100
        px, py = pt(i, frac)
        dot = Wedge(px, py, 3, 0, 360)
        dot.fillColor  = C_ACCENT; dot.strokeColor = None
        d.add(dot)

    label_offsets = {0: (0, 6), 1: (8, 0), 2: (6, -8), 3: (-6, -8), 4: (-8, 0)}
    for i, cat in enumerate(cats):
        lx, ly = pt(i, 1.15)
        ox, oy = label_offsets.get(i, (0, 0))
        score_val = scores[cat]
        col = cat_color(score_val)
        d.add(String(lx + ox, ly + oy + 8, f"{cat[:5]}",
                     fontName="Helvetica-Bold", fontSize=6.5,
                     fillColor=col, textAnchor="middle"))
        d.add(String(lx + ox, ly + oy, f"{score_val:.0f}",
                     fontName="Helvetica", fontSize=6,
                     fillColor=col, textAnchor="middle"))

    return d


def footer_canvas(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(C_PAPER2)
    canvas.line(MARGIN, 15*mm, PAGE_W - MARGIN, 15*mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_LITE)
    # Include depth in footer
    depth_str = getattr(footer_canvas, '_depth_str', '')
    canvas.drawCentredString(PAGE_W/2, 10*mm,
        f"QuantProof Institutional Analysis  |  quantproof-frontend.onrender.com  |  "
        f"{datetime.date.today().strftime('%d %B %Y')}  |  {depth_str}  |  Page {doc.page}")
    canvas.restoreState()


def grade_color(grade):
    if grade.startswith("A"):  return C_OK
    if "B+" in grade:          return C_ACCENT
    if grade.startswith("B"):  return C_WARN
    return C_DANGER


def cat_color(s):
    if s >= 88: return C_ACCENT
    if s >= 70: return C_OK
    if s >= 50: return C_WARN
    return C_DANGER


def _clean_text(raw: str) -> str:
    v = raw or ""
    v = v.replace("[UNCALIBRATED] ", "").replace("[UNCALIBRATED]", "")
    for e in ("🟢", "🟡", "🟠", "🔴"):
        v = v.replace(e, "")
    v = v.replace("✅", "PASS").replace("✓", "OK").replace("✗", "FAIL").replace("❌", "FAIL")
    v = v.replace("📊", "").replace("ℹ", "").replace("⚠", "WARNING:").replace("📈", "")
    v = v.replace("α", "alpha").replace("β", "beta").replace("Δ", "delta")
    v = v.replace("≥", ">=").replace("≤", "<=").replace("≈", "~").replace("→", "->").replace("←", "<-")
    v = v.replace("×", "x").replace("÷", "/").replace("±", "+/-")
    v = v.replace("σ", "sigma")
    v = v.replace("—", " - ").replace("–", "-").replace("‒", "-")
    v = v.replace("\u2018", "'").replace("\u2019", "'")
    v = v.replace("\u201c", '"').replace("\u201d", '"')
    while "  " in v:
        v = v.replace("  ", " ")
    v = v.encode("latin-1", errors="replace").decode("latin-1")
    return v.strip()


def _clean_verdict(raw: str) -> str:
    return _clean_text(raw)


def _get(obj, key, default=""):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def generate_pdf_report(vr: dict) -> bytes:
    buf = io.BytesIO()
    S   = make_styles()

    score     = float(vr.get("fundable_score", 0))
    grade     = vr.get("grade", "F - Do Not Deploy")
    sharpe    = float(vr.get("sharpe", 0))
    dd        = float(vr.get("max_drawdown", 0)) * 100
    wr        = float(vr.get("win_rate", 0))
    n         = int(vr.get("total_trades", 0))
    dr        = vr.get("date_range", "N/A")
    checks    = vr.get("checks", [])
    crashes   = vr.get("crash_sims", [])
    issues    = vr.get("top_issues", [])
    strengths = vr.get("top_strengths", [])

    # n_trials depth — read from vr, fall back to 10
    n_trials   = int(vr.get("n_trials", 10))
    depth_short, depth_desc, depth_col = _depth_label(n_trials)

    # Store depth string for footer
    footer_canvas._depth_str = depth_short

    grade_short = grade.split("-")[0].strip().split(" ")[0]
    gc = grade_color(grade)

    import sys
    print(f"[PDF] score={score} grade={grade} n_trials={n_trials} depth={depth_short}", file=sys.stderr, flush=True)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16*mm, bottomMargin=22*mm,
        title="QuantProof Analysis Report",
    )
    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    is_verified  = vr.get("is_verified", False)
    report_label = "VERIFIED INSTITUTIONAL REPORT" if is_verified else "METHODOLOGY ANALYSIS REPORT"

    hdr = Table([[
        Paragraph("QUANTPROOF", _s("qplogo", fontName="Helvetica-Bold",
            fontSize=11, textColor=C_ACCENT, leading=14)),
        Paragraph(report_label, _s("rptlabel", fontName="Helvetica", fontSize=9,
            textColor=C_LITE, alignment=TA_RIGHT, leading=14)),
    ]], colWidths=[INNER_W*0.5, INNER_W*0.5])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_BLACK),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
    ]))
    story.append(hdr)

    # ── DEPTH BADGE STRIP ─────────────────────────────────────────────────────
    depth_bg = colors.HexColor("#1A1040") if n_trials > 10 else colors.HexColor("#0A1A14")
    depth_strip = Table([[
        Paragraph(
            f"VALIDATION DEPTH: {depth_short}",
            _s("ds", fontName="Helvetica-Bold", fontSize=8,
               textColor=depth_col, leading=11)),
        Paragraph(
            depth_desc,
            _s("dd", fontName="Helvetica", fontSize=8,
               textColor=C_LITE, alignment=TA_RIGHT, leading=11)),
    ]], colWidths=[INNER_W*0.55, INNER_W*0.45])
    depth_strip.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), depth_bg),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
    ]))
    story.append(depth_strip)
    story.append(Spacer(1, 6*mm))

    # ── SCORE + GRADE ─────────────────────────────────────────────────────────
    verdict_text = {
        "A":  ("Institutionally Viable",
               f"Passed all critical institutional checks. Sharpe {sharpe:.2f}, "
               f"max drawdown {dd:.1f}%. "
               f"{'Connect a broker account for a verified compliance certificate.' if not is_verified else 'Eligible for prop firm submission.'}"),
        "B+": ("Prop Firm Ready",
               f"Core edge is real and scalable. Some execution or regime checks need refinement."),
        "B":  ("Live Tradeable - With Caution",
               f"Real but fragile edge (Sharpe {sharpe:.2f}). Deploy at reduced size and monitor closely."),
        "C":  ("Promising - Needs Work",
               f"Early-stage edge (Sharpe {sharpe:.2f}) but fails key institutional checks. "
               f"Fix the ranked issues below before deploying real capital."),
        "D":  ("Major Issues",
               f"Critical failures across key institutional checks. See the action plan below."),
        "F":  ("Do Not Deploy",
               f"Fails institutional standards. Do not deploy capital until major issues are resolved."),
    }
    tier = ("A"  if score >= 90 else "B+" if score >= 80 else "B"  if score >= 70 else
            "C"  if score >= 50 else "D"  if score >= 35 else "F")
    verdict_title, verdict_body = verdict_text.get(tier, ("Unknown", "Unable to determine verdict."))

    score_row = Table([[
        score_gauge(score, grade_short),
        [
            Paragraph("OVERALL VERDICT", S["label"]),
            Spacer(1, 4),
            Paragraph(grade, _s("grd", fontName="Helvetica-Bold",
                fontSize=17, leading=21, textColor=gc)),
            Spacer(1, 5),
            Paragraph(verdict_body, S["body"]),
            Spacer(1, 6),
            # Depth context note under verdict
            Paragraph(
                f"Validated at {depth_short}. "
                + (f"DSR and overfitting checks used {n_trials} Monte Carlo simulations. "
                   "This is institutional-grade depth." if n_trials > 10
                   else f"DSR used {n_trials} Monte Carlo simulation(s). "
                   "Upgrade to Institutional depth (n=20+) to stress-test further."),
                S["depth_note"]),
        ]
    ]], colWidths=[46*mm, INNER_W - 46*mm])
    score_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (1,0),(1,0),   8),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(score_row)
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
    story.append(Spacer(1, 4*mm))

    # ── KEY METRICS ───────────────────────────────────────────────────────────
    story.append(Paragraph("KEY METRICS", S["label"]))
    story.append(Spacer(1, 3))

    def metric_cell(lbl, val, sub):
        return Table([[Paragraph(lbl, S["label"])],
                      [Paragraph(val, _s("mv", fontName="Helvetica-Bold",
                          fontSize=19, leading=23, textColor=C_BLACK))],
                      [Paragraph(sub, S["body_sm"])]],
                     colWidths=[36*mm])

    metrics_row = Table([[
        metric_cell("SHARPE RATIO",     f"{sharpe:.2f}", "Target > 1.5"),
        metric_cell("MAX DRAWDOWN",     f"{dd:.1f}%",    "Target < 20%"),
        metric_cell("WIN RATE",         f"{wr:.1f}%",    "Target > 50%"),
        metric_cell("TOTAL TRADES",     str(n),          dr[:18] if dr != "N/A" else "No dates"),
    ]], colWidths=[INNER_W/4]*4)
    metrics_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_PAPER),
        ("TOPPADDING",    (0,0),(-1,-1), 9),
        ("BOTTOMPADDING", (0,0),(-1,-1), 9),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("LINEAFTER",     (0,0),(2,0),   0.6, C_PAPER2),
    ]))
    story.append(metrics_row)
    story.append(Spacer(1, 6*mm))

    # ── EQUITY CURVE ──────────────────────────────────────────────────────────
    equity_data = vr.get("equity_curve")
    if equity_data:
        try:
            equity_data = [float(x) for x in equity_data]
            if len(equity_data) < 3:
                equity_data = None
        except (TypeError, ValueError):
            equity_data = None

    if equity_data and len(equity_data) > 5:
        story.append(Paragraph("EQUITY CURVE", S["label"]))
        story.append(Spacer(1, 3))
        is_real = True
        curve_note = "actual trade-by-trade equity curve from validator"
    else:
        import random as _random
        _random.seed(abs(hash(str(sharpe) + str(n))) % (2**31))
        eq = 100.0
        equity_data = [100.0]
        mean_r = sharpe * 0.015 / max((252 / max(n, 1)) ** 0.5, 0.1)
        std_r  = abs(mean_r / (sharpe + 1e-9))
        for _ in range(min(n, 200) - 1):
            r = _random.gauss(mean_r, std_r)
            eq *= (1 + max(-0.99, min(0.99, r)))
            equity_data.append(round(eq, 4))
        is_real = False
        curve_note = f"illustrative - derived from Sharpe {sharpe:.2f} (upload dated CSV for real curve)"
        story.append(Paragraph("EQUITY CURVE (ILLUSTRATIVE)", S["label"]))
        story.append(Spacer(1, 3))

    final_return = (equity_data[-1] / equity_data[0] - 1) * 100 if equity_data[0] else 0
    eq_stats_col = [
        Paragraph("TOTAL RETURN", S["label"]),
        Paragraph(f"{'+'if final_return>=0 else ''}{final_return:.1f}%",
                  _s("eq_ret", fontName="Helvetica-Bold", fontSize=16,
                     textColor=C_OK if final_return >= 0 else C_DANGER, leading=20)),
        Spacer(1, 6),
        Paragraph("DATA SOURCE", S["label"]),
        Paragraph("Real trades" if is_real else "Synthetic",
                  _s("eq_src", fontSize=8, textColor=C_LITE, leading=11)),
        Spacer(1, 4),
        Paragraph(curve_note, _s("eq_note", fontSize=7, textColor=C_LITE,
                                  leading=10, fontName="Helvetica-Oblique")),
    ]
    eq_tbl = Table([[equity_curve_chart(equity_data, width=INNER_W*0.72), eq_stats_col]],
                   colWidths=[INNER_W*0.72, INNER_W*0.28])
    eq_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("BACKGROUND",    (0,0),(-1,-1), C_PAPER),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    story.append(eq_tbl)
    story.append(Spacer(1, 6*mm))

    if issues or strengths:
        col_a, col_b = [], []
        if issues:
            col_a.append(Paragraph("PRIMARY ISSUES", S["label"]))
            col_a.append(Spacer(1, 3))
            for iss in issues[:4]:
                col_a.append(Paragraph(f"- {_clean_text(iss)[:110]}", S["body_sm"]))
        if strengths:
            col_b.append(Paragraph("CONFIRMED STRENGTHS", S["label"]))
            col_b.append(Spacer(1, 3))
            for st in strengths[:3]:
                col_b.append(Paragraph(f"- {_clean_text(st)}", S["body_sm"]))
        if col_a or col_b:
            summ_row = Table([[col_a or [""], col_b or [""]]],
                             colWidths=[INNER_W*0.52, INNER_W*0.48])
            summ_row.setStyle(TableStyle([
                ("VALIGN",       (0,0),(-1,-1), "TOP"),
                ("BACKGROUND",   (0,0),(0,0),   colors.HexColor("#FFF4F0")),
                ("BACKGROUND",   (1,0),(1,0),   colors.HexColor("#F0FBF5")),
                ("TOPPADDING",   (0,0),(-1,-1), 9),
                ("BOTTOMPADDING",(0,0),(-1,-1), 9),
                ("LEFTPADDING",  (0,0),(-1,-1), 10),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
            ]))
            story.append(summ_row)
            story.append(Spacer(1, 6*mm))

    # ── AUDIT TRAIL ───────────────────────────────────────────────────────────
    audit_flags = vr.get("audit_flags", [])
    display_flags = [
        f for f in audit_flags
        if f and f.strip() and len(f.strip()) > 5
        and not f.strip().startswith("   -")
    ][:8]

    if display_flags:
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("DATA & VALIDATION NOTES", S["h_section"]))
        flag_rows = []
        for flag in display_flags:
            clean_flag = _clean_text(flag)[:160]
            flag_rows.append([Paragraph(clean_flag, S["body_sm"])])
        flag_tbl = Table(flag_rows, colWidths=[INNER_W])
        flag_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_PAPER),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("LINEBELOW",     (0,0),(-1,-2), 0.3, C_PAPER2),
        ]))
        story.append(flag_tbl)
        story.append(Spacer(1, 4*mm))

    # ── CRASH SIMS ────────────────────────────────────────────────────────────
    if crashes:
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("HISTORICAL CRASH STRESS TESTS", S["h_section"]))
        story.append(Paragraph(
            "Strategy stress-tested against five historical crash scenarios using "
            "a 60-day crisis window with Almgren-Chriss volatility scaling and liquidity drag.",
            S["body"]))
        story.append(Spacer(1, 4))

        survived_count = sum(1 for c in crashes if c.get("survived"))
        story.append(Paragraph(
            f"Crash survival: {survived_count}/5 - "
            + ("All five survived. Strong crisis resilience." if survived_count == 5 else
               "Four survived. Minor vulnerability." if survived_count == 4 else
               "Three survived. Moderate crisis risk." if survived_count == 3 else
               "Two survived. Significant vulnerability." if survived_count == 2 else
               "One survived. Extremely high crisis risk." if survived_count == 1 else
               "Failed all five. Critical crisis exposure."),
            _s("crash_sum", fontName="Helvetica-Bold", fontSize=10,
               textColor=C_BLACK, spaceAfter=6)))
        story.append(Spacer(1, 4))

        for crash in crashes:
            surv    = crash.get("survived", False)
            drop    = crash.get("strategy_drop", 0) * 100
            name    = crash.get("crash_name", "")
            year    = crash.get("year", "")
            mkt_d   = crash.get("market_drop", 0)
            desc    = crash.get("description", "")[:100]
            verdict = _clean_verdict(crash.get("emotional_verdict", ""))
            dd_breach = not surv and drop > 0
            bg_row  = (colors.HexColor("#EDF9F3") if surv else
                       colors.HexColor("#FFF8EC") if dd_breach else
                       colors.HexColor("#FDF0EF"))
            tag_col = C_OK if surv else (C_WARN if dd_breach else C_DANGER)
            tag_txt = "SURVIVED" if surv else ("DD BREACH" if dd_breach else "BLOWN UP")

            c_tbl = Table([
                [
                    Paragraph(f"{name}  {year}", S["crash_name"]),
                    Paragraph(tag_txt, _s("tag", fontName="Helvetica-Bold",
                        fontSize=10, textColor=tag_col, alignment=TA_RIGHT)),
                ],
                [
                    Paragraph(f"Market: {mkt_d:.0f}%  Strategy: {drop:+.1f}%  {desc}", S["body_sm"]),
                    Paragraph("", S["body_sm"]),
                ],
                [
                    Paragraph(verdict[:180], S["verdict"]),
                    Paragraph("", S["body_sm"]),
                ],
            ], colWidths=[INNER_W*0.72, INNER_W*0.28])
            c_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), bg_row),
                ("TOPPADDING",    (0,0),(-1,-1), 7),
                ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                ("LEFTPADDING",   (0,0),(-1,-1), 10),
                ("RIGHTPADDING",  (0,0),(-1,-1), 10),
                ("SPAN",          (0,1),(1,1)),
                ("SPAN",          (0,2),(1,2)),
                ("LINEBELOW",     (0,0),(-1,0),  0.5, C_PAPER2),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            story.append(c_tbl)
            story.append(Spacer(1, 3))

        story.append(Spacer(1, 4))
        story.append(Paragraph("CRASH IMPACT SUMMARY", S["label"]))
        story.append(Spacer(1, 3))
        story.append(crash_bar_chart(crashes, width=INNER_W, height=110))
        story.append(Spacer(1, 2*mm))

    # ── PAGE 2: DETAILED CHECKS ───────────────────────────────────────────────
    story.append(PageBreak())

    story.append(Paragraph("CATEGORY BREAKDOWN", S["h_section"]))
    radar_col   = category_radar_chart(checks, width=140, height=140)
    cats_scored = ["Overfitting", "Risk", "Regime", "Execution", "Compliance"]
    weights_map = {"Overfitting": 22, "Risk": 38, "Regime": 14, "Execution": 11, "Compliance": 15}
    cat_legend_rows = []
    for cat in cats_scored:
        cc = [c for c in checks if
              (c.get("category") if isinstance(c, dict) else getattr(c, "category", "")) == cat]
        avg_s = sum(
            (c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0))
            for c in cc
        ) / max(len(cc), 1)
        passed_n = sum(1 for c in cc if
                       (c.get("passed") if isinstance(c, dict) else getattr(c, "passed", False)))
        cat_legend_rows.append([
            Paragraph(f"{cat} ({weights_map.get(cat,0)}%)",
                      _s("cl", fontName="Helvetica-Bold", fontSize=8, textColor=cat_color(avg_s))),
            [score_bar(avg_s, width=46, height=5),
             Paragraph(f"{avg_s:.0f}/100  {passed_n}/{len(cc)} pass",
                       _s("clv", fontSize=7.5, textColor=C_LITE, leading=10))],
        ])
    legend_tbl = Table(cat_legend_rows, colWidths=[INNER_W*0.40 - 140, INNER_W*0.60 - 10])
    legend_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
    ]))
    radar_tbl = Table([[radar_col, legend_tbl]],
                      colWidths=[140 + 10, INNER_W - 150])
    radar_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,0),(-1,-1), C_PAPER),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ]))
    story.append(radar_tbl)
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("DETAILED CHECK BREAKDOWN - 44 CHECKS", S["h_section"]))
    story.append(Paragraph(
        "Every check was run against your specific strategy data. "
        "Red rows are institutional failures. Orange rows are warnings. "
        "The Score column shows each check's individual contribution (0-100).",
        S["body"]))
    story.append(Spacer(1, 4*mm))

    cat_order  = ["Overfitting", "Risk", "Regime", "Execution", "Compliance",
                  "Plausibility", "Operational", "SEBI Compliance"]
    cat_labels = {
        "Overfitting":     "OVERFITTING DETECTION",
        "Risk":            "RISK MANAGEMENT",
        "Regime":          "REGIME ANALYSIS",
        "Execution":       "EXECUTION & CAPACITY",
        "Compliance":      "COMPLIANCE",
        "Plausibility":    "PLAUSIBILITY AUDIT",
        "Operational":     "OPERATIONAL METRICS",
        "SEBI Compliance": "SEBI COMPLIANCE (India)",
    }

    by_cat = {}
    for c in checks:
        cat = c.get("category", "Other") if isinstance(c, dict) else getattr(c, "category", "Other")
        by_cat.setdefault(cat, []).append(c)

    for cat in cat_order:
        cat_checks = by_cat.get(cat, [])
        if not cat_checks:
            continue

        avg = sum(_get(c, "score", 0) for c in cat_checks) / len(cat_checks)

        cat_hdr = Table([[
            Paragraph(cat_labels.get(cat, cat.upper()),
                      _s("ch", fontName="Helvetica-Bold", fontSize=10,
                         textColor=C_WHITE, leading=13)),
            [Paragraph(f"AVG {avg:.0f}/100",
                       _s("cavg", fontName="Helvetica-Bold", fontSize=9,
                          textColor=C_WHITE, alignment=TA_RIGHT, leading=13)),
             score_bar(avg, width=46, height=6)],
        ]], colWidths=[INNER_W*0.68, INNER_W*0.32])
        cat_hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(KeepTogether([cat_hdr]))
        story.append(Spacer(1, 1))

        for chk in cat_checks:
            passed  = _get(chk, "passed", False)
            cscore  = float(_get(chk, "score", 0))
            name    = _get(chk, "name", "")
            value   = _clean_text(str(_get(chk, "value", "")))[:180]
            insight = _clean_text(str(_get(chk, "insight", "")))
            fix     = _clean_text(str(_get(chk, "fix", "")))

            row_bg  = (colors.HexColor("#F0FBF5") if passed else
                       colors.HexColor("#FFF8EC") if cscore > 20 else
                       colors.HexColor("#FEF0EE"))
            ico_col = C_OK if passed else C_WARN if cscore > 20 else C_DANGER
            ico_txt = "PASS" if passed else "WARN" if cscore > 20 else "FAIL"

            # Highlight DSR check with n_trials context
            is_dsr = "deflated" in name.lower() or "dsr" in name.lower()
            dsr_note = ""
            if is_dsr:
                dsr_note = f" [validated at n={n_trials} — {'institutional depth' if n_trials > 10 else 'standard depth, upgrade for deeper stress test'}]"

            rows = [[
                Paragraph(f"{name}{dsr_note}",
                          _s("ckn", fontName="Helvetica-Bold", fontSize=9,
                             textColor=depth_col if is_dsr else C_BLACK, leading=12)),
                [score_bar(cscore, width=38, height=5),
                 Paragraph(f" {cscore:.0f}", _s("ckscore", fontName="Helvetica-Bold",
                    fontSize=8, textColor=cat_color(cscore), alignment=TA_CENTER))],
                Paragraph(ico_txt, _s("ckico", fontName="Helvetica-Bold",
                    fontSize=9, textColor=ico_col, alignment=TA_CENTER, leading=12)),
            ], [
                Paragraph(value, S["body_sm"]), "", "",
            ]]
            if insight:
                rows.append([Paragraph(insight[:200], S["insight"]), "", ""])
            if fix and not passed:
                rows.append([Paragraph(f"Fix: {fix[:200]}", S["fix_txt"]), "", ""])

            spans = [("SPAN", (0,1),(2,1))]
            if len(rows) > 2: spans.append(("SPAN", (0,2),(2,2)))
            if len(rows) > 3: spans.append(("SPAN", (0,3),(2,3)))

            ck_tbl = Table(rows, colWidths=[INNER_W*0.66, INNER_W*0.21, INNER_W*0.13])
            ck_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), row_bg),
                ("TOPPADDING",    (0,0),(-1,-1), 5),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("LEFTPADDING",   (0,0),(-1,-1), 10),
                ("RIGHTPADDING",  (0,0),(-1,-1), 6),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
                ("LINEBELOW",     (0,-1),(-1,-1), 0.3, C_PAPER2),
                *spans,
            ]))
            story.append(ck_tbl)

        story.append(Spacer(1, 5*mm))

    # ── AI PERSONALISED ANALYSIS ──────────────────────────────────────────────
    ai_plan = vr.get("ai_plan")
    if ai_plan and not isinstance(ai_plan, dict):
        ai_plan = {"summary": str(ai_plan), "fixes": [], "encouragement": ""}

    if ai_plan and isinstance(ai_plan, dict):
        story.append(PageBreak())
        story.append(Paragraph("AI PERSONALISED ANALYSIS", S["h_section"]))
        story.append(Spacer(1, 8))

        summary_text = ai_plan.get("summary", "")
        if summary_text:
            story.append(Paragraph(_clean_text(summary_text), S["body"]))
            story.append(Spacer(1, 12))

        fixes = ai_plan.get("fixes", [])
        if fixes:
            story.append(Paragraph("RANKED FIX PLAN",
                _s("h_sub", fontName="Helvetica-Bold", fontSize=11, textColor=C_BLACK)))
            story.append(Spacer(1, 6))
            for fix in fixes:
                rank         = fix.get("rank", "")
                check        = fix.get("check", "")
                why          = fix.get("why_it_matters", "")
                specific     = fix.get("specific_fix", "")
                score_impact = fix.get("score_impact") or ""
                effort       = fix.get("effort") or ""
                header       = f"#{rank} - {check}"
                if score_impact: header += f"  |  {score_impact}"
                if effort:       header += f"  |  Effort: {effort}"
                story.append(Paragraph(_clean_text(header), S["label"]))
                if why:      story.append(Paragraph(f"Why it matters: {_clean_text(why)}", S["body"]))
                if specific: story.append(Paragraph(f"Fix: {_clean_text(specific)}", S["body"]))
                story.append(Spacer(1, 8))

        enc = ai_plan.get("encouragement", "")
        if enc:
            story.append(Spacer(1, 4))
            story.append(Paragraph(_clean_text(enc), S["body"]))
        story.append(Spacer(1, 16))

    # ── ACTION PLAN ───────────────────────────────────────────────────────────
    failed_checks = [c for c in checks
                     if not _get(c, "passed", True) and _get(c, "category") != "Plausibility"]
    if failed_checks:
        story.append(PageBreak())
        story.append(Paragraph("YOUR ACTION PLAN", S["h_section"]))
        story.append(Paragraph(
            f"You have {len(failed_checks)} failed checks. Fix the top items first.",
            S["body"]))
        story.append(Spacer(1, 5*mm))

        priority = {"Overfitting": 1, "Compliance": 2, "Risk": 3,
                    "Execution": 4, "Regime": 5, "Plausibility": 9}
        ranked = sorted(failed_checks,
                        key=lambda c: priority.get(_get(c, "category", ""), 8))

        for i, chk in enumerate(ranked[:10], 1):
            cat       = _get(chk, "category", "")
            badge_col = (C_DANGER if priority.get(cat, 8) <= 2 else
                         C_WARN   if priority.get(cat, 8) <= 4 else C_LITE)
            row = Table([[
                Paragraph(str(i), _s("bnum", fontName="Helvetica-Bold",
                    fontSize=16, textColor=badge_col, alignment=TA_CENTER, leading=20)),
                [
                    Paragraph(f"{_get(chk, 'name', '')}  [{cat}]", S["action_h"]),
                    Spacer(1, 4),
                    Paragraph(_clean_text(str(_get(chk, "insight", "")))[:220], S["action_b"]),
                    Spacer(1, 4),
                    Paragraph(f"Fix: {_clean_text(str(_get(chk, 'fix', '')))[:220]}",
                        _s("act_fix", fontSize=9, leading=13, textColor=C_ORANGE)),
                ],
            ]], colWidths=[13*mm, INNER_W - 13*mm])
            row.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), C_PAPER),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
                ("TOPPADDING",    (0,0),(-1,-1), 10),
                ("BOTTOMPADDING", (0,0),(-1,-1), 10),
                ("LEFTPADDING",   (0,0),(-1,-1), 8),
                ("RIGHTPADDING",  (0,0),(-1,-1), 10),
                ("LINEAFTER",     (0,0),(0,0),   2.5, badge_col),
                ("LINEBELOW",     (0,-1),(-1,-1), 0.3, C_PAPER2),
            ]))
            story.append(row)
            story.append(Spacer(1, 3))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
    story.append(Spacer(1, 3*mm))
    vh = vr.get("validation_hash", f"QP-{abs(hash(str(score)+grade+str(n)))%100000:05d}")
    story.append(Paragraph(
        f"Report ID: {vh}  |  Engine: v6.2  |  44 checks  |  5 crash simulations  |  "
        f"Depth: {depth_short}  |  "
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        S["footer"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "For educational and analytical purposes only. Not financial advice. "
        "Past backtest performance does not guarantee future results. "
        "Methodology: Lopez de Prado (2018) AFML, Almgren-Chriss (2001), Bailey & Lopez de Prado (2012).",
        S["footer"]))

    doc.build(story, onLaterPages=footer_canvas, onFirstPage=footer_canvas)
    return buf.getvalue()
