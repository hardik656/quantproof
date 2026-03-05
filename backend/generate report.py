"""
QuantProof PDF Report Generator v1.0
Institutional-grade analysis report using ReportLab.
Called from FastAPI: generate_pdf_report(validation_response_dict) -> bytes
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
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Wedge
from reportlab.graphics import renderPDF
import io, datetime, math

# ─── BRAND ────────────────────────────────────────────────────────────────────
C_BLACK     = colors.HexColor("#0A0A0A")
C_WHITE     = colors.white
C_ACCENT    = colors.HexColor("#00C896")   # QuantProof teal
C_ORANGE    = colors.HexColor("#FF6B35")
C_DANGER    = colors.HexColor("#FF3B30")
C_WARN      = colors.HexColor("#FF9F0A")
C_OK        = colors.HexColor("#30D158")
C_DARK      = colors.HexColor("#1C1C1E")
C_MID       = colors.HexColor("#3A3A3C")
C_LITE      = colors.HexColor("#8E8E93")
C_PAPER     = colors.HexColor("#F2F2F7")
C_PAPER2    = colors.HexColor("#E5E5EA")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
INNER_W = PAGE_W - 2 * MARGIN   # ~159mm


def _s(name, **kw):
    """Shorthand style factory."""
    defaults = dict(fontName="Helvetica", fontSize=10, leading=14,
                    textColor=C_DARK, spaceAfter=0, spaceBefore=0)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


def make_styles():
    return {
        "h_section":  _s("h_section",  fontName="Helvetica-Bold", fontSize=12,
                          textColor=C_BLACK, spaceBefore=12, spaceAfter=5,
                          borderPad=0),
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
                          fontName="Helvetica-Oblique", textColor=C_MID,
                          leftIndent=8),
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
    }


# ─── SCORE GAUGE ──────────────────────────────────────────────────────────────
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
            x0 = cx + R * math.cos(a0);  y0 = cy + R * math.sin(a0)
            x1 = cx + R * math.cos(a1);  y1 = cy + R * math.sin(a1)
            ln = Line(x0, y0, x1, y1)
            ln.strokeColor = col; ln.strokeWidth = 9
            d.add(ln)

    # needle
    na = math.radians(180 - score * 1.8)
    nx = cx + (R - 6) * math.cos(na);  ny = cy + (R - 6) * math.sin(na)
    needle = Line(cx, cy, nx, ny)
    needle.strokeColor = C_BLACK;  needle.strokeWidth = 2.2
    d.add(needle)

    # hub dot
    hub = Wedge(cx, cy, 5, 0, 360)
    hub.fillColor = C_BLACK;  hub.strokeColor = None
    d.add(hub)

    d.add(String(cx, cy + 22, f"{score:.0f}",
                 fontName="Helvetica-Bold", fontSize=24,
                 fillColor=C_BLACK, textAnchor="middle"))
    d.add(String(cx, cy + 11, grade_short,
                 fontName="Helvetica", fontSize=8,
                 fillColor=C_LITE, textAnchor="middle"))
    return d


def score_bar(score: float, width=50, height=7) -> Drawing:
    """Mini horizontal progress bar."""
    d = Drawing(width, height)
    bg = Rect(0, 0, width, height, fillColor=C_PAPER2, strokeColor=None)
    d.add(bg)
    col = C_DANGER if score < 40 else C_WARN if score < 70 else C_OK if score < 88 else C_ACCENT
    fill = Rect(0, 0, score / 100 * width, height, fillColor=col, strokeColor=None)
    d.add(fill)
    return d


def footer_canvas(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(C_PAPER2)
    canvas.line(MARGIN, 15*mm, PAGE_W - MARGIN, 15*mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_LITE)
    canvas.drawCentredString(PAGE_W/2, 10*mm,
        f"QuantProof Institutional Analysis  ·  quantproof-frontend.onrender.com  ·  "
        f"{datetime.date.today().strftime('%d %B %Y')}  ·  Page {doc.page}")
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def generate_pdf_report(vr: dict) -> bytes:
    """
    vr = dict matching ValidationResponse fields from main.py.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    S   = make_styles()

    score   = float(vr.get("fundable_score", 0))
    grade   = vr.get("grade", "F — Do Not Deploy")
    sharpe  = float(vr.get("sharpe", 0))
    dd      = float(vr.get("max_drawdown", 0)) * 100
    wr      = float(vr.get("win_rate", 0))
    n       = int(vr.get("total_trades", 0))
    dr      = vr.get("date_range", "N/A")
    checks  = vr.get("checks", [])
    crashes = vr.get("crash_sims", [])
    issues  = vr.get("top_issues", [])
    strengths = vr.get("top_strengths", [])
    grade_short = grade.split("—")[0].strip()
    gc = grade_color(grade)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16*mm, bottomMargin=22*mm,
        title="QuantProof Analysis Report",
    )
    story = []

    # ══ HEADER BAND ══════════════════════════════════════════════════════════
    hdr = Table([[
        Paragraph("QUANTPROOF", _s("qplogo", fontName="Helvetica-Bold",
            fontSize=11, textColor=C_ACCENT, leading=14)),
        Paragraph("INSTITUTIONAL ANALYSIS REPORT",
            _s("rptlabel", fontName="Helvetica", fontSize=9,
               textColor=C_LITE, alignment=TA_RIGHT, leading=14)),
    ]], colWidths=[INNER_W*0.5, INNER_W*0.5])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_BLACK),
        ("TOPPADDING",   (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",  (0,0),(-1,-1), 12),
        ("RIGHTPADDING", (0,0),(-1,-1), 12),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 6*mm))

    # ══ SCORE + GRADE ROW ════════════════════════════════════════════════════
    verdict_text = {
        "A":  ("Institutionally Viable",
               f"This strategy passed all critical institutional checks. "
               f"A Sharpe of {sharpe:.2f} with {dd:.1f}% max drawdown places it in the top tier "
               f"of systematic strategies. It is viable for prop firm submission."),
        "B+": ("Prop Firm Ready",
               f"Core edge is real and scalable. Some execution or regime checks need refinement "
               f"before institutional submission. Suitable for live trading with proper sizing."),
        "B":  ("Live Tradeable — With Caution",
               f"The strategy shows a real but fragile edge (Sharpe {sharpe:.2f}). "
               f"One or more critical robustness checks flagged vulnerabilities. "
               f"Deploy at reduced size and monitor closely."),
        "F":  ("Do Not Deploy",
               f"Critical failures detected across key institutional checks. "
               f"Deploying this strategy with real capital risks significant loss. "
               f"The action plan below shows exactly what to fix first."),
    }
    tier = "A" if score >= 90 else "B+" if score >= 80 else "B" if score >= 70 else "F"
    verdict_title, verdict_body = verdict_text[tier]

    score_row = Table([[
        score_gauge(score, grade_short),
        [
            Paragraph("OVERALL VERDICT", S["label"]),
            Spacer(1, 4),
            Paragraph(grade, _s("grd", fontName="Helvetica-Bold",
                fontSize=17, leading=21, textColor=gc)),
            Spacer(1, 5),
            Paragraph(verdict_body, S["body"]),
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

    # ══ KEY METRICS ROW ══════════════════════════════════════════════════════
    story.append(Paragraph("KEY METRICS", S["label"]))
    story.append(Spacer(1, 3))

    def metric_cell(lbl, val, sub):
        return Table([[Paragraph(lbl, S["label"])],
                      [Paragraph(val, _s("mv", fontName="Helvetica-Bold",
                          fontSize=19, leading=23, textColor=C_BLACK))],
                      [Paragraph(sub, S["body_sm"])]],
                     colWidths=[36*mm])

    sharpe_col = C_OK if sharpe >= 1.5 else C_WARN if sharpe >= 0.5 else C_DANGER
    dd_col     = C_OK if dd < 15 else C_WARN if dd < 25 else C_DANGER
    wr_col     = C_OK if wr >= 55 else C_WARN if wr >= 45 else C_DANGER

    metrics_row = Table([[
        metric_cell("SHARPE RATIO",  f"{sharpe:.2f}",  "Target > 1.5"),
        metric_cell("MAX DRAWDOWN",  f"{dd:.1f}%",     "Target < 20%"),
        metric_cell("WIN RATE",      f"{wr:.1f}%",     "Target > 50%"),
        metric_cell("TOTAL TRADES",  str(n),           dr[:18] if dr != "N/A" else "No dates"),
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

    # ══ TOP ISSUES + STRENGTHS ════════════════════════════════════════════════
    if issues or strengths:
        col_a = []
        col_b = []
        if issues:
            col_a.append(Paragraph("PRIMARY ISSUES", S["label"]))
            col_a.append(Spacer(1, 3))
            for iss in issues[:4]:
                clean = iss[:110] if iss else ""
                col_a.append(Paragraph(f"<bullet>&#x2022;</bullet> {clean}", S["body_sm"]))
        if strengths:
            col_b.append(Paragraph("CONFIRMED STRENGTHS", S["label"]))
            col_b.append(Spacer(1, 3))
            for st in strengths[:3]:
                col_b.append(Paragraph(f"<bullet>&#x2022;</bullet> {st}", S["body_sm"]))

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

    # ══ CRASH SIMS ═══════════════════════════════════════════════════════════
    if crashes:
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("HISTORICAL CRASH STRESS TESTS", S["h_section"]))
        story.append(Paragraph(
            "Your strategy was stress-tested against three major market crashes using "
            "a 60-day crisis window with Almgren-Chriss volatility scaling and liquidity drag. "
            "A strategy that fails these tests would have likely blown up in real conditions.",
            S["body"]))
        story.append(Spacer(1, 4))

        survived_count = sum(1 for c in crashes if c.get("survived"))
        story.append(Paragraph(
            f"Crash survival: <b>{survived_count}/3</b> — "
            + ("All three survived. Strong crisis resilience." if survived_count == 3 else
               "Two survived. One critical failure." if survived_count == 2 else
               "One survived. Significant crisis vulnerability." if survived_count == 1 else
               "Failed all three. Extremely high crisis risk."),
            _s("crash_sum", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK,
               spaceAfter=6)))
        story.append(Spacer(1, 4))

        for crash in crashes:
            surv    = crash.get("survived", False)
            drop    = crash.get("strategy_drop", 0) * 100
            name    = crash.get("crash_name", "")
            year    = crash.get("year", "")
            mkt_d   = crash.get("market_drop", 0)
            verdict = crash.get("emotional_verdict", "").replace("🟢","").replace(
                      "🟡","").replace("🟠","").replace("🔴","").strip()
            desc    = crash.get("description", "")[:100]

            bg_row  = colors.HexColor("#EDF9F3") if surv else colors.HexColor("#FDF0EF")
            tag_col = C_OK if surv else C_DANGER
            tag_txt = "SURVIVED" if surv else "BLOWN UP"

            c_tbl = Table([
                [
                    Paragraph(f"<b>{name}</b>  <font color='#8E8E93' size=8>{year}</font>",
                              S["crash_name"]),
                    Paragraph(tag_txt, _s("tag", fontName="Helvetica-Bold",
                        fontSize=10, textColor=tag_col, alignment=TA_RIGHT)),
                ],
                [
                    Paragraph(
                        f"Market: {mkt_d:.0f}%  ·  Strategy: <b>{drop:+.1f}%</b>  ·  {desc}",
                        S["body_sm"]),
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

    # ══ PAGE 2: DETAILED CHECKS ═══════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("DETAILED CHECK BREAKDOWN — ALL 31 CHECKS", S["h_section"]))
    story.append(Paragraph(
        "Every check below was run against your specific strategy data. "
        "Red rows are institutional failures. Orange rows are warnings that won't disqualify "
        "you from retail trading but will at prop firm level. "
        "The 'Score' column shows each check's individual contribution (0–100).",
        S["body"]))
    story.append(Spacer(1, 4*mm))

    cat_order  = ["Overfitting", "Risk", "Regime", "Execution", "Compliance", "Plausibility"]
    cat_labels = {
        "Overfitting":  "OVERFITTING DETECTION",
        "Risk":         "RISK MANAGEMENT",
        "Regime":       "REGIME ANALYSIS",
        "Execution":    "EXECUTION & CAPACITY",
        "Compliance":   "COMPLIANCE",
        "Plausibility": "PLAUSIBILITY AUDIT",
    }

    by_cat = {}
    for c in checks:
        by_cat.setdefault(c.get("category","Other"), []).append(c)

    for cat in cat_order:
        cat_checks = by_cat.get(cat, [])
        if not cat_checks: continue

        avg = sum(c.get("score",0) for c in cat_checks) / len(cat_checks)

        # Category header
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
            passed  = chk.get("passed", False)
            cscore  = float(chk.get("score", 0))
            name    = chk.get("name", "")
            value   = chk.get("value", "").replace("✅","").replace("⚠","").replace("🔴","").replace("🟢","").replace("🟡","").replace("🟠","").strip()
            insight = chk.get("insight", "")
            fix     = chk.get("fix", "")

            row_bg = colors.HexColor("#F0FBF5") if passed else \
                     colors.HexColor("#FFF8EC") if cscore > 20 else \
                     colors.HexColor("#FEF0EE")
            ico_col = C_OK if passed else C_WARN if cscore > 20 else C_DANGER
            ico_txt = "PASS" if passed else "WARN" if cscore > 20 else "FAIL"

            rows = [[
                Paragraph(f"<b>{name}</b>", _s("ckn", fontName="Helvetica-Bold",
                    fontSize=9, textColor=C_BLACK, leading=12)),
                [score_bar(cscore, width=38, height=5),
                 Paragraph(f"{cscore:.0f}", _s("ckscore", fontName="Helvetica-Bold",
                    fontSize=8, textColor=cat_color(cscore), alignment=TA_CENTER))],
                Paragraph(ico_txt, _s("ckico", fontName="Helvetica-Bold",
                    fontSize=9, textColor=ico_col, alignment=TA_CENTER, leading=12)),
            ],[
                Paragraph(value[:120], S["body_sm"]), "", "",
            ]]
            if insight:
                rows.append([Paragraph(insight[:180], S["insight"]), "", ""])
            if fix and not passed:
                rows.append([Paragraph(f"Fix: {fix[:160]}", S["fix_txt"]), "", ""])

            spans = [("SPAN",(0,1),(2,1))]
            if len(rows) > 2: spans.append(("SPAN",(0,2),(2,2)))
            if len(rows) > 3: spans.append(("SPAN",(0,3),(2,3)))

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

    # ══ PAGE 3: ACTION PLAN ═══════════════════════════════════════════════════
    failed = [c for c in checks if not c.get("passed", True) and
              c.get("category") != "Plausibility"]
    if failed:
        story.append(PageBreak())
        story.append(Paragraph("YOUR ACTION PLAN", S["h_section"]))
        story.append(Paragraph(
            f"You have <b>{len(failed)} failed checks</b>. Below are the highest-priority fixes "
            f"ranked by impact — fix the top items first. Each issue includes the exact "
            f"mathematical reason it failed and what to do about it.",
            S["body"]))
        story.append(Spacer(1, 5*mm))

        priority = {"Overfitting": 1, "Compliance": 2, "Risk": 3,
                    "Execution": 4, "Regime": 5, "Plausibility": 9}
        ranked = sorted(failed, key=lambda c: priority.get(c.get("category",""), 8))

        for i, chk in enumerate(ranked[:10], 1):
            cat = chk.get("category","")
            badge_col = C_DANGER if priority.get(cat,8) <= 2 else \
                        C_WARN   if priority.get(cat,8) <= 4 else C_LITE

            row = Table([[
                Paragraph(str(i), _s("bnum", fontName="Helvetica-Bold",
                    fontSize=16, textColor=badge_col, alignment=TA_CENTER, leading=20)),
                [
                    Paragraph(
                        f"<b>{chk.get('name','')}</b>"
                        f"<font color='#8E8E93' size=8>  {cat}</font>",
                        S["action_h"]),
                    Spacer(1, 4),
                    Paragraph(chk.get("insight","")[:220], S["action_b"]),
                    Spacer(1, 4),
                    Paragraph(
                        f"<b>Fix:</b> {chk.get('fix','')[:220]}",
                        _s("act_fix", fontSize=9, leading=13,
                           textColor=C_ORANGE, fontName="Helvetica")),
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

    # ══ DISCLAIMER ════════════════════════════════════════════════════════════
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
    story.append(Spacer(1, 3*mm))
    rid = abs(hash(str(score) + grade + str(n))) % 100000
    story.append(Paragraph(
        f"Report ID: QP-{rid:05d}  ·  "
        f"Engine: v1.6  ·  31 checks  ·  Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        S["footer"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "This report is for educational and analytical purposes only and does not constitute "
        "financial advice. Past backtest performance does not guarantee future results. "
        "Methodology: López de Prado (2018) AFML, Almgren-Chriss (2001), Bailey & López de Prado (2012).",
        S["footer"]))

    doc.build(story, onLaterPages=footer_canvas, onFirstPage=footer_canvas)
    return buf.getvalue()


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, pandas as pd
    sys.path.insert(0, '/mnt/user-data/outputs/quantproof/backend')
    from validator import QuantProofValidator

    for label, seed, mean, std, n_trades in [
        ("institutional", 1,   0.005,  0.008, 300),
        ("retail_ok",    42,   0.003,  0.012, 250),
        ("disaster",      9,  -0.003,  0.018,  60),
    ]:
        r  = np.random.default_rng(seed).normal(mean, std, n_trades)
        df = pd.DataFrame({'date': pd.date_range('2022-01-01', periods=n_trades), 'pnl': r})
        rep = QuantProofValidator(df).run()

        scored = [c for c in rep.checks if c.category != "Plausibility"]
        vr = {
            "fundable_score": rep.score,
            "grade": rep.grade,
            "sharpe": round(rep.sharpe, 2),
            "max_drawdown": round(rep.max_drawdown, 4),
            "win_rate": round(rep.win_rate, 1),
            "total_trades": rep.total_trades,
            "date_range": "2022-01-01 to 2023-01-01",
            "checks": [{"name":c.name,"passed":c.passed,"score":c.score,
                        "value":c.value,"insight":c.insight,"fix":c.fix,
                        "category":c.category} for c in rep.checks],
            "crash_sims": [{"crash_name":s.crash_name,"year":s.year,
                "description":s.description,"market_drop":s.market_drop,
                "strategy_drop":s.strategy_drop,"survived":s.survived,
                "emotional_verdict":s.emotional_verdict} for s in rep.crash_sims],
            "top_issues":    [c.fix  for c in sorted(scored, key=lambda x: x.score)[:4]],
            "top_strengths": [c.name for c in sorted(scored, key=lambda x: x.score, reverse=True)[:3]],
        }
        pdf = generate_pdf_report(vr)
        out = f"/mnt/user-data/outputs/quantproof_report_{label}.pdf"
        open(out, "wb").write(pdf)
        print(f"[{label}] {len(pdf)//1024}KB -> {out}")
