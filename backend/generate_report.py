"""
QuantProof PDF Report Generator v1.2
Synced to validator v6.3

v1.2 changes:
  - n_trials depth shown in report header (Standard / Institutional)
  - DSR check row highlighted with n_trials context note
  - Footer shows validation depth used
  - Depth badge on cover page
"""

import datetime
import io
import math

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String, Wedge
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Brand
C_BLACK = colors.HexColor("#0A0A0A")
C_WHITE = colors.white
C_ACCENT = colors.HexColor("#00C896")
C_ORANGE = colors.HexColor("#FF6B35")
C_DANGER = colors.HexColor("#FF3B30")
C_WARN = colors.HexColor("#FF9F0A")
C_OK = colors.HexColor("#30D158")
C_DARK = colors.HexColor("#1C1C1E")
C_MID = colors.HexColor("#3A3A3C")
C_LITE = colors.HexColor("#8E8E93")
C_PAPER = colors.HexColor("#F2F2F7")
C_PAPER2 = colors.HexColor("#E5E5EA")
C_INST = colors.HexColor("#7B61FF")  # purple for Institutional depth badge

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
INNER_W = PAGE_W - 2 * MARGIN


def _s(name, **kw):
    defaults = dict(
        fontName="Helvetica", fontSize=10, leading=14, textColor=C_DARK, spaceAfter=0, spaceBefore=0
    )
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


def make_styles():
    return {
        "h_section": _s(
            "h_section",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=C_BLACK,
            spaceBefore=12,
            spaceAfter=5,
        ),
        "body": _s("body", leading=15, alignment=TA_JUSTIFY, textColor=C_DARK, spaceAfter=4),
        "body_sm": _s("body_sm", fontSize=9, leading=13, textColor=C_MID),
        "label": _s("label", fontName="Helvetica-Bold", fontSize=8, textColor=C_LITE, leading=10),
        "big_num": _s("big_num", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=C_BLACK),
        "grade": _s("grade", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=C_BLACK),
        "insight": _s(
            "insight", fontSize=9, leading=13, fontName="Helvetica-Oblique", textColor=C_MID, leftIndent=8
        ),
        "fix_txt": _s("fix_txt", fontSize=9, leading=13, textColor=C_ORANGE, leftIndent=8),
        "footer": _s("footer", fontSize=7.5, leading=11, textColor=C_LITE, alignment=TA_CENTER),
        "crash_name": _s("crash_name", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK),
        "verdict": _s("verdict", fontSize=9, leading=13, fontName="Helvetica-Oblique", textColor=C_MID),
        "num_badge": _s(
            "num_badge", fontName="Helvetica-Bold", fontSize=13, textColor=C_ACCENT, alignment=TA_CENTER
        ),
        "action_h": _s("action_h", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK),
        "action_b": _s("action_b", fontSize=9, leading=13, textColor=C_DARK, alignment=TA_JUSTIFY),
        "watermark": _s(
            "watermark", fontName="Helvetica-Bold", fontSize=8, textColor=C_ACCENT, alignment=TA_RIGHT
        ),
        "depth_note": _s(
            "depth_note", fontSize=8, leading=12, textColor=C_INST, fontName="Helvetica-Oblique"
        ),
    }


def _depth_label(n_trials: int) -> tuple:
    """Return (short_label, description, color) for a given n_trials value."""
    if n_trials <= 1:
        return "QUICK (n=1)", "Rapid scan â€” less rigorous", C_LITE
    elif n_trials <= 10:
        return f"STANDARD (n={n_trials})", "Standard free-tier depth", C_ACCENT
    elif n_trials <= 20:
        return f"INSTITUTIONAL (n={n_trials})", "Institutional depth â€” fund-grade stress test", C_INST
    else:
        return (
            f"DEEP INSTITUTIONAL (n={n_trials})",
            f"Deep stress test â€” {n_trials}x Monte Carlo simulations",
            C_INST,
        )


# Score gauge
def score_gauge(score: float, grade_short: str, width=130, height=80) -> Drawing:
    d = Drawing(width, height)
    cx = width / 2
    cy = 10
    R = 54
    arcs = [(0, 40, C_DANGER), (40, 70, C_WARN), (70, 88, C_OK), (88, 100, C_ACCENT)]
    for s0, s1, col in arcs:
        steps = max(3, int((s1 - s0) * 1.8))
        for i in range(steps):
            a0 = math.radians(180 - (s0 + (s1 - s0) * i / steps) * 1.8)
            a1 = math.radians(180 - (s0 + (s1 - s0) * (i + 1) / steps) * 1.8)
            x0 = cx + R * math.cos(a0)
            y0 = cy + R * math.sin(a0)
            x1 = cx + R * math.cos(a1)
            y1 = cy + R * math.sin(a1)
            ln = Line(x0, y0, x1, y1)
            ln.strokeColor = col
            ln.strokeWidth = 9
            d.add(ln)
    na = math.radians(180 - score * 1.8)
    nx = cx + (R - 6) * math.cos(na)
    ny = cy + (R - 6) * math.sin(na)
    needle = Line(cx, cy, nx, ny)
    needle.strokeColor = C_BLACK
    needle.strokeWidth = 2.2
    d.add(needle)
    hub = Wedge(cx, cy, 5, 0, 360)
    hub.fillColor = C_BLACK
    hub.strokeColor = None
    d.add(hub)
    d.add(
        String(
            cx,
            cy + 22,
            f"{score:.0f}",
            fontName="Helvetica-Bold",
            fontSize=24,
            fillColor=C_BLACK,
            textAnchor="middle",
        )
    )
    d.add(
        String(
            cx, cy + 11, grade_short, fontName="Helvetica", fontSize=8, fillColor=C_LITE, textAnchor="middle"
        )
    )
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
        d.add(
            String(
                w / 2,
                height / 2,
                "No equity curve data",
                fontName="Helvetica",
                fontSize=8,
                fillColor=C_LITE,
                textAnchor="middle",
            )
        )
        return d

    d = Drawing(w, height)
    n = len(equity)
    pad = dict(l=8, r=6, t=10, b=18)
    aw = w - pad["l"] - pad["r"]
    ah = height - pad["t"] - pad["b"]
    mn = min(equity)
    mx = max(equity)
    rng = mx - mn or 1.0

    d.add(
        Rect(
            pad["l"],
            pad["b"],
            aw,
            ah,
            fillColor=colors.HexColor("#F8F8FA"),
            strokeColor=C_PAPER2,
            strokeWidth=0.5,
        )
    )

    if mn <= 100 <= mx:
        y100 = pad["b"] + (100 - mn) / rng * ah
        ln = Line(pad["l"], y100, pad["l"] + aw, y100)
        ln.strokeColor = C_PAPER2
        ln.strokeWidth = 0.8
        ln.strokeDashArray = [3, 3]
        d.add(ln)

    pts = [(pad["l"] + i / (n - 1) * aw, pad["b"] + (equity[i] - mn) / rng * ah) for i in range(n)]

    fill_pts = [pad["l"], pad["b"]]
    for px, py in pts:
        fill_pts += [px, py]
    fill_pts += [pad["l"] + aw, pad["b"]]
    poly = Polygon(fill_pts)
    end_col = C_OK if equity[-1] >= equity[0] else C_DANGER
    poly.fillColor = colors.Color(end_col.red, end_col.green, end_col.blue, alpha=0.10)
    poly.strokeColor = None
    d.add(poly)

    for i in range(n - 1):
        seg = Line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        seg.strokeColor = end_col
        seg.strokeWidth = 1.4
        d.add(seg)

    for val, label in [(mn, f"{mn:.0f}"), (100, "100"), (mx, f"{mx:.0f}")]:
        if mn == mx:
            break
        yy = pad["b"] + (val - mn) / rng * ah
        d.add(
            String(
                pad["l"] - 2,
                yy - 3,
                label,
                fontName="Helvetica",
                fontSize=6,
                fillColor=C_LITE,
                textAnchor="end",
            )
        )

    for idx, label in [(0, "T1"), (n // 2, f"T{n // 2}"), (n - 1, f"T{n}")]:
        xx = pad["l"] + idx / max(n - 1, 1) * aw
        d.add(
            String(
                xx,
                pad["b"] - 10,
                label,
                fontName="Helvetica",
                fontSize=6,
                fillColor=C_LITE,
                textAnchor="middle",
            )
        )

    final_x = pad["l"] + aw + 2
    final_y = pts[-1][1] - 3
    d.add(
        String(
            final_x,
            final_y,
            f"{equity[-1]:.1f}",
            fontName="Helvetica-Bold",
            fontSize=7,
            fillColor=end_col,
            textAnchor="start",
        )
    )

    return d


def crash_bar_chart(crashes: list, width=None, height=110) -> Drawing:
    w = width or INNER_W
    if not crashes:
        return Drawing(w, height)

    d = Drawing(w, height)
    n = len(crashes)
    pad = dict(l=32, r=10, t=8, b=28)
    aw = w - pad["l"] - pad["r"]
    ah = height - pad["t"] - pad["b"]
    bw = aw / n
    bw2 = bw * 0.32
    gap = bw * 0.08

    all_vals = [c.get("market_drop", 0) for c in crashes] + [c.get("strategy_drop", 0) * 100 for c in crashes]
    mn = min(all_vals + [0])
    mx = max(all_vals + [0])
    rng = mx - mn or 1.0

    def y_of(v):
        return pad["b"] + (v - mn) / rng * ah

    y0 = y_of(0)
    zl = Line(pad["l"], y0, pad["l"] + aw, y0)
    zl.strokeColor = C_MID
    zl.strokeWidth = 0.6
    d.add(zl)

    for tick_v in [mn, mn / 2, 0]:
        yt = y_of(tick_v)
        tl = Line(pad["l"] - 3, yt, pad["l"], yt)
        tl.strokeColor = C_LITE
        tl.strokeWidth = 0.5
        d.add(tl)
        d.add(
            String(
                pad["l"] - 4,
                yt - 3,
                f"{tick_v:.0f}%",
                fontName="Helvetica",
                fontSize=6,
                fillColor=C_LITE,
                textAnchor="end",
            )
        )

    for i, crash in enumerate(crashes):
        cx = pad["l"] + i * bw + bw * 0.1
        mktd = crash.get("market_drop", 0)
        strd = crash.get("strategy_drop", 0) * 100
        surv = crash.get("survived", False)

        mkt_y0 = y_of(min(mktd, 0))
        mkt_h = abs(y_of(mktd) - y0)
        mr = Rect(cx, mkt_y0, bw2, mkt_h)
        mr.fillColor = colors.Color(C_DANGER.red, C_DANGER.green, C_DANGER.blue, alpha=0.7)
        mr.strokeColor = None
        d.add(mr)

        str_col = C_OK if surv else C_WARN
        str_y = y_of(min(strd, 0))
        str_h = abs(y_of(strd) - y0)
        sr = Rect(cx + bw2 + gap, str_y, bw2, max(str_h, 1))
        sr.fillColor = colors.Color(str_col.red, str_col.green, str_col.blue, alpha=0.75)
        sr.strokeColor = None
        d.add(sr)

        year_raw = crash.get("year", "")
        label = year_raw.split("-")[0].split(" ")[-1][:4]
        d.add(
            String(
                cx + bw2 + gap / 2,
                pad["b"] - 14,
                label,
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=C_DARK,
                textAnchor="middle",
            )
        )

        lbl_y = y_of(strd) + (3 if strd >= 0 else -10)
        d.add(
            String(
                cx + bw2 + gap / 2,
                lbl_y,
                f"{strd:+.0f}%",
                fontName="Helvetica-Bold",
                fontSize=6,
                fillColor=str_col,
                textAnchor="middle",
            )
        )

    lx = pad["l"]
    ly = pad["b"] - 24
    mr2 = Rect(lx, ly + 2, 8, 6)
    mr2.fillColor = colors.Color(C_DANGER.red, C_DANGER.green, C_DANGER.blue, alpha=0.7)
    mr2.strokeColor = None
    d.add(mr2)
    d.add(String(lx + 10, ly + 2, "Market", fontName="Helvetica", fontSize=6.5, fillColor=C_LITE))
    sr2 = Rect(lx + 50, ly + 2, 8, 6)
    sr2.fillColor = colors.Color(C_OK.red, C_OK.green, C_OK.blue, alpha=0.75)
    sr2.strokeColor = None
    d.add(sr2)
    d.add(String(lx + 62, ly + 2, "Your Strategy", fontName="Helvetica", fontSize=6.5, fillColor=C_LITE))

    return d


def category_radar_chart(checks: list, width=120, height=120) -> Drawing:
    cats = ["Overfitting", "Risk", "Regime", "Execution", "Compliance"]
    scores = {}
    for cat in cats:
        cc = [
            c
            for c in checks
            if (c.get("category") if isinstance(c, dict) else getattr(c, "category", "")) == cat
        ]
        if cc:
            scores[cat] = sum(
                (c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0)) for c in cc
            ) / len(cc)
        else:
            scores[cat] = 0

    d = Drawing(width, height)
    cx = width / 2
    cy = height / 2 - 6
    R = min(width, height) / 2 - 18
    n = len(cats)

    def pt(i, r_frac):
        ang = math.pi / 2 + 2 * math.pi * i / n
        return (cx + r_frac * R * math.cos(ang), cy + r_frac * R * math.sin(ang))

    for frac in [0.25, 0.5, 0.75, 1.0]:
        ring_pts = []
        for i in range(n):
            px, py = pt(i, frac)
            ring_pts += [px, py]
        ring = Polygon(ring_pts)
        ring.fillColor = None
        ring.strokeColor = C_PAPER2
        ring.strokeWidth = 0.5
        d.add(ring)

    for i in range(n):
        px, py = pt(i, 1.0)
        spoke = Line(cx, cy, px, py)
        spoke.strokeColor = C_PAPER2
        spoke.strokeWidth = 0.5
        d.add(spoke)

    score_pts = []
    for i, cat in enumerate(cats):
        frac = scores[cat] / 100
        px, py = pt(i, frac)
        score_pts += [px, py]
    spoly = Polygon(score_pts)
    spoly.fillColor = colors.Color(C_ACCENT.red, C_ACCENT.green, C_ACCENT.blue, alpha=0.18)
    spoly.strokeColor = C_ACCENT
    spoly.strokeWidth = 1.4
    d.add(spoly)

    for i, cat in enumerate(cats):
        frac = scores[cat] / 100
        px, py = pt(i, frac)
        dot = Wedge(px, py, 3, 0, 360)
        dot.fillColor = C_ACCENT
        dot.strokeColor = None
        d.add(dot)

    label_offsets = {0: (0, 6), 1: (8, 0), 2: (6, -8), 3: (-6, -8), 4: (-8, 0)}
    for i, cat in enumerate(cats):
        lx, ly = pt(i, 1.15)
        ox, oy = label_offsets.get(i, (0, 0))
        score_val = scores[cat]
        col = cat_color(score_val)
        d.add(
            String(
                lx + ox,
                ly + oy + 8,
                f"{cat[:5]}",
                fontName="Helvetica-Bold",
                fontSize=6.5,
                fillColor=col,
                textAnchor="middle",
            )
        )
        d.add(
            String(
                lx + ox,
                ly + oy,
                f"{score_val:.0f}",
                fontName="Helvetica",
                fontSize=6,
                fillColor=col,
                textAnchor="middle",
            )
        )

    return d


def footer_canvas(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(C_PAPER2)
    canvas.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_LITE)
    # Include depth in footer
    depth_str = getattr(footer_canvas, "_depth_str", "")
    canvas.drawCentredString(
        PAGE_W / 2,
        10 * mm,
        f"QuantProof Institutional Analysis  |  quantproof-frontend.onrender.com  |  "
        f"{datetime.date.today().strftime('%d %B %Y')}  |  {depth_str}  |  Page {doc.page}",
    )
    canvas.restoreState()


def grade_color(grade):
    if grade.startswith("A"):
        return C_OK
    if "B+" in grade:
        return C_ACCENT
    if grade.startswith("B"):
        return C_WARN
    return C_DANGER


def cat_color(s):
    if s >= 88:
        return C_ACCENT
    if s >= 70:
        return C_OK
    if s >= 50:
        return C_WARN
    return C_DANGER


def _clean_text(raw: str) -> str:
    v = raw or ""
    v = v.replace("[UNCALIBRATED] ", "").replace("[UNCALIBRATED]", "")
    for e in ("ðŸŸ¢", "ðŸŸ¡", "ðŸŸ ", "ðŸ”´"):
        v = v.replace(e, "")
    v = v.replace("âœ…", "PASS").replace("âœ“", "OK").replace("âœ—", "FAIL").replace("âŒ", "FAIL")
    v = v.replace("ðŸ“Š", "").replace("â„¹", "").replace("âš ", "WARNING:").replace("ðŸ“ˆ", "")
    v = v.replace("Î±", "alpha").replace("Î²", "beta").replace("Î”", "delta")
    v = (
        v.replace("â‰¥", ">=")
        .replace("â‰¤", "<=")
        .replace("â‰ˆ", "~")
        .replace("â†’", "->")
        .replace("â†", "<-")
    )
    v = v.replace("Ã—", "x").replace("Ã·", "/").replace("Â±", "+/-")
    v = v.replace("Ïƒ", "sigma")
    v = v.replace("â€”", " - ").replace("â€“", "-").replace("â€’", "-")
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


# â”€â”€â”€ PORTFOLIO CHARTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def correlation_heatmap(names: list, matrix: list, width=None, height=None) -> Drawing:
    """
    Draw a correlation heatmap as a coloured grid with values.
    """
    n = len(names)
    w = width or min(INNER_W, n * 60 + 40)
    h = height or min(160, n * 60 + 40)
    d = Drawing(w, h)
    pad = dict(l=48, r=8, t=8, b=48)
    cell_w = (w - pad["l"] - pad["r"]) / max(n, 1)
    cell_h = (h - pad["t"] - pad["b"]) / max(n, 1)

    def corr_color(v):
        # -1=red, 0=white, 1=green
        v = max(-1.0, min(1.0, v))
        if v >= 0:
            return colors.Color(1 - v * 0.6, 1.0, 1 - v * 0.6)
        else:
            return colors.Color(1.0, 1 + v * 0.6, 1 + v * 0.6)

    for i in range(n):
        for j in range(n):
            x = pad["l"] + j * cell_w
            y = pad["b"] + (n - 1 - i) * cell_h
            val = float(matrix[i][j]) if matrix and len(matrix) > i and len(matrix[i]) > j else 0.0
            cell = Rect(x, y, cell_w - 1, cell_h - 1)
            cell.fillColor = corr_color(val)
            cell.strokeColor = C_PAPER2
            cell.strokeWidth = 0.5
            d.add(cell)
            # Value label
            d.add(
                String(
                    x + cell_w / 2,
                    y + cell_h / 2 - 4,
                    f"{val:.2f}",
                    fontName="Helvetica-Bold" if i == j else "Helvetica",
                    fontSize=7,
                    fillColor=C_DARK,
                    textAnchor="middle",
                )
            )

    # Row labels (left)
    for i, name in enumerate(names):
        y = pad["b"] + (n - 1 - i) * cell_h + cell_h / 2 - 3
        d.add(
            String(
                pad["l"] - 4,
                y,
                name[:12],
                fontName="Helvetica",
                fontSize=7,
                fillColor=C_DARK,
                textAnchor="end",
            )
        )

    # Column labels (bottom)
    for j, name in enumerate(names):
        x = pad["l"] + j * cell_w + cell_w / 2
        d.add(
            String(
                x,
                pad["b"] - 12,
                name[:10],
                fontName="Helvetica",
                fontSize=7,
                fillColor=C_DARK,
                textAnchor="middle",
            )
        )

    return d


def portfolio_combined_curve(strategies: list, width=None, height=90) -> Drawing:
    """
    Overlaid equity curves for each strategy, normalised to 100.
    Drawn as line segments â€” one colour per strategy.
    """
    w = width or INNER_W
    d = Drawing(w, height)
    pad = dict(l=10, r=10, t=8, b=16)
    aw = w - pad["l"] - pad["r"]
    ah = height - pad["t"] - pad["b"]

    line_colors = [C_ACCENT, C_ORANGE, C_INST]

    # Background
    d.add(
        Rect(
            pad["l"],
            pad["b"],
            aw,
            ah,
            fillColor=colors.HexColor("#F8F8FA"),
            strokeColor=C_PAPER2,
            strokeWidth=0.5,
        )
    )

    for idx, strat in enumerate(strategies):
        strat.get("fundable_score", 50)
        sharpe = strat.get("sharpe", 0)
        n_pts = min(strat.get("total_trades", 50), 80)
        name = strat.get("name", f"S{idx + 1}")
        col = line_colors[idx % len(line_colors)]

        # Synthetic curve from Sharpe (real curve not stored in portfolio response)
        import random as _r

        _r.seed(abs(hash(name + str(sharpe))) % (2**31))
        eq, pts_raw = 100.0, [100.0]
        mean_r = sharpe * 0.012 / max((252 / max(n_pts, 1)) ** 0.5, 0.1)
        std_r = abs(mean_r / (sharpe + 1e-9))
        for _ in range(n_pts - 1):
            r = _r.gauss(mean_r, std_r)
            eq *= 1 + max(-0.5, min(0.5, r))
            pts_raw.append(eq)

        mn = min(pts_raw)
        mx = max(pts_raw)
        rng = mx - mn or 1.0
        pts = [
            (pad["l"] + i / (len(pts_raw) - 1) * aw, pad["b"] + (pts_raw[i] - mn) / rng * ah)
            for i in range(len(pts_raw))
        ]
        for i in range(len(pts) - 1):
            seg = Line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            seg.strokeColor = col
            seg.strokeWidth = 1.3
            d.add(seg)

        # Legend entry
        lx = pad["l"] + idx * (aw / len(strategies))
        d.add(Rect(lx, pad["b"] - 13, 8, 5, fillColor=col, strokeColor=None))
        d.add(String(lx + 10, pad["b"] - 13, name[:14], fontName="Helvetica", fontSize=6.5, fillColor=C_DARK))

    return d


# â”€â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_pdf_report(vr: dict) -> bytes:
    buf = io.BytesIO()
    S = make_styles()

    score = float(vr.get("fundable_score", 0))
    grade = vr.get("grade", "F - Do Not Deploy")
    sharpe = float(vr.get("sharpe", 0))
    dd = float(vr.get("max_drawdown", 0)) * 100
    wr = float(vr.get("win_rate", 0))
    n = int(vr.get("total_trades", 0))
    dr = vr.get("date_range", "N/A")
    checks = vr.get("checks", [])
    crashes = vr.get("crash_sims", [])
    issues = vr.get("top_issues", [])
    strengths = vr.get("top_strengths", [])

    # n_trials depth â€” read from vr, fall back to 10
    n_trials = int(vr.get("n_trials", 10))
    depth_short, depth_desc, depth_col = _depth_label(n_trials)

    # Store depth string for footer
    footer_canvas._depth_str = depth_short

    grade_short = grade.split("-")[0].strip().split(" ")[0]
    gc = grade_color(grade)

    import sys

    print(
        f"[PDF] score={score} grade={grade} n_trials={n_trials} depth={depth_short}",
        file=sys.stderr,
        flush=True,
    )

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=16 * mm,
        bottomMargin=22 * mm,
        title="QuantProof Analysis Report",
    )
    story = []

    # â”€â”€ HEADER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    is_verified = vr.get("is_verified", False)
    report_label = "VERIFIED INSTITUTIONAL REPORT" if is_verified else "METHODOLOGY ANALYSIS REPORT"

    hdr = Table(
        [
            [
                Paragraph(
                    "QUANTPROOF",
                    _s("qplogo", fontName="Helvetica-Bold", fontSize=11, textColor=C_ACCENT, leading=14),
                ),
                Paragraph(
                    report_label,
                    _s(
                        "rptlabel",
                        fontName="Helvetica",
                        fontSize=9,
                        textColor=C_LITE,
                        alignment=TA_RIGHT,
                        leading=14,
                    ),
                ),
            ]
        ],
        colWidths=[INNER_W * 0.5, INNER_W * 0.5],
    )
    hdr.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), C_BLACK),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(hdr)

    # â”€â”€ DEPTH BADGE STRIP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    depth_bg = colors.HexColor("#1A1040") if n_trials > 10 else colors.HexColor("#0A1A14")
    depth_strip = Table(
        [
            [
                Paragraph(
                    f"VALIDATION DEPTH: {depth_short}",
                    _s("ds", fontName="Helvetica-Bold", fontSize=8, textColor=depth_col, leading=11),
                ),
                Paragraph(
                    depth_desc,
                    _s(
                        "dd",
                        fontName="Helvetica",
                        fontSize=8,
                        textColor=C_LITE,
                        alignment=TA_RIGHT,
                        leading=11,
                    ),
                ),
            ]
        ],
        colWidths=[INNER_W * 0.55, INNER_W * 0.45],
    )
    depth_strip.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), depth_bg),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(depth_strip)
    story.append(Spacer(1, 6 * mm))

    # â”€â”€ SCORE + GRADE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    verdict_text = {
        "A": (
            "Institutionally Viable",
            f"Passed all critical institutional checks. Sharpe {sharpe:.2f}, "
            f"max drawdown {dd:.1f}%. "
            f"{'Connect a broker account for a verified compliance certificate.' if not is_verified else 'Eligible for prop firm submission.'}",
        ),
        "B+": (
            "Prop Firm Ready",
            "Core edge is real and scalable. Some execution or regime checks need refinement.",
        ),
        "B": (
            "Live Tradeable - With Caution",
            f"Real but fragile edge (Sharpe {sharpe:.2f}). Deploy at reduced size and monitor closely.",
        ),
        "C": (
            "Promising - Needs Work",
            f"Early-stage edge (Sharpe {sharpe:.2f}) but fails key institutional checks. "
            f"Fix the ranked issues below before deploying real capital.",
        ),
        "D": (
            "Major Issues",
            "Critical failures across key institutional checks. See the action plan below.",
        ),
        "F": (
            "Do Not Deploy",
            "Fails institutional standards. Do not deploy capital until major issues are resolved.",
        ),
    }
    tier = (
        "A"
        if score >= 90
        else "B+"
        if score >= 80
        else "B"
        if score >= 70
        else "C"
        if score >= 50
        else "D"
        if score >= 35
        else "F"
    )
    verdict_title, verdict_body = verdict_text.get(tier, ("Unknown", "Unable to determine verdict."))

    score_row = Table(
        [
            [
                score_gauge(score, grade_short),
                [
                    Paragraph("OVERALL VERDICT", S["label"]),
                    Spacer(1, 4),
                    Paragraph(
                        grade, _s("grd", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=gc)
                    ),
                    Spacer(1, 5),
                    Paragraph(verdict_body, S["body"]),
                    Spacer(1, 6),
                    # Depth context note under verdict
                    Paragraph(
                        f"Validated at {depth_short}. "
                        + (
                            f"DSR and overfitting checks used {n_trials} Monte Carlo simulations. "
                            "This is institutional-grade depth."
                            if n_trials > 10
                            else f"DSR used {n_trials} Monte Carlo simulation(s). "
                            "Upgrade to Institutional depth (n=20+) to stress-test further."
                        ),
                        S["depth_note"],
                    ),
                ],
            ]
        ],
        colWidths=[46 * mm, INNER_W - 46 * mm],
    )
    score_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(score_row)
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
    story.append(Spacer(1, 4 * mm))

    # â”€â”€ KEY METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(Paragraph("KEY METRICS", S["label"]))
    story.append(Spacer(1, 3))

    def metric_cell(lbl, val, sub):
        return Table(
            [
                [Paragraph(lbl, S["label"])],
                [
                    Paragraph(
                        val, _s("mv", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=C_BLACK)
                    )
                ],
                [Paragraph(sub, S["body_sm"])],
            ],
            colWidths=[36 * mm],
        )

    metrics_row = Table(
        [
            [
                metric_cell("SHARPE RATIO", f"{sharpe:.2f}", "Target > 1.5"),
                metric_cell("MAX DRAWDOWN", f"{dd:.1f}%", "Target < 20%"),
                metric_cell("WIN RATE", f"{wr:.1f}%", "Target > 50%"),
                metric_cell("TOTAL TRADES", str(n), dr[:18] if dr != "N/A" else "No dates"),
            ]
        ],
        colWidths=[INNER_W / 4] * 4,
    )
    metrics_row.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("LINEAFTER", (0, 0), (2, 0), 0.6, C_PAPER2),
            ]
        )
    )
    story.append(metrics_row)
    story.append(Spacer(1, 6 * mm))

    # â”€â”€ EQUITY CURVE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        std_r = abs(mean_r / (sharpe + 1e-9))
        for _ in range(min(n, 200) - 1):
            r = _random.gauss(mean_r, std_r)
            eq *= 1 + max(-0.99, min(0.99, r))
            equity_data.append(round(eq, 4))
        is_real = False
        curve_note = f"illustrative - derived from Sharpe {sharpe:.2f} (upload dated CSV for real curve)"
        story.append(Paragraph("EQUITY CURVE (ILLUSTRATIVE)", S["label"]))
        story.append(Spacer(1, 3))

    final_return = (equity_data[-1] / equity_data[0] - 1) * 100 if equity_data[0] else 0
    eq_stats_col = [
        Paragraph("TOTAL RETURN", S["label"]),
        Paragraph(
            f"{'+' if final_return >= 0 else ''}{final_return:.1f}%",
            _s(
                "eq_ret",
                fontName="Helvetica-Bold",
                fontSize=16,
                textColor=C_OK if final_return >= 0 else C_DANGER,
                leading=20,
            ),
        ),
        Spacer(1, 6),
        Paragraph("DATA SOURCE", S["label"]),
        Paragraph(
            "Real trades" if is_real else "Synthetic", _s("eq_src", fontSize=8, textColor=C_LITE, leading=11)
        ),
        Spacer(1, 4),
        Paragraph(
            curve_note, _s("eq_note", fontSize=7, textColor=C_LITE, leading=10, fontName="Helvetica-Oblique")
        ),
    ]
    eq_tbl = Table(
        [[equity_curve_chart(equity_data, width=INNER_W * 0.72), eq_stats_col]],
        colWidths=[INNER_W * 0.72, INNER_W * 0.28],
    )
    eq_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(eq_tbl)
    story.append(Spacer(1, 6 * mm))

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
            summ_row = Table([[col_a or [""], col_b or [""]]], colWidths=[INNER_W * 0.52, INNER_W * 0.48])
            summ_row.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FFF4F0")),
                        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F0FBF5")),
                        ("TOPPADDING", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(summ_row)
            story.append(Spacer(1, 6 * mm))

    # â”€â”€ AUDIT TRAIL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    audit_flags = vr.get("audit_flags", [])
    display_flags = [
        f for f in audit_flags if f and f.strip() and len(f.strip()) > 5 and not f.strip().startswith("   -")
    ][:8]

    if display_flags:
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("DATA & VALIDATION NOTES", S["h_section"]))
        flag_rows = []
        for flag in display_flags:
            clean_flag = _clean_text(flag)[:160]
            flag_rows.append([Paragraph(clean_flag, S["body_sm"])])
        flag_tbl = Table(flag_rows, colWidths=[INNER_W])
        flag_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_PAPER2),
                ]
            )
        )
        story.append(flag_tbl)
        story.append(Spacer(1, 4 * mm))

    # â”€â”€ CRASH SIMS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if crashes:
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("HISTORICAL CRASH STRESS TESTS", S["h_section"]))
        story.append(
            Paragraph(
                "Strategy stress-tested against five historical crash scenarios using "
                "a 60-day crisis window with Almgren-Chriss volatility scaling and liquidity drag.",
                S["body"],
            )
        )
        story.append(Spacer(1, 4))

        survived_count = sum(1 for c in crashes if c.get("survived"))
        story.append(
            Paragraph(
                f"Crash survival: {survived_count}/5 - "
                + (
                    "All five survived. Strong crisis resilience."
                    if survived_count == 5
                    else "Four survived. Minor vulnerability."
                    if survived_count == 4
                    else "Three survived. Moderate crisis risk."
                    if survived_count == 3
                    else "Two survived. Significant vulnerability."
                    if survived_count == 2
                    else "One survived. Extremely high crisis risk."
                    if survived_count == 1
                    else "Failed all five. Critical crisis exposure."
                ),
                _s("crash_sum", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK, spaceAfter=6),
            )
        )
        story.append(Spacer(1, 4))

        for crash in crashes:
            surv = crash.get("survived", False)
            drop = crash.get("strategy_drop", 0) * 100
            name = crash.get("crash_name", "")
            year = crash.get("year", "")
            mkt_d = crash.get("market_drop", 0)
            desc = crash.get("description", "")[:100]
            verdict = _clean_verdict(crash.get("emotional_verdict", ""))
            dd_breach = not surv and drop > 0
            bg_row = (
                colors.HexColor("#EDF9F3")
                if surv
                else colors.HexColor("#FFF8EC")
                if dd_breach
                else colors.HexColor("#FDF0EF")
            )
            tag_col = C_OK if surv else (C_WARN if dd_breach else C_DANGER)
            tag_txt = "SURVIVED" if surv else ("DD BREACH" if dd_breach else "BLOWN UP")

            c_tbl = Table(
                [
                    [
                        Paragraph(f"{name}  {year}", S["crash_name"]),
                        Paragraph(
                            tag_txt,
                            _s(
                                "tag",
                                fontName="Helvetica-Bold",
                                fontSize=10,
                                textColor=tag_col,
                                alignment=TA_RIGHT,
                            ),
                        ),
                    ],
                    [
                        Paragraph(f"Market: {mkt_d:.0f}%  Strategy: {drop:+.1f}%  {desc}", S["body_sm"]),
                        Paragraph("", S["body_sm"]),
                    ],
                    [
                        Paragraph(verdict[:180], S["verdict"]),
                        Paragraph("", S["body_sm"]),
                    ],
                ],
                colWidths=[INNER_W * 0.72, INNER_W * 0.28],
            )
            c_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), bg_row),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("SPAN", (0, 1), (1, 1)),
                        ("SPAN", (0, 2), (1, 2)),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, C_PAPER2),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(c_tbl)
            story.append(Spacer(1, 3))

        story.append(Spacer(1, 4))
        story.append(Paragraph("CRASH IMPACT SUMMARY", S["label"]))
        story.append(Spacer(1, 3))
        story.append(crash_bar_chart(crashes, width=INNER_W, height=110))
        story.append(Spacer(1, 2 * mm))

    # â”€â”€ PAGE 2: DETAILED CHECKS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(PageBreak())

    story.append(Paragraph("CATEGORY BREAKDOWN", S["h_section"]))
    radar_col = category_radar_chart(checks, width=140, height=140)
    cats_scored = ["Overfitting", "Risk", "Regime", "Execution", "Compliance"]
    weights_map = {"Overfitting": 22, "Risk": 38, "Regime": 14, "Execution": 11, "Compliance": 15}
    cat_legend_rows = []
    for cat in cats_scored:
        cc = [
            c
            for c in checks
            if (c.get("category") if isinstance(c, dict) else getattr(c, "category", "")) == cat
        ]
        avg_s = sum((c.get("score", 0) if isinstance(c, dict) else getattr(c, "score", 0)) for c in cc) / max(
            len(cc), 1
        )
        passed_n = sum(
            1 for c in cc if (c.get("passed") if isinstance(c, dict) else getattr(c, "passed", False))
        )
        cat_legend_rows.append(
            [
                Paragraph(
                    f"{cat} ({weights_map.get(cat, 0)}%)",
                    _s("cl", fontName="Helvetica-Bold", fontSize=8, textColor=cat_color(avg_s)),
                ),
                [
                    score_bar(avg_s, width=46, height=5),
                    Paragraph(
                        f"{avg_s:.0f}/100  {passed_n}/{len(cc)} pass",
                        _s("clv", fontSize=7.5, textColor=C_LITE, leading=10),
                    ),
                ],
            ]
        )
    legend_tbl = Table(cat_legend_rows, colWidths=[INNER_W * 0.40 - 140, INNER_W * 0.60 - 10])
    legend_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    radar_tbl = Table([[radar_col, legend_tbl]], colWidths=[140 + 10, INNER_W - 150])
    radar_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(radar_tbl)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("DETAILED CHECK BREAKDOWN - 37 SCORED CHECKS", S["h_section"]))
    story.append(
        Paragraph(
            "Every check was run against your specific strategy data. "
            "Red rows are institutional failures. Orange rows are warnings. "
            "The Score column shows each check's individual contribution (0-100).",
            S["body"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    cat_order = [
        "Overfitting",
        "Risk",
        "Regime",
        "Execution",
        "Compliance",
        "Plausibility",
        "Operational",
        "SEBI Compliance",
    ]
    cat_labels = {
        "Overfitting": "OVERFITTING DETECTION",
        "Risk": "RISK MANAGEMENT",
        "Regime": "REGIME ANALYSIS",
        "Execution": "EXECUTION & CAPACITY",
        "Compliance": "COMPLIANCE",
        "Plausibility": "PLAUSIBILITY AUDIT",
        "Operational": "OPERATIONAL METRICS",
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

        cat_hdr = Table(
            [
                [
                    Paragraph(
                        cat_labels.get(cat, cat.upper()),
                        _s("ch", fontName="Helvetica-Bold", fontSize=10, textColor=C_WHITE, leading=13),
                    ),
                    [
                        Paragraph(
                            f"AVG {avg:.0f}/100",
                            _s(
                                "cavg",
                                fontName="Helvetica-Bold",
                                fontSize=9,
                                textColor=C_WHITE,
                                alignment=TA_RIGHT,
                                leading=13,
                            ),
                        ),
                        score_bar(avg, width=46, height=6),
                    ],
                ]
            ],
            colWidths=[INNER_W * 0.68, INNER_W * 0.32],
        )
        cat_hdr.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(KeepTogether([cat_hdr]))
        story.append(Spacer(1, 1))

        for chk in cat_checks:
            passed = _get(chk, "passed", False)
            cscore = float(_get(chk, "score", 0))
            name = _get(chk, "name", "")
            value = _clean_text(str(_get(chk, "value", "")))[:180]
            insight = _clean_text(str(_get(chk, "insight", "")))
            fix = _clean_text(str(_get(chk, "fix", "")))

            row_bg = (
                colors.HexColor("#F0FBF5")
                if passed
                else colors.HexColor("#FFF8EC")
                if cscore > 20
                else colors.HexColor("#FEF0EE")
            )
            ico_col = C_OK if passed else C_WARN if cscore > 20 else C_DANGER
            ico_txt = "PASS" if passed else "WARN" if cscore > 20 else "FAIL"

            # Highlight DSR check with n_trials context
            is_dsr = "deflated" in name.lower() or "dsr" in name.lower()
            dsr_note = ""
            if is_dsr:
                dsr_note = f" [validated at n={n_trials} â€” {'institutional depth' if n_trials > 10 else 'standard depth, upgrade for deeper stress test'}]"

            rows = [
                [
                    Paragraph(
                        f"{name}{dsr_note}",
                        _s(
                            "ckn",
                            fontName="Helvetica-Bold",
                            fontSize=9,
                            textColor=depth_col if is_dsr else C_BLACK,
                            leading=12,
                        ),
                    ),
                    [
                        score_bar(cscore, width=38, height=5),
                        Paragraph(
                            f" {cscore:.0f}",
                            _s(
                                "ckscore",
                                fontName="Helvetica-Bold",
                                fontSize=8,
                                textColor=cat_color(cscore),
                                alignment=TA_CENTER,
                            ),
                        ),
                    ],
                    Paragraph(
                        ico_txt,
                        _s(
                            "ckico",
                            fontName="Helvetica-Bold",
                            fontSize=9,
                            textColor=ico_col,
                            alignment=TA_CENTER,
                            leading=12,
                        ),
                    ),
                ],
                [
                    Paragraph(value, S["body_sm"]),
                    "",
                    "",
                ],
            ]
            if insight:
                rows.append([Paragraph(insight[:200], S["insight"]), "", ""])
            if fix and not passed:
                rows.append([Paragraph(f"Fix: {fix[:200]}", S["fix_txt"]), "", ""])

            spans = [("SPAN", (0, 1), (2, 1))]
            if len(rows) > 2:
                spans.append(("SPAN", (0, 2), (2, 2)))
            if len(rows) > 3:
                spans.append(("SPAN", (0, 3), (2, 3)))

            ck_tbl = Table(rows, colWidths=[INNER_W * 0.66, INNER_W * 0.21, INNER_W * 0.13])
            ck_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), row_bg),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.3, C_PAPER2),
                        *spans,
                    ]
                )
            )
            story.append(ck_tbl)

        story.append(Spacer(1, 5 * mm))

    # â”€â”€ AI PERSONALISED ANALYSIS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            story.append(
                Paragraph(
                    "RANKED FIX PLAN", _s("h_sub", fontName="Helvetica-Bold", fontSize=11, textColor=C_BLACK)
                )
            )
            story.append(Spacer(1, 6))
            for fix in fixes:
                rank = fix.get("rank", "")
                check = fix.get("check", "")
                why = fix.get("why_it_matters", "")
                specific = fix.get("specific_fix", "")
                score_impact = fix.get("score_impact") or ""
                effort = fix.get("effort") or ""
                header = f"#{rank} - {check}"
                if score_impact:
                    header += f"  |  {score_impact}"
                if effort:
                    header += f"  |  Effort: {effort}"
                story.append(Paragraph(_clean_text(header), S["label"]))
                if why:
                    story.append(Paragraph(f"Why it matters: {_clean_text(why)}", S["body"]))
                if specific:
                    story.append(Paragraph(f"Fix: {_clean_text(specific)}", S["body"]))
                story.append(Spacer(1, 8))

        enc = ai_plan.get("encouragement", "")
        if enc:
            story.append(Spacer(1, 4))
            story.append(Paragraph(_clean_text(enc), S["body"]))
        story.append(Spacer(1, 16))

    # â”€â”€ ACTION PLAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    failed_checks = [
        c for c in checks if not _get(c, "passed", True) and _get(c, "category") != "Plausibility"
    ]
    if failed_checks:
        story.append(PageBreak())
        story.append(Paragraph("YOUR ACTION PLAN", S["h_section"]))
        story.append(
            Paragraph(f"You have {len(failed_checks)} failed checks. Fix the top items first.", S["body"])
        )
        story.append(Spacer(1, 5 * mm))

        priority = {
            "Overfitting": 1,
            "Compliance": 2,
            "Risk": 3,
            "Execution": 4,
            "Regime": 5,
            "Plausibility": 9,
        }
        ranked = sorted(failed_checks, key=lambda c: priority.get(_get(c, "category", ""), 8))

        for i, chk in enumerate(ranked[:10], 1):
            cat = _get(chk, "category", "")
            badge_col = (
                C_DANGER if priority.get(cat, 8) <= 2 else C_WARN if priority.get(cat, 8) <= 4 else C_LITE
            )
            row = Table(
                [
                    [
                        Paragraph(
                            str(i),
                            _s(
                                "bnum",
                                fontName="Helvetica-Bold",
                                fontSize=16,
                                textColor=badge_col,
                                alignment=TA_CENTER,
                                leading=20,
                            ),
                        ),
                        [
                            Paragraph(f"{_get(chk, 'name', '')}  [{cat}]", S["action_h"]),
                            Spacer(1, 4),
                            Paragraph(_clean_text(str(_get(chk, "insight", "")))[:220], S["action_b"]),
                            Spacer(1, 4),
                            Paragraph(
                                f"Fix: {_clean_text(str(_get(chk, 'fix', '')))[:220]}",
                                _s("act_fix", fontSize=9, leading=13, textColor=C_ORANGE),
                            ),
                        ],
                    ]
                ],
                colWidths=[13 * mm, INNER_W - 13 * mm],
            )
            row.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("LINEAFTER", (0, 0), (0, 0), 2.5, badge_col),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.3, C_PAPER2),
                    ]
                )
            )
            story.append(row)
            story.append(Spacer(1, 3))

    # â”€â”€ PORTFOLIO SECTION (if present) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    portfolio = vr.get("portfolio")
    if portfolio and isinstance(portfolio, dict):
        story.append(PageBreak())
        story.append(Paragraph("PORTFOLIO CORRELATION ANALYSIS", S["h_section"]))

        strategies = portfolio.get("strategies", [])
        corr_matrix = portfolio.get("correlation_matrix", [])
        names = portfolio.get("strategy_names", [s.get("name", "S") for s in strategies])
        div_score = float(portfolio.get("diversification_score", 0))
        combined_sh = float(portfolio.get("combined_sharpe", 0))
        combined_dd = float(portfolio.get("combined_drawdown", 0))
        combined_wr = float(portfolio.get("combined_win_rate", 0))
        verdict = _clean_text(portfolio.get("portfolio_verdict", ""))
        div_col = C_OK if div_score >= 70 else C_WARN if div_score >= 45 else C_DANGER

        port_metrics = Table(
            [
                [
                    Table(
                        [
                            [Paragraph("DIVERSIFICATION", S["label"])],
                            [
                                Paragraph(
                                    f"{div_score:.0f}/100",
                                    _s(
                                        "dv",
                                        fontName="Helvetica-Bold",
                                        fontSize=19,
                                        leading=23,
                                        textColor=div_col,
                                    ),
                                )
                            ],
                            [Paragraph("Higher = better", S["body_sm"])],
                        ],
                        colWidths=[36 * mm],
                    ),
                    Table(
                        [
                            [Paragraph("COMBINED SHARPE", S["label"])],
                            [
                                Paragraph(
                                    f"{combined_sh:.2f}",
                                    _s(
                                        "cs",
                                        fontName="Helvetica-Bold",
                                        fontSize=19,
                                        leading=23,
                                        textColor=C_BLACK,
                                    ),
                                )
                            ],
                            [Paragraph("Equal-weight", S["body_sm"])],
                        ],
                        colWidths=[36 * mm],
                    ),
                    Table(
                        [
                            [Paragraph("COMBINED DRAWDOWN", S["label"])],
                            [
                                Paragraph(
                                    f"{abs(combined_dd) * 100:.1f}%",
                                    _s(
                                        "cd",
                                        fontName="Helvetica-Bold",
                                        fontSize=19,
                                        leading=23,
                                        textColor=C_OK if abs(combined_dd) < 0.15 else C_WARN,
                                    ),
                                )
                            ],
                            [Paragraph("Max combined", S["body_sm"])],
                        ],
                        colWidths=[36 * mm],
                    ),
                    Table(
                        [
                            [Paragraph("COMBINED WIN RATE", S["label"])],
                            [
                                Paragraph(
                                    f"{combined_wr:.1f}%",
                                    _s(
                                        "cw",
                                        fontName="Helvetica-Bold",
                                        fontSize=19,
                                        leading=23,
                                        textColor=C_BLACK,
                                    ),
                                )
                            ],
                            [Paragraph("Blended", S["body_sm"])],
                        ],
                        colWidths=[36 * mm],
                    ),
                ]
            ],
            colWidths=[INNER_W / 4] * 4,
        )
        port_metrics.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("LINEAFTER", (0, 0), (2, 0), 0.6, C_PAPER2),
                ]
            )
        )
        story.append(port_metrics)
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(verdict, S["body"]))
        story.append(Spacer(1, 6 * mm))

        # Per-strategy table
        story.append(Paragraph("PER-STRATEGY BREAKDOWN", S["label"]))
        story.append(Spacer(1, 4))
        hdr_s = _s("th", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE)
        strat_rows = [
            [
                Paragraph("Strategy", hdr_s),
                Paragraph(
                    "Score",
                    _s("th2", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE, alignment=TA_CENTER),
                ),
                Paragraph(
                    "Sharpe",
                    _s("th3", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE, alignment=TA_CENTER),
                ),
                Paragraph(
                    "Max DD",
                    _s("th4", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE, alignment=TA_CENTER),
                ),
                Paragraph(
                    "Win Rate",
                    _s("th5", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE, alignment=TA_CENTER),
                ),
                Paragraph(
                    "Trades",
                    _s("th6", fontName="Helvetica-Bold", fontSize=8, textColor=C_WHITE, alignment=TA_CENTER),
                ),
            ]
        ]
        for strat in strategies:
            sc = float(strat.get("fundable_score", 0))
            sc_col = C_OK if sc >= 70 else C_WARN if sc >= 50 else C_DANGER
            strat_rows.append(
                [
                    Paragraph(strat.get("name", "")[:20], S["body_sm"]),
                    Paragraph(
                        f"{sc:.0f}",
                        _s(
                            "sv", fontName="Helvetica-Bold", fontSize=9, textColor=sc_col, alignment=TA_CENTER
                        ),
                    ),
                    Paragraph(
                        f"{strat.get('sharpe', 0):.2f}",
                        _s("sv2", fontSize=9, textColor=C_DARK, alignment=TA_CENTER),
                    ),
                    Paragraph(
                        f"{abs(strat.get('max_drawdown', 0)) * 100:.1f}%",
                        _s("sv3", fontSize=9, textColor=C_DARK, alignment=TA_CENTER),
                    ),
                    Paragraph(
                        f"{strat.get('win_rate', 0):.1f}%",
                        _s("sv4", fontSize=9, textColor=C_DARK, alignment=TA_CENTER),
                    ),
                    Paragraph(
                        str(strat.get("total_trades", 0)),
                        _s("sv5", fontSize=9, textColor=C_DARK, alignment=TA_CENTER),
                    ),
                ]
            )
        strat_tbl = Table(
            strat_rows,
            colWidths=[
                INNER_W * 0.35,
                INNER_W * 0.13,
                INNER_W * 0.13,
                INNER_W * 0.13,
                INNER_W * 0.13,
                INNER_W * 0.13,
            ],
        )
        strat_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_PAPER, C_WHITE]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_PAPER2),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(strat_tbl)
        story.append(Spacer(1, 6 * mm))

        # Correlation heatmap
        if corr_matrix and len(corr_matrix) >= 2:
            story.append(Paragraph("CORRELATION MATRIX", S["label"]))
            story.append(Spacer(1, 4))
            hmap_size = min(INNER_W * 0.5, len(names) * 70 + 60)
            hmap = correlation_heatmap(names, corr_matrix, width=hmap_size, height=hmap_size * 0.7)
            legend_col = [
                Paragraph("READING THE HEATMAP", S["label"]),
                Spacer(1, 6),
                Paragraph("1.00 = perfectly correlated", S["body_sm"]),
                Paragraph("0.00 = no correlation (independent)", S["body_sm"]),
                Paragraph("-1.0 = perfectly inverse", S["body_sm"]),
                Spacer(1, 8),
                Paragraph("GREEN = low/negative correlation", _s("lg", fontSize=8, textColor=C_OK)),
                Paragraph("RED = high positive correlation", _s("lr", fontSize=8, textColor=C_DANGER)),
                Spacer(1, 8),
                Paragraph(
                    "Institutional target: avg off-diagonal < 0.3. "
                    "High correlation = strategies drawdown together.",
                    S["body_sm"],
                ),
            ]
            hmap_tbl = Table([[hmap, legend_col]], colWidths=[hmap_size + 10, INNER_W - hmap_size - 10])
            hmap_tbl.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(hmap_tbl)
            story.append(Spacer(1, 6 * mm))

        # Overlaid equity curves
        story.append(Paragraph("STRATEGY EQUITY CURVES (OVERLAID, NORMALISED TO 100)", S["label"]))
        story.append(Spacer(1, 4))
        story.append(portfolio_combined_curve(strategies, width=INNER_W, height=100))
        story.append(Spacer(1, 2 * mm))

        # Portfolio fix plan
        plan = portfolio.get("portfolio_plan")
        if plan and isinstance(plan, dict):
            story.append(PageBreak())
            story.append(Paragraph("PORTFOLIO FIX PLAN", S["h_section"]))
            story.append(Spacer(1, 4))

            plan_summary = plan.get("summary", "")
            if plan_summary:
                story.append(Paragraph(_clean_text(plan_summary), S["body"]))
                story.append(Spacer(1, 8))

            alloc_verdict = plan.get("allocation_verdict", "")
            if alloc_verdict:
                v_tbl = Table(
                    [
                        [
                            Paragraph("ALLOCATION VERDICT", S["label"]),
                            Paragraph(
                                _clean_text(alloc_verdict),
                                _s("av", fontSize=9, leading=13, textColor=C_DARK, alignment=TA_JUSTIFY),
                            ),
                        ]
                    ],
                    colWidths=[36 * mm, INNER_W - 36 * mm],
                )
                v_tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8EC")),
                            ("TOPPADDING", (0, 0), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LINEAFTER", (0, 0), (0, 0), 2.5, C_WARN),
                        ]
                    )
                )
                story.append(v_tbl)
                story.append(Spacer(1, 6))

            port_fixes = plan.get("portfolio_fixes", [])
            if port_fixes:
                story.append(
                    Paragraph(
                        "RANKED PORTFOLIO FIXES",
                        _s("h_sub", fontName="Helvetica-Bold", fontSize=11, textColor=C_BLACK),
                    )
                )
                story.append(Spacer(1, 6))
                priority_map = {}
                for fix in port_fixes:
                    issue = fix.get("issue", "")
                    if "Coincidence" in issue or "Weak" in issue:
                        priority_map[fix.get("rank", 99)] = C_DANGER
                    elif "Correlation" in issue:
                        priority_map[fix.get("rank", 99)] = C_WARN
                    else:
                        priority_map[fix.get("rank", 99)] = C_LITE

                for fix in sorted(port_fixes, key=lambda x: x.get("rank", 99)):
                    rank = fix.get("rank", "")
                    issue = fix.get("issue", "")
                    why = fix.get("why_it_matters", "")
                    specific = fix.get("specific_fix", "")
                    impact = fix.get("score_impact", "")
                    effort = fix.get("effort", "")
                    badge_col = priority_map.get(rank, C_LITE)
                    header = f"#{rank} - {issue}"
                    if impact:
                        header += f"  |  {impact}"
                    if effort:
                        header += f"  |  Effort: {effort}"
                    fix_row = Table(
                        [
                            [
                                Paragraph(
                                    str(rank),
                                    _s(
                                        "bnum",
                                        fontName="Helvetica-Bold",
                                        fontSize=16,
                                        textColor=badge_col,
                                        alignment=TA_CENTER,
                                        leading=20,
                                    ),
                                ),
                                [
                                    Paragraph(
                                        _clean_text(issue),
                                        _s("ph", fontName="Helvetica-Bold", fontSize=10, textColor=C_BLACK),
                                    ),
                                    Spacer(1, 3),
                                    Paragraph(
                                        f"Impact: {_clean_text(impact)}  |  Effort: {_clean_text(effort)}",
                                        S["label"],
                                    )
                                    if (impact or effort)
                                    else Spacer(1, 1),
                                    Spacer(1, 4),
                                    Paragraph(_clean_text(why)[:220], S["action_b"]) if why else Spacer(1, 1),
                                    Spacer(1, 4),
                                    Paragraph(
                                        f"Fix: {_clean_text(specific)[:220]}",
                                        _s("pf", fontSize=9, leading=13, textColor=C_ORANGE),
                                    )
                                    if specific
                                    else Spacer(1, 1),
                                ],
                            ]
                        ],
                        colWidths=[13 * mm, INNER_W - 13 * mm],
                    )
                    fix_row.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, -1), C_PAPER),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("TOPPADDING", (0, 0), (-1, -1), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                ("LINEAFTER", (0, 0), (0, 0), 2.5, badge_col),
                                ("LINEBELOW", (0, -1), (-1, -1), 0.3, C_PAPER2),
                            ]
                        )
                    )
                    story.append(fix_row)
                    story.append(Spacer(1, 3))

            strongest = plan.get("strongest_combination", "")
            if strongest:
                story.append(Spacer(1, 6))
                story.append(Paragraph("STRONGEST PAIR", S["label"]))
                story.append(
                    Paragraph(
                        _clean_text(strongest),
                        _s("sp", fontSize=9, leading=13, textColor=C_DARK, fontName="Helvetica-Oblique"),
                    )
                )

    # -- SEBI COMPLIANCE PAGE --
    sebi_checks = vr.get("sebi_checks") or {}
    sebi_decl = vr.get("sebi_declaration") or {}  # self-declared answers from frontend

    if sebi_checks:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "SEBI Algo Framework â€” Compliance Pre-Check",
                _s(
                    "h2",
                    fontSize=14,
                    fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#ffaa00"),
                    spaceAfter=4,
                ),
            )
        )
        story.append(
            Paragraph(
                "April 2026 | Kite TradeBook Validation", _s("sp", fontSize=9, textColor=C_DARK, spaceAfter=6)
            )
        )
        story.append(
            Paragraph(
                "3 of 5 checks verified from trade data. 2 are self-declared. "
                "SEBI compliance is shown separately â€” it does not affect the Fundable Score.",
                _s("sp", fontSize=9, leading=13, textColor=C_DARK, spaceAfter=16),
            )
        )

        # Method labels
        METHOD_COLORS = {
            "FROM_DATA": colors.HexColor("#00ff88"),
            "SOFT_CHECK": colors.HexColor("#ffaa00"),
            "SELF_DECLARED": colors.HexColor("#5a7a8a"),
        }

        def _sebi_row(label, method, status_label, status_ok, detail, warning=""):
            method_col = METHOD_COLORS.get(method, C_DARK)
            status_col = colors.HexColor("#00ff88") if status_ok else colors.HexColor("#ff4444")
            if "WARNING" in status_label or "NOT CHECKED" in status_label or "UNCONFIRMED" in status_label:
                status_col = colors.HexColor("#ffaa00")
            method_str = method.replace("_", " ")
            rows = [
                [
                    Paragraph(
                        f"<b>{label}</b><br/><font size=8 color='#5a7a8a'>{detail}</font>"
                        + (f"<br/><font size=8 color='#ff8844'>{warning}</font>" if warning else ""),
                        _s("sp", fontSize=9, leading=13),
                    ),
                    Paragraph(
                        method_str, _s("sp", fontSize=8, textColor=method_col, fontName="Helvetica-Oblique")
                    ),
                    Paragraph(
                        status_label, _s("sp", fontSize=9, fontName="Helvetica-Bold", textColor=status_col)
                    ),
                ]
            ]
            t = Table(rows, colWidths=[310, 90, 80])
            t.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#1e2d3d")),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (0, -1), 0),
                    ]
                )
            )
            return t

        # Header row
        hdr = Table([["CHECK", "METHOD", "RESULT"]], colWidths=[310, 90, 80])
        hdr.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("TEXTCOLOR", (0, 0), (-1, -1), C_DARK),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#1e2d3d")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (0, -1), 0),
                ]
            )
        )
        story.append(hdr)

        # OPS threshold
        ops = sebi_checks.get("ops_threshold", {})
        story.append(
            _sebi_row(
                "OPS Threshold (<10 orders/second)",
                "FROM_DATA",
                "PASS" if ops.get("passed", True) else "FAIL",
                ops.get("passed", True),
                f"Max order frequency: {ops.get('value', 'N/A')} orders/second. "
                + (
                    "Within SEBI limit."
                    if ops.get("passed", True)
                    else "Exceeds 10 OPS â€” mandatory registration required."
                ),
                ""
                if ops.get("passed", True)
                else "Strategies above 10 OPS require mandatory exchange registration.",
            )
        )

        # Session logout
        sess = sebi_checks.get("session_logout", {})
        story.append(
            _sebi_row(
                "Daily Session Logout",
                "FROM_DATA",
                "PASS" if sess.get("passed", True) else "WARNING",
                sess.get("passed", True),
                "No orders detected outside NSE market hours."
                if sess.get("passed", True)
                else f"{sess.get('value', 0)} order(s) detected outside market hours â€” verify API auto-logout.",
                ""
                if sess.get("passed", True)
                else "SEBI requires automated daily logout before the next trading day.",
            )
        )

        # Static IP
        ip = sebi_checks.get("ip_check", {})
        ip_result = ip.get("result", "not_entered")
        ip_label = (
            "PASS" if ip_result == "match" else ("MISMATCH" if ip_result == "mismatch" else "NOT CHECKED")
        )
        ip_ok = ip_result == "match"
        ip_detail = (
            "Current IP matches registered IP. Note: soft check only â€” VPN use would bypass this."
            if ip_result == "match"
            else f"Current IP does not match registered IP ({ip.get('registered_ip', '')})."
            if ip_result == "mismatch"
            else "No registered IP provided. Verify directly with Zerodha that your API app has a registered static IP."
        )
        story.append(
            _sebi_row(
                "Static IP Registration",
                "SOFT_CHECK",
                ip_label,
                ip_ok,
                ip_detail,
                "IP mismatch â€” algo orders must originate from your registered static IP."
                if ip_result == "mismatch"
                else "",
            )
        )

        # Algo-ID (self-declared)
        algo_id = sebi_decl.get("algo_id", "unsure")
        aid_label = (
            "DECLARED" if algo_id == "yes" else ("NOT REGISTERED" if algo_id == "no" else "UNCONFIRMED")
        )
        aid_ok = algo_id == "yes"
        aid_detail = (
            "You declared this strategy has an exchange-assigned Algo-ID. QuantProof cannot independently verify this."
            if algo_id == "yes"
            else "You declared no Algo-ID. SEBI requires all automated strategies to carry an Algo-ID from April 2026."
            if algo_id == "no"
            else "Status unknown â€” contact Zerodha to confirm Algo-ID registration status."
        )
        story.append(
            _sebi_row(
                "Algo-ID Registration",
                "SELF_DECLARED",
                aid_label,
                aid_ok,
                aid_detail,
                "Register with Zerodha before April 1 2026 to receive an Algo-ID." if algo_id == "no" else "",
            )
        )

        # White/black box (self-declared)
        box_type = sebi_decl.get("box_type", "white")
        box_label = "WHITE BOX" if box_type == "white" else "BLACK BOX"
        box_ok = box_type == "white"
        box_detail = (
            "You declared transparent strategy logic. White box strategies can be registered directly through your broker."
            if box_type == "white"
            else "You declared proprietary/hidden logic. Black box algos require a SEBI-registered Research Analyst (RA) license."
        )
        story.append(
            _sebi_row(
                "White Box / Black Box Classification",
                "SELF_DECLARED",
                box_label,
                box_ok,
                box_detail,
                "Consult a SEBI-registered RA for black box compliance requirements."
                if box_type == "black"
                else "",
            )
        )

        story.append(Spacer(1, 16))

        # Score impact box
        base_score = vr.get("fundable_score", 0)
        impact = (
            (8 if algo_id == "no" else 0)
            + (6 if box_type == "black" else 0)
            + (10 if not ops.get("passed", True) else 0)
            + (5 if ip_result == "mismatch" else 0)
        )
        proj_low = max(0, base_score - impact - 5)
        proj_high = max(0, base_score - impact)

        impact_box = Table(
            [
                [
                    Paragraph(
                        f"<b>Score Impact â€” Illustrative Only</b><br/>"
                        f"Current Fundable Score: <b>{base_score}/100</b>. "
                        f"If SEBI checks were factored in, unresolved failures could reduce your score to approximately "
                        f"<b>{proj_low}â€“{proj_high}/100</b>.<br/>"
                        f"<font size=8 color='#5a7a8a'>This is illustrative only. SEBI compliance is not currently factored into the Fundable Score. "
                        f"When full verification infrastructure is available, only verified data will be used â€” not self-declarations.</font>",
                        _s("sp", fontSize=9, leading=13),
                    )
                ]
            ],
            colWidths=[480],
        )
        impact_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a1500")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffaa00")),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(impact_box)
        story.append(Spacer(1, 12))

        # Disclaimer
        story.append(
            Paragraph(
                "DISCLAIMER: QuantProof is not a legal compliance tool. This section is informational only. "
                "3 of 5 checks are verified from trade data. 2 are self-certified by the user â€” QuantProof cannot independently verify them. "
                "Static IP is a soft comparison only â€” QuantProof does not store IP addresses. "
                "Consult SEBI circulars and your broker for authoritative compliance guidance.",
                _s("sp", fontSize=8, leading=12, textColor=C_DARK),
            )
        )

    # -- DISCLAIMER --

    # Prop Firm Compliance Section
    prop_firm_data = vr.get("prop_firm_compliance")
    if prop_firm_data:
        story.append(Spacer(1, 8 * mm))
        story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
        story.append(Spacer(1, 3 * mm))

        # Header
        story.append(Paragraph("PROP FIRM COMPLIANCE", S["label"]))
        story.append(Spacer(1, 4))

        # Summary
        summary = prop_firm_data.get("summary", {})
        story.append(
            Paragraph(
                f"Total Firms Analyzed: {summary.get('total_firms', 0)} | "
                f"Eligible Firms: {summary.get('eligible_firms', 0)} | "
                f"Average Score: {summary.get('average_score', 0):.1f}/100",
                S["body"],
            )
        )
        story.append(Spacer(1, 6))

        # Firm-by-firm breakdown
        firms = prop_firm_data.get("firms", {})
        for firm_name, firm_data in firms.items():
            eligible = firm_data.get("eligible", False)
            score = firm_data.get("score", 0)
            violations = firm_data.get("violations", [])
            passes = firm_data.get("passes", [])

            # Firm header
            firm_color = C_OK if eligible else C_DANGER
            story.append(
                Paragraph(
                    f"{firm_name.upper()} - {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'} (Score: {score}/100)",
                    _s("label", fontSize=10, textColor=firm_color, leading=12),
                )
            )
            story.append(Spacer(1, 2))

            # Violations and passes
            if violations:
                story.append(Paragraph("Violations:", _s("body_sm", fontSize=9, textColor=C_DANGER)))
                for violation in violations:
                    story.append(Paragraph(f"• {violation}", _s("body_sm", fontSize=8, textColor=C_DARK)))
                story.append(Spacer(1, 2))

            if passes:
                story.append(Paragraph("Passes:", _s("body_sm", fontSize=9, textColor=C_OK)))
                for pass_item in passes:
                    story.append(Paragraph(f"• {pass_item}", _s("body_sm", fontSize=8, textColor=C_DARK)))
                story.append(Spacer(1, 4))

        # Disclaimer
        story.append(
            Paragraph(
                "DISCLAIMER: Prop firm compliance analysis is based on publicly available rules and requirements. "
                "Individual firms may have additional criteria not captured here. Always verify with the specific prop firm. "
                "This analysis is for informational purposes only and does not guarantee acceptance.",
                _s("sp", fontSize=8, leading=12, textColor=C_DARK),
            )
        )
        story.append(Spacer(1, 6 * mm))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_PAPER2))
    story.append(Spacer(1, 3 * mm))
    vh = vr.get("validation_hash", f"QP-{abs(hash(str(score) + grade + str(n))) % 100000:05d}")
    story.append(
        Paragraph(
            f"Report ID: {vh}  |  Engine: v6.3  |  37 scored checks  |  5 crash simulations  |  "
            f"Depth: {depth_short}  |  "
            f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            S["footer"],
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        Paragraph(
            "For educational and analytical purposes only. Not financial advice. "
            "Past backtest performance does not guarantee future results. "
            "Methodology: Lopez de Prado (2018) AFML, Almgren-Chriss (2001), Bailey & Lopez de Prado (2012).",
            S["footer"],
        )
    )

    doc.build(story, onLaterPages=footer_canvas, onFirstPage=footer_canvas)
    return buf.getvalue()
