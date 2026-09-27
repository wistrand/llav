#!/usr/bin/env python3
"""Draws the paper's figures as SVG from paper/data (standard library only); paper/build.sh converts them."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "data/summary.json"))
ORDER = ["qwen4b", "muse", "gptoss", "granite3b", "granitetiny", "smollm3", "qwen2b"]
SHORT = {"qwen4b": "Qwen3.5-4B", "muse": "Muse Glimmer 30B", "gptoss": "gpt-oss-20b", "granite3b": "Granite 4.2 3B",
         "granitetiny": "Granite 4.0 H Tiny", "smollm3": "SmolLM3-3B", "qwen2b": "Qwen3.5-2B"}
SRC = {"dbpedia": "DBpedia (14)", "ag_news": "AG News (4)", "mnli": "MNLI (3)", "semif_evidence_interpretation": "SemIf evidence (3)",
       "semif_rule_application": "SemIf rules (3)", "clinc150": "CLINC150 (12)", "trec_fine": "TREC fine (2-17)",
       "banking_cards": "Banking77 cards (13)", "banking77": "Banking77 (12)", "goemotions": "GoEmotions (12)"}
BLUE, ORANGE, GRAY, INK, INK2, GRID, TINT = "#2a78d6", "#eb6834", "#c3c2b7", "#1d1f23", "#52514e", "#e8e8e4", "#f3f3ef"
FONT = "font-family='Helvetica, Arial, sans-serif'"


def svg(width, height, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            f"{FONT} font-size='9'>\n<rect width='{width}' height='{height}' fill='white'/>\n{body}</svg>\n")


def text(x, y, s, size=9, anchor="start", fill=INK, weight="normal", rotate=None):
    tr = f" transform='rotate({rotate} {x} {y})'" if rotate else ""
    return f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' fill='{fill}' font-weight='{weight}'{tr}>{s}</text>\n"


def dot(x, y, color):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.6' fill='{color}' stroke='white' stroke-width='1'/>\n"


def tri(x, y, color):
    return (f"<polygon points='{x:.1f},{y-4.6:.1f} {x+4.2:.1f},{y+2.8:.1f} {x-4.2:.1f},{y+2.8:.1f}' "
            f"fill='{color}' stroke='white' stroke-width='1'/>\n")


def fig1():
    """Wrong-answer rate by base-order confidence band, stable (circle) against flipped (triangle), one panel per
    model; the confident region the flag is used in is tinted."""
    bands = ["0.0-0.5", "0.5-0.7", "0.7-0.9", "0.9-0.97", "0.97-0.99", "0.99-0.999", "0.999-1"]
    labels = ["&lt;.5", ".5-.7", ".7-.9", ".9-.97", ".97-.99", ".99-.999", "&gt;.999"]
    cols, rows, pw, ph, gapx, gapy, ml, mt = 3, 3, 150, 92, 22, 44, 30, 16
    W = ml + cols * pw + (cols - 1) * gapx + 30
    H = mt + rows * ph + (rows - 1) * gapy + 30
    body = ""
    for i, key in enumerate(ORDER):
        e = S["models"][key]
        c, r = i % cols, i // cols
        x0 = ml + c * (pw + gapx)
        y0 = mt + r * (ph + gapy)
        sy = lambda v: y0 + ph - v * ph
        sx = lambda j: x0 + 12 + j * (pw - 24) / (len(bands) - 1)
        # the region the flag is used in: base-order confidence 0.9 and up
        body += f"<rect x='{sx(3) - 10:.1f}' y='{y0}' width='{sx(6) - sx(3) + 20:.1f}' height='{ph}' fill='{TINT}'/>\n"
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            body += f"<line x1='{x0}' y1='{sy(v):.1f}' x2='{x0+pw}' y2='{sy(v):.1f}' stroke='{GRID}' stroke-width='1'/>\n"
            if c == 0:
                body += text(x0 - 4, sy(v) + 3, f"{int(v*100)}%", 8, "end", INK2)
        body += f"<line x1='{x0}' y1='{sy(0):.1f}' x2='{x0+pw}' y2='{sy(0):.1f}' stroke='{GRAY}'/>\n"
        for j, lab in enumerate(labels):
            body += text(sx(j) + 3, y0 + ph + 9, lab, 7.5, "end", INK2, rotate=-40)
        bs = {b["band"]: b for b in e["bands"]}
        for j, band in enumerate(bands):
            b = bs[band]
            if b["n_stable"] >= 10:
                body += dot(sx(j) - 3, sy(b["wrong_stable"]), BLUE)
            if b["n_flipped"] >= 10:
                body += tri(sx(j) + 3, sy(b["wrong_flipped"]), ORANGE)
                body += text(sx(j) + 3, sy(b["wrong_flipped"]) - 7, f"{b['n_flipped']:,}", 6.5, "middle", INK2)
        body += text(x0, y0 - 5, SHORT[key], 9, "start", INK, "bold")
        body += text(x0 + pw, y0 - 5, f"n={e['n']:,}", 8, "end", INK2)
    # legend in the empty ninth cell
    lx, ly = ml + 2 * (pw + gapx), mt + 2 * (ph + gapy) + 8
    body += dot(lx + 6, ly, BLUE) + text(lx + 16, ly + 3, "same answer in every order", 8.5)
    body += tri(lx + 6, ly + 16, ORANGE) + text(lx + 16, ly + 19, "answer changes under some order", 8.5)
    body += text(lx + 16, ly + 30, "(number: answers in the changing group)", 7.5, "start", INK2)
    body += f"<rect x='{lx}' y='{ly+40}' width='12' height='9' fill='{TINT}'/>" + text(lx + 16, ly + 48, "base-order confidence 0.9 and up", 8.5)
    body += text(lx, ly + 64, "x: probability of the answer in the base order;", 8, "start", INK2)
    body += text(lx, ly + 75, "y: share of those answers that are wrong.", 8, "start", INK2)
    body += text(lx, ly + 86, "Groups with fewer than 10 answers are omitted.", 8, "start", INK2)
    return svg(W, H, body)


def fig2():
    """Flip rate by source: the min-max over the seven models as a band, the 4B and the 30B marked."""
    srcs = sorted(SRC, key=lambda s: S["models"]["qwen4b"]["per_source"][s]["flip_rate"])
    ml, rh, W = 122, 18, 400
    H = 32 + len(srcs) * rh + 32
    x0, x1 = ml, W - 14
    sx = lambda v: x0 + v * (x1 - x0)
    body = ""
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        body += f"<line x1='{sx(v):.1f}' y1='26' x2='{sx(v):.1f}' y2='{H-30}' stroke='{GRID}'/>\n"
        body += text(sx(v), H - 18, f"{int(v*100)}%", 8.5, "middle", INK2)
    for i, s in enumerate(srcs):
        y = 32 + i * rh + rh / 2
        vals = [S["models"][k]["per_source"][s]["flip_rate"] for k in ORDER if s in S["models"][k]["per_source"]]
        body += f"<line x1='{sx(min(vals)):.1f}' y1='{y}' x2='{sx(max(vals)):.1f}' y2='{y}' stroke='{GRAY}' stroke-width='6' stroke-linecap='round'/>\n"
        body += f"<circle cx='{sx(S['models']['muse']['per_source'][s]['flip_rate']):.1f}' cy='{y}' r='4' fill='white' stroke='{ORANGE}' stroke-width='2'/>\n"
        body += f"<circle cx='{sx(S['models']['qwen4b']['per_source'][s]['flip_rate']):.1f}' cy='{y}' r='4' fill='{BLUE}'/>\n"
        body += text(ml - 6, y + 3, SRC[s], 9, "end")
    body += text(x0, H - 5, "questions whose answer changes under up to 6 additional random orders", 8.5, "start", INK2)
    ly = 13
    body += f"<circle cx='{x0+6}' cy='{ly-3}' r='4' fill='{BLUE}'/>" + text(x0 + 13, ly, "Qwen3.5-4B", 8.5, "start", INK2)
    body += f"<circle cx='{x0+76}' cy='{ly-3}' r='4' fill='white' stroke='{ORANGE}' stroke-width='2'/>" + text(x0 + 83, ly, "Muse Glimmer 30B", 8.5, "start", INK2)
    body += f"<line x1='{x0+172}' y1='{ly-3}' x2='{x0+192}' y2='{ly-3}' stroke='{GRAY}' stroke-width='6' stroke-linecap='round'/>" + text(x0 + 197, ly, "min-max, 7 models", 8.5, "start", INK2)
    return svg(W, H, body)


def fig3():
    """Jensen-Shannon divergence from ChaosNLI's 100-annotator distribution: base order to averaged."""
    ml, rh, W = 122, 19, 400
    keys = sorted(ORDER, key=lambda k: S["models"][k]["jsd"][1])
    H = 22 + len(keys) * rh + 32
    x0, x1 = ml, W - 46
    sx = lambda v: x0 + v / 0.36 * (x1 - x0)
    body = ""
    for v in (0, 0.1, 0.2, 0.3):
        body += f"<line x1='{sx(v):.1f}' y1='16' x2='{sx(v):.1f}' y2='{H-30}' stroke='{GRID}'/>\n"
        body += text(sx(v), H - 18, f"{v:.1f}", 8.5, "middle", INK2)
    for i, k in enumerate(keys):
        y = 22 + i * rh + rh / 2
        a, b = S["models"][k]["jsd"]
        body += f"<line x1='{sx(a):.1f}' y1='{y}' x2='{sx(b):.1f}' y2='{y}' stroke='{GRAY}' stroke-width='2'/>\n"
        body += f"<circle cx='{sx(a):.1f}' cy='{y}' r='4' fill='white' stroke='{INK2}' stroke-width='1.5'/>\n"
        body += f"<circle cx='{sx(b):.1f}' cy='{y}' r='4' fill='{BLUE}'/>\n"
        body += text(sx(max(a, b)) + 8, y + 3, f"−{(1-b/a)*100:.0f}%", 8, "start", INK2)
        body += text(ml - 6, y + 3, SHORT[k], 9, "end")
    body += text(x0, H - 5, "Jensen-Shannon divergence from the human distribution (bits)", 8.5, "start", INK2)
    body += f"<circle cx='{x0+8}' cy='9' r='4' fill='white' stroke='{INK2}' stroke-width='1.5'/>" + text(x0 + 15, 12, "base order", 8.5, "start", INK2)
    body += f"<circle cx='{x0+78}' cy='9' r='4' fill='{BLUE}'/>" + text(x0 + 85, 12, "averaged over the fixed budget", 8.5, "start", INK2)
    return svg(W, H, body)


def fig4():
    """For a wrong averaged answer: the held-out AUROC gain of spread over averaged confidence (interval), beside
    the direct AUROC difference between averaged and base-order confidence."""
    ml, rh, W = 122, 19, 400
    keys = ORDER
    H = 36 + len(keys) * rh + 32
    x0, x1 = ml, W - 14
    lo, hi = -0.012, 0.10
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    body = ""
    for v in (-0.01, 0, 0.02, 0.04, 0.06, 0.08, 0.10):
        body += f"<line x1='{sx(v):.1f}' y1='32' x2='{sx(v):.1f}' y2='{H-30}' stroke='{GRID if v else GRAY}'/>\n"
        body += text(sx(v), H - 18, f"{v:+.2f}" if v else "0", 8.5, "middle", INK2)
    for i, k in enumerate(keys):
        y = 36 + i * rh + rh / 2
        e = S["models"][k]
        g, l, h = e["h1_t"]
        body += f"<line x1='{sx(l):.1f}' y1='{y}' x2='{sx(h):.1f}' y2='{y}' stroke='{ORANGE}' stroke-width='2'/>\n"
        body += f"<circle cx='{sx(g):.1f}' cy='{y}' r='3.6' fill='{ORANGE}'/>\n"
        d = e["auroc_conf_avg_t"] - e["auroc_conf_t"]
        body += f"<circle cx='{sx(d):.1f}' cy='{y}' r='3.6' fill='{BLUE}'/>\n"
        body += text(ml - 6, y + 3, SHORT[k], 9, "end")
    body += text(x0, H - 5, "change in AUROC for ranking a wrong averaged answer", 8.5, "start", INK2)
    body += f"<circle cx='{x0+6}' cy='9' r='3.6' fill='{ORANGE}'/>" + text(x0 + 13, 12, "spread added to the averaged confidence (held out, 95% CI)", 8.5, "start", INK2)
    body += f"<circle cx='{x0+6}' cy='23' r='3.6' fill='{BLUE}'/>" + text(x0 + 13, 26, "averaged confidence instead of base-order confidence", 8.5, "start", INK2)
    return svg(W, H, body)


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    for name, fn in (("fig1-bands", fig1), ("fig2-flips", fig2), ("fig3-jsd", fig3), ("fig4-h1", fig4)):
        (HERE / "figures" / f"{name}.svg").write_text(fn())
        print("wrote", name)
