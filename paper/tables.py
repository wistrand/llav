#!/usr/bin/env python3
"""Writes the paper's tables (tables/*.tex) from paper/data/summary.json, so a recompute changes them with
the figures. main.tex inputs them."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "data/summary.json"))["models"]
ORDER = ["qwen4b", "muse", "gptoss", "granite3b", "granitetiny", "smollm3", "qwen2b"]
HEAD = {"qwen4b": "Qwen 4B", "muse": "Muse 30B", "gptoss": "gpt-oss 20B", "granite3b": "Granite 3B",
        "granitetiny": "Granite Tiny", "smollm3": "SmolLM3 3B", "qwen2b": "Qwen 2B"}
SOURCES = [("dbpedia", "DBpedia"), ("ag_news", "AG News"), ("mnli", "MNLI"), ("semif_evidence_interpretation", "SemIf evidence"),
           ("semif_rule_application", "SemIf rules"), ("clinc150", "CLINC150"), ("trec_fine", "TREC fine"),
           ("banking_cards", "Banking77 cards"), ("banking77", "Banking77"), ("goemotions", "GoEmotions")]
pct = lambda v: f"{v * 100:.1f}\\%"
pct0 = lambda v: f"{v * 100:.0f}\\%"
f3 = lambda v: f"{v:.3f}"
ci = lambda c, f: f"{f(c[1])} to {f(c[2])}"
rel = lambda v: f"${v * 100:+.0f}\\%$"


def row(label, fn):
    return label + " & " + " & ".join(fn(S[k]) for k in ORDER) + " \\\\\n"


def head(spec):
    return "\\begin{tabular}{" + spec + "}\n\\toprule\n & " + " & ".join(HEAD[k] for k in ORDER) + " \\\\\n\\midrule\n"


tail = "\\bottomrule\n\\end{tabular}\n"
(HERE / "tables").mkdir(exist_ok=True)

# Main text: the compact table
out = head("@{}l" + "r" * 7 + "@{}")
out += row("Wrong, base order", lambda e: pct(e["wrong"]))
out += row("Wrong, averaged", lambda e: pct(e["wrong_avg"]))
out += row("\\quad relative change", lambda e: rel(e["wrong_change"][0]))
out += row("\\quad 95\\% interval", lambda e: f"${e['wrong_change'][1] * 100:+.0f}, {e['wrong_change'][2] * 100:+.0f}$")
out += "\\midrule\n"
out += row("Confident answers: share that change", lambda e: pct0(e["flips"] / e["confident"]))
out += row("Adjusted risk ratio", lambda e: f"{e['adj_rr'][0]:.2f}")
out += row("\\quad 95\\% interval", lambda e: f"{e['adj_rr'][1]:.2f}--{e['adj_rr'][2]:.2f}")
out += "\\midrule\n"
out += row("ChaosNLI divergence, base order", lambda e: f3(e["jsd"][0]))
out += row("ChaosNLI divergence, averaged", lambda e: f3(e["jsd"][1]))
out += row("\\quad relative change", lambda e: rel(e["jsd_change"][0]))
out += row("\\quad 95\\% interval", lambda e: f"${e['jsd_change'][1] * 100:+.0f}, {e['jsd_change'][2] * 100:+.0f}$")
out += tail
(HERE / "tables/main.tex").write_text(out)

# Appendix: every signal, both error targets, the confident subgroup
out = head("@{}l" + "r" * 7 + "@{}")
out += "\\multicolumn{8}{@{}l}{AUROC for a wrong base-order answer} \\\\\n"
out += row("\\quad base-order confidence", lambda e: f3(e["auroc_conf"]))
out += row("\\quad averaged confidence", lambda e: f3(e["auroc_conf_avg"]))
out += row("\\quad spread", lambda e: f3(e["auroc_spread"]))
out += row("\\quad flip", lambda e: f3(e["auroc_flip"]))
out += row("\\quad candidate mass", lambda e: f3(e["auroc_mass"]))
out += row("\\quad spread added, held-out gain", lambda e: f"${e['h1'][0]:+.4f}$")
out += row("\\quad flip added, held-out gain", lambda e: f"${e['flip_gain'][0]:+.4f}$")
out += "\\multicolumn{8}{@{}l}{AUROC for a wrong averaged answer} \\\\\n"
out += row("\\quad base-order confidence", lambda e: f3(e["auroc_conf_t"]))
out += row("\\quad averaged confidence", lambda e: f3(e["auroc_conf_avg_t"]))
out += row("\\quad spread", lambda e: f3(e["auroc_spread_t"]))
out += row("\\quad spread added, held-out gain", lambda e: f"${e['h1_t'][0]:+.4f}$")
out += row("\\quad flip added, held-out gain", lambda e: f"${e['flip_gain_t'][0]:+.4f}$")
out += "\\midrule\n"
out += row("Answers at base-order confidence $\\ge 0.9$", lambda e: pct0(e["confident_share"]))
out += row("\\quad of which changing under some order", lambda e: pct0(e["flips"] / e["confident"]))
out += row("\\quad wrong when changing", lambda e: pct(e["wrong_flipped"]))
out += row("\\quad wrong when stable", lambda e: pct(e["wrong_stable"]))
out += row("\\quad wrong, all confident answers", lambda e: pct(e["abstain"]["wrong_all"]))
out += row("Adjusted odds ratio", lambda e: f"{e['adj_or'][0]:.2f}")
out += row("Adjusted risk difference", lambda e: f"${e['adj_rd'][0]:+.3f}$")
out += row("Mantel--Haenszel odds ratio", lambda e: f"{e['mh_or'][0]:.2f}")
out += row("\\quad 95\\% interval", lambda e: f"{e['mh_or'][1]:.2f}--{e['mh_or'][2]:.2f}")
out += tail
(HERE / "tables/signals.tex").write_text(out)

# Appendix: ChaosNLI per model
out = "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n & MNLI wrong & Changing: wrong & Stable: wrong & AUROC, flip & AUROC, averaged conf. \\\\\n\\midrule\n"
for k in ORDER:
    e = S[k]
    c = e["chaos"]
    fl, st = c["flipped"], c["stable"]
    out += (f"{HEAD[k]} & {pct(e['per_source']['mnli']['wrong'])} & {fl[0]}/{fl[1]} ({fl[0] / fl[1] * 100:.0f}\\%) & "
            f"{st[0]}/{st[1]} ({st[0] / st[1] * 100:.0f}\\%) & {c['auroc_contested']['flip']:.2f} & {c['auroc_contested']['conf_avg']:.2f} \\\\\n")
out += tail
(HERE / "tables/chaos.tex").write_text(out)

# Appendix: per-source Mantel-Haenszel odds ratios with the number of confident flips
out = "\\begin{tabular}{@{}l" + "r" * 7 + "@{}}\n\\toprule\nSource & " + " & ".join(HEAD[k] for k in ORDER) + " \\\\\n\\midrule\n"
for src, name in SOURCES:
    cells = []
    for k in ORDER:
        v = S[k]["per_source_or"].get(src)
        if v is None:
            cells.append("--")
        elif v["or"] is None:
            cells.append(f"-- ({v['flips']})")
        else:
            star = "" if v["low"] > 1 or v["high"] < 1 else "$^{\\dagger}$"
            cells.append(f"{v['or']:.1f} ({v['flips']}){star}")
    out += name + " & " + " & ".join(cells) + " \\\\\n"
out += tail
(HERE / "tables/persource.tex").write_text(out)
print("wrote tables/main.tex signals.tex chaos.tex persource.tex")
