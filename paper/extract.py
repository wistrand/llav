#!/usr/bin/env python3
"""Collects every number the paper reports from the result files into paper/data/summary.json and the
figure data files. Rerun after any run changes; the text and figures read from these."""
import json, math, re, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import perturb  # noqa: E402
perturb.BUDGET = "fixed"  # the paper's estimand: the base order and the random permutations

R = ROOT / "local/results/perturb"  # the raw run outputs; gitignored, available from the author (1.3 GB)
F = R / "fixed"  # analyses rerun with --budget fixed by fixed/recompute.sh
MODELS = [  # key, label, family, size label, run file, analysis dir
    ("qwen4b", "Qwen3.5-4B", "Qwen", "4B", R / "full.jsonl", R),
    ("qwen2b", "Qwen3.5-2B", "Qwen", "2B", R / "models/m-qwen3.5-2b.jsonl", R / "models"),
    ("granite3b", "Granite 4.2 3B", "Granite", "3B", R / "models/m-granite-4.2-3b.jsonl", R / "models"),
    ("granitetiny", "Granite 4.0 H Tiny", "Granite", "7B/1B", R / "models/m-granite-4.0-h-tiny.jsonl", R / "models"),
    ("smollm3", "SmolLM3-3B", "SmolLM", "3B", R / "models/m-smollm3-3b.jsonl", R / "models"),
    ("gptoss", "gpt-oss-20b", "gpt-oss", "21B/4B", R / "big/m-gpt-oss-20b.jsonl", R / "big"),
    ("muse", "Muse Glimmer 30B", "Muse", "30B", R / "big/m-muse-glimmer-30b.jsonl", R / "big"),
]
BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.97), (0.97, 0.99), (0.99, 0.999), (0.999, 1.01)]
SOURCES = ["dbpedia", "ag_news", "mnli", "semif_evidence_interpretation", "semif_rule_application", "clinc150",
           "trec_fine", "banking_cards", "banking77", "goemotions"]

def stem(key, path):
    return "full" if key == "qwen4b" else path.name[:-len(".jsonl")]

def parse_text(path, pattern, group=1, cast=float):
    text = path.read_text() if path.exists() else ""
    m = re.search(pattern, text)
    return cast(m.group(group)) if m else None

summary = {"models": {}}
bands_rows, flips_rows, h1_rows, jsd_rows = [], [], [], []
for key, label, family, size, run, adir in MODELS:
    rows = [perturb.features(r) for r in map(json.loads, open(run)) if not perturb.fixed_order(r)]
    wrong = [r["wrong"] for r in rows]
    wrong_avg = [r["wrong_avg"] for r in rows]
    s = stem(key, run)
    repeated = [r for r in rows if r["noise"] is not None]
    conf = [r for r in rows if r["conf"] >= 0.9]
    fl = [r for r in conf if r["changes_fixed"] > 0]; st = [r for r in conf if r["changes_fixed"] == 0]
    entry = {
        "label": label, "family": family, "size": size, "n": len(rows),
        "wrong": statistics.fmean(wrong), "wrong_avg": statistics.fmean(r["wrong_avg"] for r in rows),
        "flip_rate": statistics.fmean(r["changes_fixed"] > 0 for r in rows),
        "flip_rate_any": statistics.fmean(r["changes"] > 0 for r in rows),
        # AUROC against the base-order answer being wrong, and against the averaged answer being wrong
        "auroc_conf": perturb.auroc([-r["conf"] for r in rows], wrong),
        "auroc_conf_avg": perturb.auroc([-r["conf_avg"] for r in rows], wrong),
        "auroc_spread": perturb.auroc([r["spread"] for r in rows], wrong),
        "auroc_mass": perturb.auroc([-(r["mass"] if r["mass"] is not None else 1.0) for r in rows], wrong),
        "auroc_flip": perturb.auroc([r["changes_fixed"] for r in rows], wrong),
        "auroc_conf_avg_t": perturb.auroc([-r["conf_avg"] for r in rows], wrong_avg),
        "auroc_spread_t": perturb.auroc([r["spread"] for r in rows], wrong_avg),
        "auroc_conf_t": perturb.auroc([-r["conf"] for r in rows], wrong_avg),
        "orders_median": statistics.median(r["passes"] for r in rows),
        "repeated": len(repeated),
        "noise_spread": statistics.fmean(r["noise"] for r in repeated) if repeated else None,
        "noise_changes": statistics.fmean(r["noise_changes"] > 0 for r in repeated) if repeated else None,
        "perm_spread": statistics.fmean(r["spread"] for r in rows),
        "confident": len(conf), "confident_share": len(conf) / len(rows), "flips": len(fl),
        "wrong_flipped": sum(r["wrong"] for r in fl) / len(fl), "wrong_stable": sum(r["wrong"] for r in st) / len(st),
        "median_conf": statistics.median(r["conf"] for r in rows),
        "per_source": {}, "bands": [],
    }
    for src in SOURCES:
        g = [r for r in rows if r["source"] == src]
        if g:
            entry["per_source"][src] = {"n": len(g), "wrong": statistics.fmean(r["wrong"] for r in g),
                                        "wrong_avg": statistics.fmean(r["wrong_avg"] for r in g),
                                        "flip_rate": statistics.fmean(r["changes_fixed"] > 0 for r in g),
                                        "options": g[0]["options"]}
            flips_rows.append((key, src, entry["per_source"][src]["flip_rate"]))
    for lo, hi in BANDS:
        b = [r for r in rows if lo <= r["conf"] < hi]
        sb = [r for r in b if r["changes_fixed"] == 0]; fb = [r for r in b if r["changes_fixed"] > 0]
        cell = {"band": f"{lo}-{min(hi, 1)}", "n_stable": len(sb), "n_flipped": len(fb),
                "wrong_stable": statistics.fmean(r["wrong"] for r in sb) if sb else None,
                "wrong_flipped": statistics.fmean(r["wrong"] for r in fb) if fb else None}
        entry["bands"].append(cell)
        bands_rows.append((key, cell))
    # From the analysis text files: H1 gains, adjusted H2, ChaosNLI.
    an = F / f"{s}-analysis.txt"
    h1 = re.search(r"wrong\s+1-conf_avg\s+\+ spread: ([+-][0-9.]+)\s+\[([+-][0-9.]+), ([+-][0-9.]+)\]", an.read_text())
    entry["h1"] = [float(h1.group(i)) for i in (1, 2, 3)] if h1 else None  # target: base-order answer wrong
    h1t = re.search(r"wrong_avg\s+1-conf_avg\s+\+ spread: ([+-][0-9.]+)\s+\[([+-][0-9.]+), ([+-][0-9.]+)\]", an.read_text())
    entry["h1_t"] = [float(h1t.group(i)) for i in (1, 2, 3)] if h1t else None  # target: averaged answer wrong
    h1m = re.search(r"wrong_avg\s+-margin_avg\s+\+ spread: ([+-][0-9.]+)\s+\[([+-][0-9.]+), ([+-][0-9.]+)\]", an.read_text())
    entry["h1_margin_t"] = [float(h1m.group(i)) for i in (1, 2, 3)] if h1m else None
    a = an.read_text()
    m = re.search(r"relative change ([+-][0-9.]+)% \[([+-][0-9.]+)%, ([+-][0-9.]+)%\]", a)
    entry["wrong_change"] = [float(m.group(i)) / 100 for i in (1, 2, 3)]  # paired bootstrap over questions
    entry["wrong_mean"] = float(re.search(r"mean single-order wrong-answer rate ([0-9.]+)", a).group(1))
    m = re.search(r"wrong\s+1-conf\s+\+ flip: ([+-][0-9.]+)\s+\[([+-][0-9.]+), ([+-][0-9.]+)\]", a)
    entry["flip_gain"] = [float(m.group(i)) for i in (1, 2, 3)]
    m = re.search(r"wrong_avg\s+1-conf_avg\s+\+ flip: ([+-][0-9.]+)\s+\[([+-][0-9.]+), ([+-][0-9.]+)\]", a)
    entry["flip_gain_t"] = [float(m.group(i)) for i in (1, 2, 3)]
    m = re.search(r"confident (\d+) wrong ([0-9.]+); kept (\d+) \(([0-9.]+)%\) wrong ([0-9.]+)", a)
    entry["abstain"] = {"confident": int(m.group(1)), "wrong_all": float(m.group(2)), "kept": int(m.group(3)),
                        "coverage": float(m.group(4)) / 100, "wrong_kept": float(m.group(5))}
    h1_rows.append((key, entry["h1_t"]))
    h2 = F / f"{s}-h2.txt"
    text = h2.read_text() if h2.exists() else ""
    block = text.split("flip across base + random permutations")[-1]
    entry["adj_rr"] = [float(x) for x in re.search(r"risk ratio ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", block).groups()]
    entry["adj_or"] = [float(x) for x in re.search(r"odds ratio ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", block).groups()]
    entry["adj_rd"] = [float(x) for x in re.search(r"risk difference ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", block).groups()]
    entry["mh_or"] = [float(x) for x in re.search(r"Mantel-Haenszel over source x confidence decile: odds ratio ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", block).groups()]
    entry["per_source_or"] = {}
    for src, flips, o, lo, hi in re.findall(r"^\s+(\w+)\s+flips\s+(\d+)\s+odds ratio\s+(nan|[0-9.]+) \[(nan|[0-9.]+), (nan|[0-9.]+)\]", block, re.M):
        entry["per_source_or"][src] = {"flips": int(flips), "or": None if o == "nan" else float(o),
                                       "low": None if lo == "nan" else float(lo), "high": None if hi == "nan" else float(hi)}
    loso = re.search(r"leave one source out, adjusted odds ratio / risk ratio: (.*)", block).group(1)
    ors = [float(x.split("/")[0]) for x in re.findall(r" ([0-9.]+/[0-9.]+)", loso)]
    entry["loso_or"] = [min(ors), max(ors)]
    # Sensitivity fits (fixed/sens/run.sh): other confidence cutoffs and a flip x confidence interaction
    entry["sensitivity"] = {}
    for tag, name in (("c080", "cutoff_0.8"), ("c095", "cutoff_0.95"), ("inter", "interaction_0.9")):
        sp = F / "sens" / f"{s}-{tag}.txt"
        if sp.exists():
            blk = sp.read_text().split("flip across base + random permutations")[-1]
            m = re.search(r"risk ratio ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", blk)
            n = re.search(r"^(\d+) base-order answers", sp.read_text(), re.M)
            entry["sensitivity"][name] = {"rr": [float(x) for x in m.groups()], "n": int(n.group(1))}
    hc = F / f"{s}-human-chaos.txt"
    m = re.search(r"base order ([0-9.]+), averaged over orders ([0-9.]+)", hc.read_text())
    entry["jsd"] = [float(m.group(1)), float(m.group(2))]  # Jensen-Shannon divergence, bits
    m = re.search(r"divergence change from averaging: ([+-][0-9.]+)% \[([+-][0-9.]+)%, ([+-][0-9.]+)%\]", hc.read_text())
    entry["jsd_change"] = [float(m.group(i)) / 100 for i in (1, 2, 3)]
    entry["jsd_mean"] = float(re.search(r"mean single-order divergence ([0-9.]+)", hc.read_text()).group(1))
    m = re.search(r"agreement >= 80% \(confidence >= 0.9\): flipped (\d+)/(\d+) wrong \(([0-9.]+)%\), stable (\d+)/(\d+) \(([0-9.]+)%\)", hc.read_text())
    a = re.search(r"AUROC for contested: flip ([0-9.]+), spread ([0-9.]+), 1-conf ([0-9.]+), 1-conf_avg ([0-9.]+)", hc.read_text())
    entry["chaos"] = {"flipped": [int(m.group(1)), int(m.group(2))] if m else None, "stable": [int(m.group(4)), int(m.group(5))] if m else None,
                      "auroc_contested": dict(zip(("flip", "spread", "conf", "conf_avg"), map(float, a.groups())))}
    mnli = entry["per_source"]["mnli"]
    jsd_rows.append((key, entry["jsd"]))
    summary["models"][key] = entry

# The 4B-only arms, from their result texts.
summary["text_arm"] = {}
for prompt in ("bare", "text"):
    t = (R / f"text/text-{prompt}-analysis.txt").read_text()
    allrow = re.search(r"^all\s+(\d+)\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%", t, re.M)
    agree = re.search(r"text answer = readout answer: ([0-9.]+)%", t)
    flipw = re.search(r"text flips: text answer wrong ([0-9.]+)% when it flips \((\d+)\) vs ([0-9.]+)% when stable \((\d+)\)", t)
    summary["text_arm"][prompt] = {"wrong": float(allrow.group(2)) / 100, "wrong_majority": float(allrow.group(3)) / 100,
                                   "flip_rate": float(allrow.group(4)) / 100, "agree_readout": float(agree.group(1)) / 100,
                                   "wrong_flipped": float(flipw.group(1)) / 100, "wrong_stable": float(flipw.group(3)) / 100,
                                   "n_flipped": int(flipw.group(2)), "n_stable": int(flipw.group(4))}
rw = (F / "reword-analysis.txt").read_text()
cells = re.findall(r"order (stable|flips ), reword (stable|flips ): n=\s*(\d+)\s+wrong\s+([0-9.]+)%", rw)
summary["reword"] = {"all": {}, "confident": {}}
for i, (o, w, n, pct) in enumerate(cells):
    summary["reword"]["all" if i < 4 else "confident"][f"order_{o.strip()}_reword_{w.strip()}"] = {"n": int(n), "wrong": float(pct) / 100}
m = re.search(r"order flip ([0-9.]+), reword flip ([0-9.]+), order spread ([0-9.]+), reword spread ([0-9.]+)", rw)
summary["reword"]["auroc"] = dict(zip(("order_flip", "reword_flip", "order_spread", "reword_spread"), map(float, m.groups())))
pr = (R / "pride.txt").read_text()
summary["pride"] = {}
for name, key in (("full.jsonl", "qwen4b"), ("m-qwen3.5-2b", "qwen2b"), ("m-granite-4.2-3b", "granite3b"), ("m-granite-4.0-h-tiny", "granitetiny"), ("m-smollm3-3b", "smollm3")):
    block = pr.split(name)[1].split("\n\n")[0]
    m = re.search(r"^\s+all\s+\d+\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%", block, re.M)
    summary["pride"][key] = [float(x) / 100 for x in m.groups()]
    summary["pride_n"] = int(re.search(r"^\s+all\s+(\d+)", block, re.M).group(1))
summary["timing"] = json.load(open(R / "orders-timing.json"))
nf = (R / "nofit.jsonl")
pairs = [json.loads(l) for l in open(nf)]
summary["nofit"] = {"n": len(pairs), "mass_full": statistics.median(p["full"]["mass"] for p in pairs),
                    "mass_cut": statistics.median(p["cut"]["mass"] for p in pairs),
                    "cut_below_09": statistics.fmean(p["cut"]["mass"] < 0.9 for p in pairs),
                    "full_below_09": statistics.fmean(p["full"]["mass"] < 0.9 for p in pairs)}
json.dump(summary, open(ROOT / "paper/data/summary.json", "w"), indent=1)
# Figure data
with open(ROOT / "paper/data/bands.tsv", "w") as f:
    f.write("model\tband\tn_stable\tn_flipped\twrong_stable\twrong_flipped\n")
    for key, c in bands_rows:
        f.write(f"{key}\t{c['band']}\t{c['n_stable']}\t{c['n_flipped']}\t{'' if c['wrong_stable'] is None else round(c['wrong_stable'], 4)}\t{'' if c['wrong_flipped'] is None else round(c['wrong_flipped'], 4)}\n")
with open(ROOT / "paper/data/flips.tsv", "w") as f:
    f.write("model\tsource\tflip_rate\n")
    for key, src, v in flips_rows: f.write(f"{key}\t{src}\t{v:.4f}\n")
with open(ROOT / "paper/data/h1.tsv", "w") as f:
    f.write("model\tgain\tlow\thigh\tauroc_conf\tauroc_conf_avg\n")
    for key, h in h1_rows:
        e = summary["models"][key]; f.write(f"{key}\t{h[0]}\t{h[1]}\t{h[2]}\t{e['auroc_conf']:.4f}\t{e['auroc_conf_avg']:.4f}\n")
with open(ROOT / "paper/data/jsd.tsv", "w") as f:
    f.write("model\tbase\taveraged\n")
    for key, j in jsd_rows: f.write(f"{key}\t{j[0]}\t{j[1]}\n")
for key, e in summary["models"].items():
    print(f"{e['label']:20} n={e['n']} wrong {e['wrong']:.3f}->{e['wrong_avg']:.3f} flip {e['flip_rate']:.3f} auroc {e['auroc_conf']:.3f}/{e['auroc_conf_avg']:.3f}/{e['auroc_spread']:.3f} h1 {e['h1']} conf {e['confident_share']:.2f} flips {e['flips']} {e['wrong_flipped']:.3f}/{e['wrong_stable']:.3f} rr {e['adj_rr']} jsd {e['jsd']}")
print("text", summary["text_arm"]); print("reword", summary["reword"]); print("pride", summary["pride"], summary["pride_n"]); print("nofit", summary["nofit"])
for key, e in summary["models"].items():
    print(key, "orders", e["orders_median"], "repeated", e["repeated"], "noise", e["noise_spread"], e["noise_changes"], "auroc_t", round(e["auroc_conf_avg_t"], 3), round(e["auroc_spread_t"], 3), "h1_t", e["h1_t"], "chaos", e["chaos"])
