#!/usr/bin/env python3
"""Render main.tex as a self-contained HTML page (order-shouldnt-matter.html) with the SVG figures inlined.

Covers the LaTeX subset main.tex uses: sections and paragraphs, abstract, itemize, table/table* with a
booktabs tabular, figure/figure* with includegraphics, labels and refs, natbib citations resolved from
references.bib, and a little inline math. build.sh prints the page to PDF with Chrome. Standard library only.
"""
import html
import re
import unicodedata
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; @bottom-center { content: counter(page); font: 9pt serif; } }
:root { --ink: #1a1a1a; --muted: #555; --rule: #999; }
html { font-size: 9.6pt; }
body { margin: 0; color: var(--ink); background: #fff; font-family: "Libertinus Serif", "Linux Libertine O",
  "STIX Two Text", "TeX Gyre Pagella", "Times New Roman", Times, serif; line-height: 1.32;
  text-rendering: optimizeLegibility; }
@media screen { body { max-width: 190mm; margin: 0 auto; padding: 12mm 10mm; } }
header { column-span: all; text-align: center; margin: 0 0 6mm; }
header h1 { font-size: 15.5pt; font-weight: 700; line-height: 1.22; margin: 0 8mm 3mm; }
header .authors { font-size: 10.5pt; margin: 0 0 1mm; }
header .date { font-size: 9pt; color: var(--muted); }
main { column-count: 2; column-gap: 6.5mm; column-fill: auto; }
@media screen and (max-width: 800px) { main { column-count: 1; } figure.wide, .table.wide { column-span: none; } .table { overflow-x: auto; } }
h2 { font-size: 11pt; margin: 4mm 0 1.5mm; break-after: avoid; }
h2.appendix { }
p { margin: 0 0 0 0; text-align: justify; hyphens: auto; -webkit-hyphens: auto; }
p + p { text-indent: 4mm; }
.abstract { margin: 0 0 3mm; padding: 0 0 2mm; }
.abstract h2 { text-align: center; font-size: 10pt; margin: 0 0 1.5mm; }
.abstract p { font-size: 9.1pt; text-indent: 0; }
.para { display: inline; font-weight: 700; }
p.run { text-indent: 0; margin-top: 1.6mm; }
ul { margin: 1.5mm 0 1.5mm 4.5mm; padding: 0; }
li { margin: 0 0 1mm; text-align: justify; }
figure { margin: 1mm 0 3mm; break-inside: avoid; float: left; width: 100%; }
figure.wide, .table.wide { column-span: all; float: none; }
figure svg { width: 100%; height: auto; display: block; }
figure.wide svg { width: 100%; max-width: 176mm; margin: 0 auto; }
figcaption, .table .caption { font-size: 8.6pt; line-height: 1.28; margin: 1.5mm 0 0; text-align: left; }
.table { margin: 1mm 0 3mm; break-inside: avoid; float: left; width: 100%; }
.table table { border-collapse: collapse; margin: 0 auto; font-size: 8.4pt; line-height: 1.25; }
.table.wide table { font-size: 8.2pt; }
.table td, .table th { padding: 0.35mm 1.6mm; vertical-align: top; white-space: nowrap; }
.table th { font-weight: 400; text-align: left; }
.table .l { text-align: left; } .table .r { text-align: right; } .table .c { text-align: center; }
.table tr.top td, .table tr.top th { border-top: 0.9pt solid var(--ink); }
.table tr.mid td, .table tr.mid th { border-top: 0.5pt solid var(--ink); }
.table tr.bot td, .table tr.bot th { border-bottom: 0.9pt solid var(--ink); }
.table .caption b, figcaption b { font-weight: 700; }
code, tt { font-family: "Latin Modern Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 0.92em; }
a { color: inherit; text-decoration: underline; text-decoration-color: #8a8a8a; } a:focus { outline: 2px solid #2a78d6; }
.bib { font-size: 8.6pt; line-height: 1.28; }
.bib p { text-indent: -3.5mm; padding-left: 3.5mm; margin: 0 0 1.2mm; text-align: left; }
.math { font-style: italic; } .math .up { font-style: normal; }
sup, sub { line-height: 0; font-size: 0.75em; }
"""


def parse_bib(path):
    entries = {}
    for m in re.finditer(r"@(\w+)\{(\w+),(.*?)\n\}", path.read_text(), re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(3)):
            fields[f.group(1)] = f.group(2).strip()
        entries[m.group(2)] = fields
    return entries


def bib_names(entry):
    """Last names of the authors, for citations; a work without authors is cited by its title's first word."""
    if "author" not in entry:
        return [entry.get("key") or entry["title"].split(":")[0].strip()]
    names = []
    for a in entry["author"].split(" and "):
        a = a.strip()
        if a == "others":
            names.append("others")
        elif a.startswith("{"):
            names.append(a.strip("{}"))
        elif "," in a:
            names.append(a.split(",")[0].strip())
        else:
            names.append(a.split()[-1])
    return [tex_inline(n) for n in names]


def short_cite(entry):
    names = bib_names(entry)
    if len(names) == 1:
        return names[0]
    if len(names) == 2 and names[1] != "others":
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def bib_entry_html(entry):
    if "author" in entry:
        authors = []
        for a in entry["author"].split(" and "):
            a = a.strip()
            if a == "others":
                authors[-1] += " et al."
                continue
            elif a.startswith("{"):
                authors.append(a.strip("{}"))
            elif "," in a:
                last, first = [s.strip() for s in a.split(",", 1)]
                authors.append(f"{first} {last}")
            else:
                authors.append(a)
        who = ", ".join(authors[:-1]) + (" and " if len(authors) > 1 else "") + authors[-1]
    else:
        who = ""
    parts = [tex_inline(who) + ("" if who.endswith(".") else ".") if who else ""]
    parts.append(f"{entry.get('year', '')}.")
    parts.append(tex_inline(entry["title"]) + ".")
    venue = entry.get("booktitle") or entry.get("journal") or ""
    if venue:
        v = tex_inline(venue)
        if "volume" in entry:
            v += " " + entry["volume"]
        parts.append(f"<i>{v}</i>.")
    if "howpublished" in entry:
        parts.append(tex_inline(entry["howpublished"]) + ".")
    if "note" in entry:
        parts.append(tex_inline(entry["note"]) + ".")
    return " ".join(p for p in parts if p)


def math_html(s):
    s = s.replace("\\times", "×").replace("\\ge", "≥").replace("\\le", "≤").replace("\\kappa", "κ")
    s = s.replace("\\pm", "±").replace("-", "−").replace("\\,", " ").replace("\\%", "%")
    s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(\w)", r"<sup>\1</sup>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    # digits and operators upright, letters italic
    s = re.sub(r"([0-9.,%≥≤×±−+= ]+)", r'<span class="up">\1</span>', s)
    return f'<span class="math">{s}</span>'


BIB = {}
LABELS = {}


def tex_inline(s):
    """Inline LaTeX to HTML. Escapes text, then applies the commands main.tex uses."""
    out = []
    i = 0
    # protect math first
    def repl_math(m):
        return "\x00" + math_html(m.group(1)) + "\x01"
    s = re.sub(r"\$([^$]*)\$", repl_math, s)
    # TeX accents in names: {\'e}, {\"o}, {\~n}, {\v{c}}, {\c{c}}; combine, or keep the bare letter
    accents = {"'": "\u0301", '"': "\u0308", "~": "\u0303", "v": "\u030c", "c": "\u0327", "`": "\u0300", "^": "\u0302"}
    s = re.sub(r"\{\\([\'\"~v`^c])\{?(\w)\}?\}", lambda m: unicodedata.normalize("NFC", m.group(2) + accents[m.group(1)]), s)
    codes = []
    def keep(m):
        codes.append(html.escape(m.group(1), quote=False))
        return f"\x02{len(codes) - 1}\x03"
    s = re.sub(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}", keep, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\x00(.*?)\x01", lambda m: html.unescape(m.group(1)) if False else m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&"), s, flags=re.S)
    s = s.replace("\\%", "%").replace("\\&amp;", "&amp;").replace("\\_", "_").replace("\\#", "#")
    s = s.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–").replace("~", "&nbsp;")
    s = s.replace("\\quad", "&emsp;").replace("\\,", "&thinsp;").replace("\\\\", "<br>")
    s = re.sub(r"\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\emph\{((?:[^{}]|\{[^{}]*\})*)\}", r"<i>\1</i>", s)
    s = re.sub(r"\x02(\d+)\x03", lambda m: f"<code>{codes[int(m.group(1))]}</code>", s)
    s = re.sub(r"\\url\{([^}]*)\}", r'<a href="\1">\1</a>', s)
    s = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r'<a href="\1">\2</a>', s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: LABELS.get(m.group(1), "??"), s)
    s = re.sub(r"\\citep\{([^}]*)\}", lambda m: "(" + "; ".join(f"{short_cite(BIB[k.strip()])}, {BIB[k.strip()].get('year','')}" for k in m.group(1).split(",")) + ")", s)
    s = re.sub(r"\\citet\{([^}]*)\}", lambda m: ", ".join(f"{short_cite(BIB[k.strip()])} ({BIB[k.strip()].get('year','')})" for k in m.group(1).split(",")), s)
    s = re.sub(r"\{\\(\w+)\}", r"\\\1", s)  # {\S} style: leave the command
    s = s.replace("\\@", "")
    s = re.sub(r"(?<!\\)\{|(?<!\\)\}", "", s)  # bare grouping braces
    return s


def find_env(text, env, start):
    """Return (begin_index, body, end_index) of the first \\begin{env}...\\end{env} at or after start."""
    b = text.find(f"\\begin{{{env}}}", start)
    if b < 0:
        return None
    body_start = b + len(f"\\begin{{{env}}}")
    e = text.find(f"\\end{{{env}}}", body_start)
    return b, text[body_start:e], e + len(f"\\end{{{env}}}")


def tabular_html(body, spec):
    cols = [c for c in re.sub(r"@\{[^}]*\}", "", spec) if c in "lrc"]
    rows = []
    pending = []
    for line in body.strip().split("\\\\"):
        line = line.strip()
        rules = []
        for r in ("toprule", "midrule", "bottomrule"):
            if "\\" + r in line:
                rules.append(r)
                line = line.replace("\\" + r, "").strip()
        if not line:
            if "bottomrule" in rules and rows:
                rows[-1][0].append("bot")
            else:
                pending += rules
            continue
        cells = [c.strip() for c in line.split("&")]
        cls = list(pending) + rules
        pending = []
        rows.append(([{"toprule": "top", "midrule": "mid", "bottomrule": "bot"}[c] for c in cls], cells))
    h = ["<table>"]
    for n, (cls, cells) in enumerate(rows):
        h.append(f'<tr class="{" ".join(cls)}">')
        for j, c in enumerate(cells):
            mc = re.match(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}$", c, re.S)
            if mc:
                h.append(f'<td class="l" colspan="{mc.group(1)}">{tex_inline(mc.group(2))}</td>')
            else:
                tag = 'th scope="col"' if n == 0 else ('th scope="row"' if j == 0 else 'td')
                h.append(f'<{tag} class="{cols[j] if j < len(cols) else "l"}">{tex_inline(c)}</{tag.split()[0]}>')
        h.append("</tr>")
    h.append("</table>")
    return "".join(h)


def float_html(kind, body, wide, num):
    cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.S)
    caption = tex_inline(cap.group(1)) if cap else ""
    label = f"{'Table' if kind == 'table' else 'Figure'} {num}"
    if kind == "table":
        body = re.sub(r"\\input\{([^}]*)\}", lambda m: (HERE / (m.group(1) + ".tex")).read_text(), body)
        tab = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", body, re.S)
        inner = tabular_html(tab.group(2), tab.group(1))
        return f'<div class="table{" wide" if wide else ""}">{inner}<p class="caption"><b>{label}:</b> {caption}</p></div>'
    g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body)
    svg = (HERE / g.group(1)).with_suffix(".svg").read_text()
    svg = svg[svg.find("<svg"):]
    # The caption is the figure's text alternative: the SVG is an image described by its figcaption.
    cid = f"fig{num}-caption"
    svg = svg.replace("<svg ", f'<svg role="img" aria-labelledby="{cid}" ', 1)
    return (f'<figure class="{"wide" if wide else ""}">{svg}'
            f'<figcaption id="{cid}"><b>{label}:</b> {caption}</figcaption></figure>')


def collect_labels(body):
    """First pass: number sections, floats and their labels in order of appearance."""
    sec = 0
    appendix = False
    fig = tab = 0
    pos = 0
    events = []
    for m in re.finditer(r"\\section\{|\\appendix|\\begin\{(figure\*?|table\*?)\}", body):
        events.append((m.start(), m.group(0), m.group(1)))
    for start, tok, env in events:
        if tok == "\\appendix":
            appendix = True
            sec = 0
            continue
        end = body.find("\n", start)
        if tok == "\\section{":
            sec += 1
            name = chr(64 + sec) if appendix else str(sec)
            chunk = body[start:body.find("\n", start) + 1]
            lm = re.search(r"\\label\{([^}]*)\}", body[start:start + 400])
            if lm:
                LABELS[lm.group(1)] = name
            continue
        e = find_env(body, env, start)
        if env.startswith("figure"):
            fig += 1
            n = fig
        else:
            tab += 1
            n = tab
        lm = re.search(r"\\label\{([^}]*)\}", e[1])
        if lm:
            LABELS[lm.group(1)] = str(n)


def convert(tex):
    global BIB
    BIB = parse_bib(HERE / "references.bib")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    title = re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1)
    author = re.search(r"\\author\{(.*?)\}\n", tex, re.S).group(1)
    date = re.search(r"\\date\{(.*?)\}\n", tex, re.S).group(1)
    body = tex[tex.find("\\begin{document}") + len("\\begin{document}"):tex.find("\\end{document}")]
    collect_labels(body)
    out = [f"<header><h1>{tex_inline(title)}</h1><p class=\"authors\">{tex_inline(author)}</p>"
           f"<p class=\"date\">{tex_inline(date)}</p></header><main>"]
    sec = 0
    appendix = False
    fig = tab = 0
    cited = set()
    pos = 0
    pattern = re.compile(r"\\maketitle|\\clearpage|\\newpage|\\section\*\{[^}]*\}|\\appendix|\\begin\{(abstract|itemize|figure\*?|table\*?)\}|\\section\{|\\paragraph\{|\\bibliographystyle\{[^}]*\}|\\bibliography\{[^}]*\}")

    def paragraphs(text):
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if para:
                out.append(f"<p>{tex_inline(para)}</p>")
                cited.update(k.strip() for m in re.finditer(r"\\cite[pt]\{([^}]*)\}", para) for k in m.group(1).split(","))

    while True:
        m = pattern.search(body, pos)
        if not m:
            paragraphs(body[pos:])
            break
        paragraphs(body[pos:m.start()])
        tok = m.group(0)
        env = m.group(1)
        if tok == "\\maketitle":
            pos = m.end()
        elif tok == "\\appendix":
            appendix = True
            sec = 0
            pos = m.end()
        elif tok in ("\\clearpage", "\\newpage"):
            pos = m.end()
        elif tok.startswith("\\section*"):
            out.append(f"<h2>{tex_inline(tok[len(chr(92)+'section*{'):-1])}</h2>")
            pos = m.end()
        elif tok.startswith("\\bibliography"):
            pos = m.end()
            if tok.startswith("\\bibliography{"):
                out.append('<h2>References</h2><div class="bib">')
                for key in sorted(cited, key=lambda k: (re.sub(r"<[^>]+>", "", bib_names(BIB[k])[0]).lower(), BIB[k].get("year", ""))):
                    out.append(f"<p>{bib_entry_html(BIB[key])}</p>")
                out.append("</div>")
        elif tok == "\\section{":
            e = body.find("}", m.end())
            sec += 1
            name = chr(64 + sec) if appendix else str(sec)
            title_ = tex_inline(body[m.end():e])
            out.append(f'<h2{" class=appendix" if appendix else ""}>{name}&nbsp;&nbsp;{title_}</h2>')
            pos = e + 1
            lm = re.match(r"\s*\\label\{[^}]*\}", body[pos:])
            if lm:
                pos += lm.end()
        elif tok == "\\paragraph{":
            e = body.find("}", m.end())
            head = tex_inline(body[m.end():e])
            rest_end = body.find("\n\n", e)
            rest = " ".join(body[e + 1:rest_end].split())
            rest = re.sub(r"^\\label\{[^}]*\}\s*", "", rest)
            cited.update(k.strip() for mm in re.finditer(r"\\cite[pt]\{([^}]*)\}", rest) for k in mm.group(1).split(","))
            out.append(f'<p class="run"><span class="para">{head}</span> {tex_inline(rest)}</p>')
            pos = rest_end
        elif env == "abstract":
            b, inner, e = find_env(body, "abstract", m.start())
            out.append(f'<section class="abstract"><h2>Abstract</h2><p>{tex_inline(" ".join(inner.split()))}</p></section>')
            cited.update(k.strip() for mm in re.finditer(r"\\cite[pt]\{([^}]*)\}", inner) for k in mm.group(1).split(","))
            pos = e
        elif env == "itemize":
            b, inner, e = find_env(body, "itemize", m.start())
            items = [i.strip() for i in inner.split("\\item") if i.strip()]
            out.append("<ul>" + "".join(f"<li>{tex_inline(' '.join(i.split()))}</li>" for i in items) + "</ul>")
            cited.update(k.strip() for mm in re.finditer(r"\\cite[pt]\{([^}]*)\}", inner) for k in mm.group(1).split(","))
            pos = e
        else:
            b, inner, e = find_env(body, env, m.start())
            kind = "figure" if env.startswith("figure") else "table"
            if kind == "figure":
                fig += 1
                n = fig
            else:
                tab += 1
                n = tab
            out.append(float_html(kind, inner, env.endswith("*"), n))
            cited.update(k.strip() for mm in re.finditer(r"\\cite[pt]\{([^}]*)\}", inner) for k in mm.group(1).split(","))
            pos = e
    out.append("</main>")
    page = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(re.sub(r'<[^>]+>', '', tex_inline(title)))}</title>"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><style>{CSS}</style></head><body>"
            + "".join(out) + "</body></html>")
    return page


if __name__ == "__main__":
    src = HERE / "main.tex"
    dst = HERE / (sys.argv[1] if len(sys.argv) > 1 else "order-shouldnt-matter.html")
    dst.write_text(convert(src.read_text()))
    print(dst)
