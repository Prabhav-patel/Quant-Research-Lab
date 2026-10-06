"""
report.py  -  self-contained HTML report builder for the quant project suite.

Produces a single portable .html file (plots embedded as base64) with a
professional, light/dark-aware finance theme. Also exposes setup_plot_style()
so every project's matplotlib charts share one look.

Usage:
    from report import Report, setup_plot_style
    setup_plot_style()
    r = Report("Title", "subtitle", ["tag1","tag2"], date="2026-07-11")
    r.purpose("...why this project exists...")
    r.section("Theory", "<p>...</p>" + r.formula("d1 = ..."))
    r.kpi([("Sharpe","1.02",""), ("Max DD","-34%","")])
    r.figure("outputs/plot.png", "Caption")
    r.learned(["point 1","point 2"])
    r.inference("...final takeaway...")
    r.save("report.html")
"""
import base64, html, os


def setup_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (9, 4.6), "figure.dpi": 120,
        "savefig.dpi": 120, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "axes.edgecolor": "#9aa4b1", "axes.linewidth": 0.9,
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlepad": 12, "axes.labelcolor": "#333", "text.color": "#222",
        "xtick.color": "#555", "ytick.color": "#555",
        "axes.prop_cycle": plt.cycler(color=[
            "#0E7C7B", "#C0392B", "#B7791F", "#3B6EA5", "#7D5BA6", "#1C8A4E"]),
    })


_CSS = """
:root{--bg:#F4F6F8;--surface:#fff;--surface2:#F0F3F5;--ink:#161A20;--muted:#59636F;
--border:#E2E6EB;--accent:#0E7C7B;--accent-soft:#0e7c7b16;--pos:#1C8A4E;--neg:#C0392B;
--warn:#B7791F;--code:#F4F6F8;--shadow:0 1px 2px rgba(20,26,34,.04),0 10px 30px rgba(20,26,34,.06);
--serif:Georgia,"Iowan Old Style","Times New Roman",serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;}
@media(prefers-color-scheme:dark){:root{--bg:#0E1116;--surface:#151A21;--surface2:#1B222B;
--ink:#E6EAEF;--muted:#9AA4B1;--border:#262E39;--accent:#3FB8AF;--accent-soft:#3fb8af1e;
--pos:#3FB27F;--neg:#E5736A;--warn:#E0B252;--code:#11161C;--shadow:0 1px 2px rgba(0,0,0,.3),0 12px 34px rgba(0,0,0,.45);}}
:root[data-theme=dark]{--bg:#0E1116;--surface:#151A21;--surface2:#1B222B;--ink:#E6EAEF;--muted:#9AA4B1;
--border:#262E39;--accent:#3FB8AF;--accent-soft:#3fb8af1e;--pos:#3FB27F;--neg:#E5736A;--warn:#E0B252;
--code:#11161C;--shadow:0 1px 2px rgba(0,0,0,.3),0 12px 34px rgba(0,0,0,.45);}
:root[data-theme=light]{--bg:#F4F6F8;--surface:#fff;--surface2:#F0F3F5;--ink:#161A20;--muted:#59636F;
--border:#E2E6EB;--accent:#0E7C7B;--accent-soft:#0e7c7b16;--pos:#1C8A4E;--neg:#C0392B;--warn:#B7791F;
--code:#F4F6F8;--shadow:0 1px 2px rgba(20,26,34,.04),0 10px 30px rgba(20,26,34,.06);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto;padding:0 22px 120px}
header.hero{padding:52px 0 26px;border-bottom:1px solid var(--border);margin-bottom:8px}
.kick{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-family:var(--serif);font-size:clamp(1.9rem,4.5vw,2.9rem);line-height:1.08;letter-spacing:-.02em;margin:0 0 12px;text-wrap:balance}
.lede{font-size:1.08rem;color:var(--muted);max-width:65ch;margin:0}
.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.tag{font-size:.72rem;font-weight:600;padding:4px 11px;border-radius:100px;background:var(--surface2);color:var(--muted);border:1px solid var(--border);letter-spacing:.02em}
.tag.a{background:var(--accent-soft);color:var(--accent);border-color:transparent}
section.blk{margin-top:34px}
h2{font-family:var(--serif);font-size:1.5rem;letter-spacing:-.01em;margin:0 0 4px;display:flex;align-items:center;gap:12px;text-wrap:balance}
h2 .n{font-family:var(--mono);font-size:.85rem;color:var(--accent);font-weight:700}
h3{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:26px 0 8px;font-weight:700}
p{margin:11px 0}ul,ol{margin:11px 0;padding-left:22px}li{margin:5px 0}
strong{font-weight:650}code{font-family:var(--mono);font-size:.86em;background:var(--code);padding:1px 5px;border-radius:5px;border:1px solid var(--border)}
.formula{font-family:var(--mono);font-size:.9rem;background:var(--code);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:13px 15px;margin:13px 0;overflow-x:auto;white-space:pre;line-height:1.75}
.callout{border-radius:11px;padding:16px 18px;margin:16px 0;border:1px solid var(--border);background:var(--surface);box-shadow:var(--shadow)}
.callout .lab{font-size:.68rem;text-transform:uppercase;letter-spacing:.11em;font-weight:800;margin-bottom:6px;display:block}
.callout.purpose{border-left:4px solid var(--accent)}.callout.purpose .lab{color:var(--accent)}
.callout.learned{border-left:4px solid var(--warn);background:color-mix(in srgb,var(--warn) 6%,var(--surface))}.callout.learned .lab{color:var(--warn)}
.callout.inference{border-left:4px solid var(--pos);background:color-mix(in srgb,var(--pos) 7%,var(--surface))}.callout.inference .lab{color:var(--pos)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px;box-shadow:var(--shadow)}
.kpi .v{font-family:var(--serif);font-size:1.7rem;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .k{font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.kpi .s{font-size:.76rem;color:var(--muted);margin-top:2px}
.kpi.pos .v{color:var(--pos)}.kpi.neg .v{color:var(--neg)}.kpi.acc .v{color:var(--accent)}
figure{margin:18px 0;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px;box-shadow:var(--shadow)}
figure img{width:100%;height:auto;border-radius:6px;display:block}
figcaption{font-size:.82rem;color:var(--muted);margin-top:9px;padding:0 4px;text-align:center}
.tblwrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700}
td{font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}
.foot{margin-top:52px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:.82rem}
a{color:var(--accent)}
"""


def _fmt_cell(x):
    return x if isinstance(x, str) else html.escape(str(x))


class Report:
    def __init__(self, title, subtitle, tags=None, date="", kicker="Quant Finance Project"):
        self.title = title
        self.subtitle = subtitle
        self.tags = tags or []
        self.date = date
        self.kicker = kicker
        self.parts = []

    # ---- content builders (return html strings) ----
    def formula(self, text):
        return f'<div class="formula">{html.escape(text)}</div>'

    def kpi_grid(self, items):
        """items: list of (label, value, sub, tone) ; tone in {'', 'pos','neg','acc'}"""
        cells = []
        for it in items:
            label, value, sub = it[0], it[1], it[2] if len(it) > 2 else ""
            tone = it[3] if len(it) > 3 else ""
            cells.append(
                f'<div class="kpi {tone}"><div class="v">{html.escape(str(value))}</div>'
                f'<div class="k">{html.escape(label)}</div>'
                + (f'<div class="s">{html.escape(sub)}</div>' if sub else "")
                + "</div>")
        return '<div class="kpis">' + "".join(cells) + "</div>"

    def table(self, headers, rows):
        h = "".join(f"<th>{_fmt_cell(x)}</th>" for x in headers)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(f"<td>{_fmt_cell(x)}</td>" for x in row) + "</tr>"
        return f'<div class="tblwrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'

    def figure(self, path, caption=""):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">{cap}</figure>'

    # ---- section adders (append to document) ----
    def section(self, heading, html_body, num=""):
        n = f'<span class="n">{html.escape(num)}</span>' if num else ""
        self.parts.append(f'<section class="blk"><h2>{n}{html.escape(heading)}</h2>{html_body}</section>')
        return self

    def raw(self, html_body):
        self.parts.append(html_body)
        return self

    def purpose(self, text):
        self.parts.append(f'<div class="callout purpose"><span class="lab">Purpose of the project</span>{text}</div>')
        return self

    def learned(self, points):
        lis = "".join(f"<li>{p}</li>" for p in points)
        self.parts.append(f'<div class="callout learned"><span class="lab">What I learned</span><ul>{lis}</ul></div>')
        return self

    def inference(self, text):
        self.parts.append(f'<div class="callout inference"><span class="lab">Final inference</span>{text}</div>')
        return self

    def save(self, path):
        tags = "".join(
            f'<span class="tag{" a" if i == 0 else ""}">{html.escape(t)}</span>'
            for i, t in enumerate(self.tags))
        doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header class="hero">
<div class="kick">{html.escape(self.kicker)}{(" &middot; " + html.escape(self.date)) if self.date else ""}</div>
<h1>{html.escape(self.title)}</h1>
<p class="lede">{html.escape(self.subtitle)}</p>
<div class="tags">{tags}</div>
</header>
{''.join(self.parts)}
<div class="foot">Generated from real market data by this project's <code>main.py</code>. Re-run to refresh with the latest prices. Part of Prabhav Patel's quantitative finance project suite.</div>
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"  report -> {path}")
