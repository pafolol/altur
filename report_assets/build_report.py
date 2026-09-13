"""
Build the technical report: Markdown -> HTML -> PDF.

    python report_assets/build_report.py PROJECT_TECHNICAL_REPORT.md [more.md ...]

Markdown is the source of truth; the PDF is generated from it, so the two can never disagree. Rendering
is Chromium via Playwright (already vendored for this project's page checks), because it is the only
engine here that lays out real tables, page breaks and syntax-highlighted code properly.

Callout boxes use the admonition syntax, with the three evidence classes this report is required to keep
apart:

    !!! fact "Measured"          -> FACT OBSERVED IN PROJECT
    !!! interp "Reading"         -> INTERPRETATION
    !!! bg "Background"          -> GENERAL ML BACKGROUND
    !!! gap "Unknown"            -> not recoverable from the artifacts
    !!! warn "Careful"           -> caveat / honesty note
"""
import asyncio
import os
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CHROME = os.path.expanduser("~/AppData/Local/ms-playwright/chromium-1243/chrome-win64/chrome.exe")

CSS = r"""
@page { size: A4; margin: 17mm 16mm 19mm 16mm; }

:root{
  --ink:#15191f; --dim:#5a6472; --rule:#d4dae2; --faint:#eef1f5;
  --accent:#1f4e79; --accent2:#7a3e12; --ok:#1d6b44; --bad:#a32b2b;
}

*{ box-sizing:border-box; }
html{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body{
  margin:0; color:var(--ink); background:#fff;
  font-family:"Charter","Bitstream Charter","Georgia","Times New Roman",serif;
  font-size:10.2pt; line-height:1.52;
  font-variant-numeric: tabular-nums lining-nums;
}
h1,h2,h3,h4,h5,.sans{ font-family:"Segoe UI","Inter","Helvetica Neue",Arial,sans-serif; }
code,pre,kbd,.mono{ font-family:"Cascadia Mono","Consolas","DejaVu Sans Mono",monospace; }

/* ---------------------------------------------------------------- headings */
h1{
  font-size:19pt; font-weight:650; letter-spacing:-.01em; line-height:1.15;
  margin:0 0 14pt; padding:0 0 7pt; border-bottom:2.2pt solid var(--accent);
  color:var(--accent); page-break-before:always; page-break-after:avoid;
}
h1:first-of-type{ page-break-before:avoid; }
h1 .num{ display:block; font-size:8.5pt; letter-spacing:.18em; font-weight:600;
         color:var(--dim); text-transform:uppercase; margin-bottom:5pt; }
h2{ font-size:13.2pt; font-weight:640; margin:20pt 0 7pt; color:#12314d;
    page-break-after:avoid; border-bottom:.6pt solid var(--rule); padding-bottom:3pt; }
h3{ font-size:11pt; font-weight:640; margin:14pt 0 4pt; color:#1a2530; page-break-after:avoid; }
h4{ font-size:9.6pt; font-weight:660; margin:11pt 0 3pt; color:var(--dim);
    text-transform:uppercase; letter-spacing:.07em; page-break-after:avoid; }
p{ margin:0 0 7pt; orphans:3; widows:3; }
hr{ border:0; border-top:.6pt solid var(--rule); margin:16pt 0; }

/* ---------------------------------------------------------------- lists */
ul,ol{ margin:0 0 8pt; padding-left:16pt; }
li{ margin:0 0 2.5pt; orphans:2; widows:2; }
li>ul,li>ol{ margin-top:2.5pt; }
dl{ margin:0 0 8pt; } dt{ font-weight:650; margin-top:5pt; } dd{ margin:0 0 0 14pt; }

/* ---------------------------------------------------------------- code */
code{ font-size:8.6pt; background:var(--faint); padding:.6pt 2.6pt;
      border-radius:2pt; border:.4pt solid #e0e5ec; }
pre{
  margin:7pt 0 9pt; padding:7pt 9pt; background:#fafbfc; border:.6pt solid var(--rule);
  border-left:2.4pt solid var(--accent); border-radius:2pt; overflow:hidden;
  font-size:8.1pt; line-height:1.42; white-space:pre-wrap; word-wrap:break-word;
  page-break-inside:avoid;
}
pre code{ background:none; border:0; padding:0; font-size:inherit; }
pre.wide{ font-size:7.2pt; }
.codehilite{ background:none; }
.codehilite .k,.codehilite .kn,.codehilite .kd{ color:#8a2f8a; font-weight:600; }
.codehilite .s,.codehilite .s1,.codehilite .s2,.codehilite .sd{ color:#1d6b44; }
.codehilite .c,.codehilite .c1,.codehilite .cm{ color:#7b8694; font-style:italic; }
.codehilite .nf{ color:var(--accent); font-weight:600; }
.codehilite .mi,.codehilite .mf{ color:var(--accent2); }
.codehilite .nb{ color:#0f5b8a; }

/* ASCII diagrams: never wrap, never shrink below legibility */
pre.diagram{ white-space:pre; font-size:7.6pt; line-height:1.32; background:#fff;
             border-left-color:#9aa6b4; text-align:left; }

/* ---------------------------------------------------------------- tables */
table{
  border-collapse:collapse; width:100%; margin:8pt 0 10pt; font-size:8.5pt;
  font-family:"Segoe UI",Arial,sans-serif;
}
table.wide{ font-size:7.5pt; }
/* Tables break ACROSS pages, but never mid-row, and the header repeats on every continuation.
   Forbidding the whole table from breaking looks tidy until a 40-row glossary meets a page
   boundary and pushes itself bodily onto the next page, stranding its heading above a blank. */
tr{ page-break-inside:avoid; }
thead{ display:table-header-group; }
th{
  text-align:left; font-weight:650; font-size:7.6pt; text-transform:uppercase;
  letter-spacing:.05em; color:var(--dim); padding:4pt 5pt;
  border-bottom:1.1pt solid var(--ink); vertical-align:bottom;
}
td{ padding:3.4pt 5pt; border-bottom:.4pt solid var(--rule); vertical-align:top; }
tbody tr:last-child td{ border-bottom:.8pt solid var(--ink); }
td code{ font-size:7.8pt; }
td:not(:first-child){ font-variant-numeric:tabular-nums; }

/* ---------------------------------------------------------------- callouts */
.admonition{
  margin:9pt 0; padding:7pt 10pt; border-radius:2pt; page-break-inside:avoid;
  border:.5pt solid var(--rule); border-left:3pt solid var(--dim); background:#fbfcfd;
  font-size:9.5pt;
}
.admonition>:last-child{ margin-bottom:0; }
.admonition-title{
  font-family:"Segoe UI",Arial,sans-serif; font-weight:680; font-size:7.6pt;
  text-transform:uppercase; letter-spacing:.09em; margin:0 0 4pt; color:var(--dim);
}
.admonition.fact{ border-left-color:var(--ok); background:#f4faf6; }
.admonition.fact>.admonition-title{ color:var(--ok); }
.admonition.fact>.admonition-title::before{ content:"◆ Fact observed in project — "; }
.admonition.interp{ border-left-color:var(--accent); background:#f4f8fc; }
.admonition.interp>.admonition-title{ color:var(--accent); }
.admonition.interp>.admonition-title::before{ content:"▷ Interpretation — "; }
.admonition.bg{ border-left-color:#6b5ea8; background:#f8f7fc; }
.admonition.bg>.admonition-title{ color:#6b5ea8; }
.admonition.bg>.admonition-title::before{ content:"▣ General ML background — "; }
.admonition.gap{ border-left-color:#8a8f98; background:#f6f7f8; }
.admonition.gap>.admonition-title{ color:#6a7078; }
.admonition.gap>.admonition-title::before{ content:"○ Evidence gap — "; }
.admonition.warn{ border-left-color:var(--bad); background:#fdf5f5; }
.admonition.warn>.admonition-title{ color:var(--bad); }
.admonition.warn>.admonition-title::before{ content:"▲ Caveat — "; }

blockquote{
  margin:8pt 0; padding:5pt 0 5pt 11pt; border-left:2.4pt solid var(--rule);
  color:var(--dim); font-style:italic;
}
blockquote>:last-child{ margin-bottom:0; }

/* ---------------------------------------------------------------- figures */
img{ max-width:100%; height:auto; display:block; margin:0 auto; }
p>img{ border:.5pt solid var(--rule); border-radius:2pt; padding:3pt; background:#fff; }
.figure{ margin:10pt 0 12pt; page-break-inside:avoid; text-align:center; }
.figure img{ border:.5pt solid var(--rule); border-radius:2pt; padding:3pt; }
.caption,.figure em{
  display:block; margin-top:4pt; font-family:"Segoe UI",Arial,sans-serif;
  font-size:7.9pt; color:var(--dim); font-style:normal; text-align:left;
}
.caption b,.figure em b{ color:var(--ink); font-weight:650; }

/* ---------------------------------------------------------------- title page */
.titlepage{ page-break-after:always; padding-top:34mm; }
.titlepage .kicker{
  font-family:"Segoe UI",Arial,sans-serif; font-size:9pt; font-weight:640;
  letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin-bottom:10pt;
}
.titlepage h1{
  border:0; padding:0; margin:0 0 8pt; font-size:31pt; line-height:1.06;
  color:var(--ink); page-break-before:avoid; letter-spacing:-.02em; font-weight:680;
}
.titlepage .sub{ font-size:13pt; color:var(--dim); margin:0 0 26pt; line-height:1.35; }
.titlepage .rule{ border-top:2.4pt solid var(--accent); width:62mm; margin:0 0 20pt; }
.titlepage dl{ font-size:9.4pt; }
.titlepage dt{ font-family:"Segoe UI",Arial,sans-serif; font-size:7.6pt;
               text-transform:uppercase; letter-spacing:.09em; color:var(--dim); margin-top:9pt; }
.titlepage dd{ margin:1pt 0 0; }

/* table of contents */
.toc{ page-break-after:always; }
.toc ul{ list-style:none; padding-left:0; margin:0; }
.toc li{ margin:0; }
.toc>.toctitle{ display:none; }
.toc ul ul{ padding-left:13pt; }
.toc a{ color:var(--ink); text-decoration:none; }
.toc>ul>li>a{ font-family:"Segoe UI",Arial,sans-serif; font-weight:640; font-size:9.6pt;
              display:block; margin-top:7pt; border-bottom:.4pt solid var(--faint); padding-bottom:2pt; }
.toc>ul>li>ul>li>a{ font-size:8.8pt; color:var(--dim); }
.toc>ul>li>ul>li>ul{ display:none; }

.nobreak{ page-break-inside:avoid; }
.newpage{ page-break-before:always; }
a{ color:var(--accent); text-decoration:none; }
strong{ font-weight:680; }
sup{ font-size:.7em; }
"""

def footer(label: str) -> str:
    return ('<div style="width:100%;font-size:7pt;font-family:Segoe UI,Arial,sans-serif;color:#5a6472;'
            'padding:0 16mm;display:flex;justify-content:space-between;">'
            f'<span>Altur HackMTY 2026 — synthetic-caller detection — {label}</span>'
            '<span class="pageNumber"></span></div>')


FOOTERS = {
    "PROJECT_TECHNICAL_REPORT": "technical report",
    "PROJECT_CHEAT_SHEET": "team cheat sheet",
    "JUDGE_QA": "judge Q&amp;A",
    "PROJECT_FILE_MAP": "project file map",
}
HEADER = '<div style="font-size:1pt;color:#fff;">.</div>'

EXTENSIONS = [
    "extra", "tables", "fenced_code", "codehilite", "admonition",
    "attr_list", "def_list", "md_in_html", "sane_lists", "toc", "smarty",
]
EXT_CONFIG = {
    "codehilite": {"guess_lang": False, "noclasses": False},
    "toc": {"toc_depth": "1-2", "permalink": False},
    "smarty": {"smart_dashes": False},
}


def to_html(md_text: str, title: str) -> str:
    md = markdown.Markdown(extensions=EXTENSIONS, extension_configs=EXT_CONFIG)
    body = md.convert(md_text)
    # `[TOC]` renders as div.toc; the title page is authored as a raw <div class="titlepage">.
    # Tag ASCII-art blocks so they keep their alignment: they are fenced as ```text.
    body = body.replace('<pre><code class="language-text">', '<pre class="diagram"><code>')
    body = re.sub(r'<div class="codehilite"><pre><span></span><code class="language-text">',
                  '<pre class="diagram"><code>', body)
    # The Markdown lives at the project root and points at report_assets/figures/...; the HTML is
    # rendered INSIDE report_assets/, so the same path has to lose that one leading segment.
    body = body.replace('src="report_assets/', 'src="')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


async def render(html_path: Path, pdf_path: Path, foot: str):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME)
        page = await b.new_page()
        await page.goto(html_path.as_uri(), wait_until="networkidle")
        await page.wait_for_timeout(1200)
        await page.pdf(path=str(pdf_path), format="A4", print_background=True,
                       display_header_footer=True, header_template=HEADER, footer_template=foot,
                       margin={"top": "14mm", "bottom": "16mm", "left": "0mm", "right": "0mm"})
        await b.close()


def build(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), md_path.stem)
    html_path = ROOT / "report_assets" / f"{md_path.stem}.html"
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(to_html(text, title), encoding="utf-8")
    asyncio.run(render(html_path, pdf_path,
                       footer(FOOTERS.get(md_path.stem, 'technical report'))))
    print(f"{md_path.name}: {len(text):,} chars -> {pdf_path.name} "
          f"({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(Path(arg) if Path(arg).is_absolute() else ROOT / arg)
