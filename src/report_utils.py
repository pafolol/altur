"""
A tiny document model with two renderers: Markdown (editable source) and PDF (ReportLab).
build_report.py writes the report once through this API and gets both files with identical content.

Markdown inline markup accepted inside text: **bold**, *italic*, `code`. The PDF renderer converts it to
ReportLab's mini-HTML. Images are referenced relative to the reports/ folder in the Markdown file.
"""
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                   ("DejaVu-Italic", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                   ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_FONTS / file)))
addMapping("DejaVu", 0, 0, "DejaVu")
addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Italic")
addMapping("DejaVu", 1, 1, "DejaVu-BoldItalic")

BLUE = colors.HexColor("#173a5e")
ACCENT = colors.HexColor("#d9534f")
STYLES = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=23, leading=29, textColor=BLUE, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=11.5, leading=15.5, textColor=colors.HexColor("#555555"), spaceAfter=16),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=17, leading=22, textColor=BLUE, spaceBefore=12, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=10.3, leading=14, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.2, leading=13.5, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.2, leading=13.2, leftIndent=14, firstLineIndent=-9, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Italic", fontSize=7.8, leading=10.5, textColor=colors.HexColor("#666666"),
                              alignment=1, spaceAfter=9),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=7.4, leading=9.8),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.6, leading=9.8),
    "cell_bold": ParagraphStyle("cell_bold", fontName="DejaVu-Bold", fontSize=7.3, leading=9.4, textColor=colors.white),
    "toc0": ParagraphStyle("toc0", fontName="DejaVu-Bold", fontSize=9.5, leading=13, leftIndent=10),
    "toc1": ParagraphStyle("toc1", fontName="DejaVu", fontSize=8.5, leading=11, leftIndent=26),
    "kpi_value": ParagraphStyle("kpi_value", fontName="DejaVu-Bold", fontSize=15, leading=18, textColor=BLUE, alignment=1),
    "kpi_label": ParagraphStyle("kpi_label", fontName="DejaVu", fontSize=7.4, leading=9.5, textColor=colors.HexColor("#555555"), alignment=1),
}
NOTE_COLORS = {"info": ("#eaf1fb", "#1f77b4"), "warn": ("#fff4e0", "#e08a00"), "key": ("#eaf7ea", "#2ca02c"), "danger": ("#fdecec", "#d62728")}


def md_to_rl(text):
    """**bold** / *italic* / `code` -> ReportLab mini-HTML (escaping & < > first)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font face='DejaVuMono' size='7.6'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", text)
    return text


class _Doc(BaseDocTemplate):
    def __init__(self, path, title):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                         bottomMargin=1.7 * cm, title=title, author="Altur HackMTY 2026 - acoustic team")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._footer)])
        self._doc_title, self._n = title, 0

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 1.0 * cm, self._doc_title)
        canvas.drawRightString(A4[0] - doc.rightMargin, 1.0 * cm, f"page {doc.page}")
        canvas.restoreState()

    def beforeDocument(self):
        self._n = 0

    def afterFlowable(self, flowable):
        level = getattr(flowable, "_toc_level", None)
        if level is None:
            return
        self._n += 1
        key, text = f"h{self._n}", flowable.getPlainText()
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


class Report:
    """Collects elements; render() writes <name>.md and <name>.pdf."""

    def __init__(self, title, subtitle, md_dir):
        self.title, self.subtitle, self.md_dir = title, subtitle, Path(md_dir)
        self.elements = []
        self.width_cm = (A4[0] - 4 * cm) / cm

    # ------------------------------------------------------------------ authoring API
    def h1(self, text):
        self.elements.append(("h1", text))

    def h2(self, text):
        self.elements.append(("h2", text))

    def h3(self, text):
        self.elements.append(("h3", text))

    def p(self, *texts):
        for t in texts:
            self.elements.append(("p", t))

    def bullets(self, items):
        self.elements.append(("bullets", list(items)))

    def numbered(self, items):
        self.elements.append(("numbered", list(items)))

    def code(self, text):
        self.elements.append(("code", text.strip("\n")))

    def note(self, text, kind="info", title=None):
        self.elements.append(("note", (kind, title, text if isinstance(text, list) else [text])))

    def table(self, rows, col_widths_cm=None, highlight_rows=(), font_size=7.6, caption=None):
        self.elements.append(("table", (rows, col_widths_cm, tuple(highlight_rows), font_size, caption)))

    def image(self, path, caption=None, width_cm=16, max_height_cm=20):
        self.elements.append(("image", (Path(path), caption, width_cm, max_height_cm)))

    def images(self, paths, caption=None, max_height_cm=8.5):
        self.elements.append(("images", ([Path(p) for p in paths], caption, max_height_cm)))

    def kpis(self, items):
        """items: list of (value, label) shown as highlighted number tiles."""
        self.elements.append(("kpis", list(items)))

    def page_break(self):
        self.elements.append(("page_break", None))

    # ------------------------------------------------------------------ Markdown renderer
    def to_markdown(self, path):
        out = [f"# {self.title}", "", f"*{self.subtitle}*", ""]
        for kind, payload in self.elements:
            if kind == "h1":
                out += ["", f"## {payload}", ""]
            elif kind == "h2":
                out += ["", f"### {payload}", ""]
            elif kind == "h3":
                out += ["", f"#### {payload}", ""]
            elif kind == "p":
                out += [payload, ""]
            elif kind == "bullets":
                out += [f"- {it}" for it in payload] + [""]
            elif kind == "numbered":
                out += [f"{i}. {it}" for i, it in enumerate(payload, 1)] + [""]
            elif kind == "code":
                out += ["```", payload, "```", ""]
            elif kind == "note":
                k, title, texts = payload
                tag = {"info": "NOTE", "warn": "WARNING", "key": "KEY RESULT", "danger": "CAUTION"}[k]
                head = f"> **{tag}{(' - ' + title) if title else ''}**"
                out += [head, ">"] + [f"> {t}" for t in texts] + [""]
            elif kind == "table":
                rows, _, _, _, caption = payload
                strip = lambda c: re.sub(r"<[^>]+>", "", str(c)).replace("|", "/")
                out += ["| " + " | ".join(strip(c) for c in rows[0]) + " |", "|" + "---|" * len(rows[0])]
                out += ["| " + " | ".join(strip(c) for c in r) + " |" for r in rows[1:]]
                out += [f"\n*{caption}*" if caption else "", ""]
            elif kind == "image":
                p, caption, _, _ = payload
                out += [f"![{caption or p.stem}]({self._rel(p)})", f"*{caption}*" if caption else "", ""]
            elif kind == "images":
                paths, caption, _ = payload
                out += [" ".join(f"![{p.stem}]({self._rel(p)})" for p in paths), f"*{caption}*" if caption else "", ""]
            elif kind == "kpis":
                out += ["| " + " | ".join(f"**{v}**" for v, _ in payload) + " |", "|" + "---|" * len(payload),
                        "| " + " | ".join(l for _, l in payload) + " |", ""]
            elif kind == "page_break":
                out += ["", "---", ""]
        Path(path).write_text("\n".join(out), encoding="utf-8")
        print(f"Markdown written: {path}")

    def _rel(self, p):
        try:
            return Path(p).resolve().relative_to(self.md_dir.resolve()).as_posix()
        except ValueError:
            import os
            return Path(os.path.relpath(Path(p).resolve(), self.md_dir.resolve())).as_posix()

    # ------------------------------------------------------------------ PDF renderer
    def _img(self, path, width_cm, max_height_cm):
        w, h = ImageReader(str(path)).getSize()
        width = min(width_cm, self.width_cm) * cm
        height = width * h / w
        if height > max_height_cm * cm:
            height = max_height_cm * cm
            width = height * w / h
        return Image(str(path), width=width, height=height)

    def to_pdf(self, path):
        story = [Paragraph(md_to_rl(self.title), STYLES["title"]), Paragraph(md_to_rl(self.subtitle), STYLES["subtitle"])]
        toc = TableOfContents()
        toc.levelStyles = [STYLES["toc0"], STYLES["toc1"]]
        story += [Paragraph("Contents", STYLES["h2"]), toc, PageBreak()]
        first_h1 = True
        for kind, payload in self.elements:
            if kind == "h1":
                if not first_h1:
                    story.append(PageBreak())
                first_h1 = False
                para = Paragraph(md_to_rl(payload), STYLES["h1"])
                para._toc_level = 0
                story.append(para)
            elif kind == "h2":
                para = Paragraph(md_to_rl(payload), STYLES["h2"])
                para._toc_level = 1
                story += [CondPageBreak(3.2 * cm), para]
            elif kind == "h3":
                story += [CondPageBreak(2.5 * cm), Paragraph(md_to_rl(payload), STYLES["h3"])]
            elif kind == "p":
                story.append(Paragraph(md_to_rl(payload), STYLES["body"]))
            elif kind == "bullets":
                story += [Paragraph(f"•&nbsp;&nbsp;{md_to_rl(it)}", STYLES["bullet"]) for it in payload] + [Spacer(1, 4)]
            elif kind == "numbered":
                story += [Paragraph(f"<b>{i}.</b>&nbsp;&nbsp;{md_to_rl(it)}", STYLES["bullet"]) for i, it in enumerate(payload, 1)]
                story.append(Spacer(1, 4))
            elif kind == "code":
                import textwrap
                lines = []
                for line in payload.splitlines():
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    lines += textwrap.wrap(line, 104, subsequent_indent=indent, break_on_hyphens=False) or [""]
                block = Table([[Preformatted("\n".join(lines), STYLES["code"])]], colWidths=[self.width_cm * cm])
                block.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d0d0")),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story += [block, Spacer(1, 8)]
            elif kind == "note":
                k, title, texts = payload
                bg, edge = NOTE_COLORS[k]
                content = [Paragraph(f"<b>{md_to_rl(title)}</b>", STYLES["body"])] if title else []
                content += [Paragraph(md_to_rl(t), STYLES["body"]) for t in texts]
                box = Table([[content]], colWidths=[self.width_cm * cm])
                box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                                         ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                         ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
                story += [Spacer(1, 3), KeepTogether([box]), Spacer(1, 8)]
            elif kind == "table":
                rows, widths, highlight, font_size, caption = payload
                cell = ParagraphStyle("c", parent=STYLES["cell"], fontSize=font_size, leading=font_size + 2.2)
                head = ParagraphStyle("h", parent=STYLES["cell_bold"], fontSize=font_size - 0.4, leading=font_size + 1.8)
                data = [[Paragraph(md_to_rl(str(c)), head) for c in rows[0]]]
                data += [[Paragraph(md_to_rl(str(c)), cell) for c in r] for r in rows[1:]]
                if widths is None:
                    widths = [self.width_cm / len(rows[0])] * len(rows[0])
                t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
                style = [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
                for i in range(1, len(data)):
                    if i % 2 == 0:
                        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7fb")))
                for i in highlight:
                    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3f3e3")))
                t.setStyle(TableStyle(style))
                story += [t] + ([Paragraph(md_to_rl(caption), STYLES["caption"])] if caption else [Spacer(1, 9)])
            elif kind == "image":
                p, caption, width_cm, max_h = payload
                if not p.exists():
                    story.append(Paragraph(f"<i>(figure not available: {p.name})</i>", STYLES["caption"]))
                    continue
                parts = [self._img(p, width_cm, max_h)]
                parts.append(Paragraph(md_to_rl(caption), STYLES["caption"]) if caption else Spacer(1, 8))
                story.append(KeepTogether(parts))
            elif kind == "images":
                paths, caption, max_h = payload
                paths = [p for p in paths if p.exists()]
                if not paths:
                    continue
                col = self.width_cm / len(paths)
                row = Table([[self._img(p, col - 0.3, max_h) for p in paths]], colWidths=[col * cm] * len(paths))
                story.append(KeepTogether([row, Paragraph(md_to_rl(caption), STYLES["caption"]) if caption else Spacer(1, 8)]))
            elif kind == "kpis":
                n = len(payload)
                cells = [[Paragraph(md_to_rl(str(v)), STYLES["kpi_value"]), Paragraph(md_to_rl(l), STYLES["kpi_label"])] for v, l in payload]
                t = Table([[c for c in cells]], colWidths=[self.width_cm / n * cm] * n)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4fa")),
                                       ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c9d6e8")),
                                       ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6e8")),
                                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
                story += [Spacer(1, 3), t, Spacer(1, 9)]
            elif kind == "page_break":
                story.append(PageBreak())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _Doc(path, self.title).multiBuild(story)
        print(f"PDF written: {path}")

    def render(self, stem):
        self.to_markdown(Path(stem).with_suffix(".md"))
        self.to_pdf(Path(stem).with_suffix(".pdf"))
