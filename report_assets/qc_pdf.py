"""
Quality control for a generated PDF: render every page and flag the things a reader would notice.

    python report_assets/qc_pdf.py PROJECT_TECHNICAL_REPORT.pdf [--png]

Checks, per page:
  blank / nearly blank        a page with almost no ink is a bad page break
  overflow                    any text or image whose box crosses the page edge
  tiny type                   text rendered below a legibility floor
  orphan heading              a heading sitting alone in the last few mm of a page
  image scale                 a figure rendered so small its labels cannot be read
  widow line                  a page whose only content is one or two lines
--png also writes every page to report_assets/qc/ so the pages can be looked at.
"""
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
MIN_PT = 6.0                 # below this, body text is not comfortably legible in print
MIN_IMG_WIDTH_PT = 150.0     # a figure narrower than this on an A4 page loses its labels
EDGE_TOL = 1.5               # points of slack before a box counts as crossing the edge


def qc(pdf_path: Path, dump_png: bool):
    doc = fitz.open(pdf_path)
    problems, stats = [], []
    out = ROOT / "report_assets" / "qc"
    if dump_png:
        out.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc, start=1):
        W, H = page.rect.width, page.rect.height
        d = page.get_text("dict")
        spans, sizes, images = [], [], []
        for block in d["blocks"]:
            if block["type"] == 1:
                images.append(fitz.Rect(block["bbox"]))
                continue
            for line in block.get("lines", []):
                for s in line["spans"]:
                    # The page header is a deliberate 1 pt white spacer on every page; it is not content.
                    if s["size"] < 2 and s["text"].strip() in {".", ""}:
                        continue
                    if s["text"].strip():
                        spans.append(s)
                        sizes.append(round(s["size"], 1))

        chars = sum(len(s["text"].strip()) for s in spans)
        ink = page.get_pixmap(dpi=36)
        non_white = sum(1 for p in ink.samples if p < 245) / max(len(ink.samples), 1)
        stats.append((i, chars, len(images), round(non_white * 100, 1)))

        if chars < 40 and not images:
            problems.append(f"page {i}: BLANK or near-blank ({chars} chars, no image)")
        if 0 < chars < 180 and not images:
            problems.append(f"page {i}: WIDOW — only {chars} characters of text and no figure")

        for s in spans:
            r = fitz.Rect(s["bbox"])
            if r.x0 < -EDGE_TOL or r.x1 > W + EDGE_TOL or r.y0 < -EDGE_TOL or r.y1 > H + EDGE_TOL:
                problems.append(f"page {i}: TEXT OVERFLOW {r} outside {W:.0f}x{H:.0f}: "
                                f"{s['text'][:60]!r}")
                break
        for r in images:
            if r.x0 < -EDGE_TOL or r.x1 > W + EDGE_TOL or r.y0 < -EDGE_TOL or r.y1 > H + EDGE_TOL:
                problems.append(f"page {i}: IMAGE OVERFLOW {r} outside {W:.0f}x{H:.0f}")
            if r.width < MIN_IMG_WIDTH_PT:
                problems.append(f"page {i}: TINY FIGURE {r.width:.0f}pt wide — labels unreadable")

        small = [z for z in sizes if z < MIN_PT]
        if small:
            problems.append(f"page {i}: TINY TYPE {sorted(set(small))} pt on {len(small)} spans")

        # a heading is a large bold span; flag one that lands in the last 28 pt of a page
        for s in spans:
            if s["size"] >= 12 and s["bbox"][1] > H - 28:
                problems.append(f"page {i}: ORPHAN HEADING at y={s['bbox'][1]:.0f}: {s['text'][:50]!r}")
                break

        if dump_png:
            page.get_pixmap(dpi=110).save(str(out / f"page_{i:03d}.png"))

    print(f"{pdf_path.name}: {len(doc)} pages")
    lo = sorted(stats, key=lambda r: r[3])[:6]
    print("lightest pages (page, chars, images, % ink):", lo)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
    else:
        print("\nno problems found")
    return problems


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    qc(ROOT / args[0], "--png" in sys.argv)
