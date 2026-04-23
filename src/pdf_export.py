"""
pdf_export.py
Kinetric — Customer Analytics Case Study
Produces a polished, letter-format PDF of the recommendation memo using ReportLab.

Design notes:
- Matches memo-container cream treatment (#FAFAF7 → white for print, same body typography)
- Header block mirrors .memo-header layout (To/From/Date/Re pulled as explicit args)
- Body text: Georgia-equivalent (Times-Roman in ReportLab's built-in set)
- Labels / headings: Helvetica-Bold (closest built-in to DM Sans 600)
- Mono labels: Courier (IBM Plex Mono stand-in for header metadata)
- Accent bar at top of page: teal (#2DD4BF) — mirrors ::before rule on .memo-container
- Footer: "Kinetric · kinetric.co · Prepared for client use"
- Page size: letter (8.5 x 11 in) — matches print stylesheet @page declaration
"""

import io
import re
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether


# ── Colour palette — mirrors Kinetric CSS tokens ─────────────────────────────
KIN_TEAL    = colors.HexColor("#2DD4BF")
KIN_DARK    = colors.HexColor("#0B0F1A")
KIN_BODY    = colors.HexColor("#1F2937")
KIN_DIM     = colors.HexColor("#374151")
KIN_MUTED   = colors.HexColor("#5B6B85")
KIN_RULE    = colors.HexColor("#D4D4D0")
KIN_AMBER   = colors.HexColor("#92400E")


# ── Page geometry ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter          # 612 x 792 pt
MARGIN_L = MARGIN_R = 0.85 * inch
MARGIN_T = 1.0 * inch
MARGIN_B = 0.75 * inch
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R


# ── Typography helpers ─────────────────────────────────────────────────────────
def _styles():
    """Return a dict of named ParagraphStyle objects."""
    base = getSampleStyleSheet()
    s = {}

    s["brand"] = ParagraphStyle(
        "brand",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=KIN_DARK,
        spaceAfter=4,
    )
    s["brand_tag"] = ParagraphStyle(
        "brand_tag",
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=KIN_MUTED,
        spaceBefore=0,
        spaceAfter=8,
    )
    s["meta_label"] = ParagraphStyle(
        "meta_label",
        fontName="Courier-Bold",
        fontSize=8,
        leading=11,
        textColor=KIN_MUTED,
        letterSpacing=0.8,
    )
    s["meta_value"] = ParagraphStyle(
        "meta_value",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=KIN_BODY,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=11,
        leading=18,
        textColor=KIN_BODY,
        spaceAfter=10,
        firstLineIndent=0,
    )
    s["bold_lead"] = ParagraphStyle(
        "bold_lead",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=KIN_DARK,
        spaceAfter=4,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=KIN_DARK,
        spaceBefore=14,
        spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "h3",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=KIN_DARK,
        spaceBefore=10,
        spaceAfter=4,
    )
    s["footer"] = ParagraphStyle(
        "footer",
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=KIN_MUTED,
        alignment=TA_CENTER,
    )
    s["ol_item"] = ParagraphStyle(
        "ol_item",
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        textColor=KIN_BODY,
        leftIndent=18,
        spaceAfter=6,
    )
    return s


# ── Accent bar drawn directly on canvas ──────────────────────────────────────
def _draw_accent_bar(canvas, doc):
    """4 pt teal bar at the top of every page — mirrors .memo-container::before."""
    canvas.saveState()
    canvas.setFillColor(KIN_TEAL)
    canvas.rect(0, PAGE_H - 4, PAGE_W, 4, fill=1, stroke=0)
    canvas.restoreState()


def _draw_footer(canvas, doc):
    """Footer line at bottom of every page."""
    canvas.saveState()
    canvas.setStrokeColor(KIN_RULE)
    canvas.setLineWidth(0.5)
    y = MARGIN_B - 18
    canvas.line(MARGIN_L, y + 14, PAGE_W - MARGIN_R, y + 14)
    canvas.setFont("Courier", 7.5)
    canvas.setFillColor(KIN_MUTED)
    text = "Kinetric  ·  kinetric.co  ·  Prepared for client use"
    canvas.drawCentredString(PAGE_W / 2, y, text)
    canvas.restoreState()


def _on_page(canvas, doc):
    _draw_accent_bar(canvas, doc)
    _draw_footer(canvas, doc)


# ── Markdown-to-ReportLab parser ─────────────────────────────────────────────
def _parse_markdown_to_flowables(md_text: str, styles: dict) -> list:
    """
    Converts the memo body markdown (as produced by draft_memo) into ReportLab flowables.

    Handles:
    - **bold text** inline
    - ---  horizontal rule
    - Ordered list items (1. 2. 3.)
    - Bold standalone lines (### H3, plain **Heading** lines)
    - Regular paragraphs
    - Optional trailing "Confidence note:" line from surface_eda_patterns
    """
    flowables = []
    lines = md_text.strip().split("\n")
    i = 0

    def _inline_bold(text: str) -> str:
        """Convert **bold** → <b>bold</b> for ReportLab Paragraph."""
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    while i < len(lines):
        line = lines[i].rstrip()

        # Blank line
        if not line.strip():
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        # Horizontal rule ---
        if re.match(r"^---+$", line.strip()):
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=KIN_RULE))
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # H2  ## Heading
        if line.startswith("## "):
            text = _inline_bold(line[3:].strip())
            flowables.append(Paragraph(text, styles["h2"]))
            i += 1
            continue

        # H3  ### Heading
        if line.startswith("### "):
            text = _inline_bold(line[4:].strip())
            flowables.append(Paragraph(text, styles["h3"]))
            i += 1
            continue

        # Ordered list item  1. text  or  2. text
        ol_match = re.match(r"^(\d+)\.\s+(.*)", line)
        if ol_match:
            num = ol_match.group(1)
            content = _inline_bold(ol_match.group(2))
            flowables.append(Paragraph(f"{num}. {content}", styles["ol_item"]))
            i += 1
            continue

        # Unordered list item  - text  or  * text
        ul_match = re.match(r"^[-*]\s+(.*)", line)
        if ul_match:
            content = _inline_bold(ul_match.group(1))
            flowables.append(Paragraph(f"  {content}", styles["ol_item"]))
            i += 1
            continue

        # Confidence note (trailing line from surface_eda_patterns — flows through)
        if line.strip().startswith("Confidence note:"):
            text = _inline_bold(line.strip())
            flowables.append(Paragraph(f"<i>{text}</i>", styles["body"]))
            i += 1
            continue

        # Bold-only line (acts as a section heading in the memo body)
        if re.match(r"^\*\*[^*]+\*\*$", line.strip()):
            text = line.strip()[2:-2]  # strip the ** wrappers
            flowables.append(Paragraph(text, styles["bold_lead"]))
            i += 1
            continue

        # Regular paragraph — may span multiple adjacent non-empty lines
        paragraph_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("---") and not re.match(r"^(\d+)\.\s+", lines[i]) and not re.match(r"^[-*]\s+", lines[i]) and not lines[i].startswith("## ") and not lines[i].startswith("### "):
            paragraph_lines.append(lines[i].rstrip())
            i += 1
        text = " ".join(paragraph_lines)
        text = _inline_bold(text)
        if text.strip():
            flowables.append(Paragraph(text, styles["body"]))

    return flowables


# ── Public API ────────────────────────────────────────────────────────────────
def build_memo_pdf(
    memo_body_md: str,
    to_: str = "[Client Name], Founder",
    from_: str = "Brendan Hoffman, Kinetric",
    date_: str = "Engagement Closeout",
    re_: str = "Customer Analytics Findings & Recommendations",
) -> bytes:
    """
    Renders the memo as a polished letter-format PDF.

    Args:
        memo_body_md: The markdown body string (draft_memo output or the hand-edited
                      final version). Must NOT include a To/From/Date/Re header block —
                      those are provided separately and rendered as a structured header.
        to_, from_, date_, re_: Header metadata, mirroring .memo-container fields.

    Returns:
        PDF bytes suitable for st.download_button(data=...).
    """
    buf = io.BytesIO()
    styles = _styles()

    doc = BaseDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
        title="Kinetric — Customer Analytics Memo",
        author="Kinetric / Brendan Hoffman",
    )

    frame = Frame(
        MARGIN_L, MARGIN_B,
        CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B,
        id="main",
    )
    template = PageTemplate(id="letter", frames=[frame], onPage=_on_page)
    doc.addPageTemplates([template])

    story = []

    # ── Brand line ────────────────────────────────────────────────────────────
    story.append(Paragraph("Kinetric", styles["brand"]))
    story.append(Paragraph("ADVISORY MEMORANDUM", styles["brand_tag"]))
    story.append(Spacer(1, 4))

    # ── Metadata table (To / From / Date / Re) ────────────────────────────────
    meta_rows = [
        [Paragraph("TO", styles["meta_label"]),   Paragraph(to_,    styles["meta_value"])],
        [Paragraph("FROM", styles["meta_label"]), Paragraph(from_,  styles["meta_value"])],
        [Paragraph("DATE", styles["meta_label"]), Paragraph(date_,  styles["meta_value"])],
        [Paragraph("RE", styles["meta_label"]),   Paragraph(re_,    styles["meta_value"])],
    ]
    meta_table = Table(meta_rows, colWidths=[0.75 * inch, CONTENT_W - 0.75 * inch])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)

    # ── Divider under header ──────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.2, color=KIN_BODY))
    story.append(Spacer(1, 14))

    # ── Body ──────────────────────────────────────────────────────────────────
    body_flowables = _parse_markdown_to_flowables(memo_body_md, styles)
    story.extend(body_flowables)

    doc.build(story)
    return buf.getvalue()
