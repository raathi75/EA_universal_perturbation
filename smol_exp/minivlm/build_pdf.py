"""RAPPORT.md -> RAPPORT.pdf (A4, tables and figures), with reportlab only (no LaTeX / pandoc needed).
Handles the Markdown subset used by the report: headings, paragraphs, bullet and numbered lists (nested), pipe
tables, images ![caption](path), horizontal rules, **bold**, `code`, [links](...).
  python make_figures.py && python build_pdf.py
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, HRFlowable)

HERE = Path(__file__).resolve().parent
SRC = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'RAPPORT.md')
DST = SRC.with_suffix('.pdf')
EN = SRC.stem.upper().endswith('_EN')
FOOTER = ('Universal perturbations and VLMs — experiment report' if EN else
          'Perturbations universelles et VLM — rapport d\'expériences')

import matplotlib
F = str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf') + '/'   # DejaVu, shipped with matplotlib
pdfmetrics.registerFont(TTFont('Sans', F + 'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('Sans-Bold', F + 'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Sans-Oblique', F + 'DejaVuSans-Oblique.ttf'))
pdfmetrics.registerFont(TTFont('Sans-BoldOblique', F + 'DejaVuSans-BoldOblique.ttf'))
pdfmetrics.registerFont(TTFont('Mono', F + 'DejaVuSansMono.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Sans', normal='Sans', bold='Sans-Bold', italic='Sans-Oblique', boldItalic='Sans-BoldOblique')

INK, INK2, ACCENT, RULE, HEAD_BG = (colors.HexColor(c) for c in ['#0b0b0b', '#52514e', '#2a78d6', '#d8d7d2', '#f1f0ec'])
body = ParagraphStyle('body', fontName='Sans', fontSize=9.2, leading=13, textColor=INK, spaceAfter=5)
cell = ParagraphStyle('cell', parent=body, fontSize=7.6, leading=9.6, spaceAfter=0)
cell_h = ParagraphStyle('cellh', parent=cell, fontName='Sans-Bold')
cap = ParagraphStyle('cap', parent=body, fontSize=8, leading=10.5, textColor=INK2, alignment=TA_CENTER, spaceAfter=10)
H = {1: ParagraphStyle('h1', parent=body, fontName='Sans-Bold', fontSize=16, leading=20, spaceBefore=4, spaceAfter=10),
     2: ParagraphStyle('h2', parent=body, fontName='Sans-Bold', fontSize=13, leading=16, spaceBefore=12, spaceAfter=6,
                       textColor=ACCENT),
     3: ParagraphStyle('h3', parent=body, fontName='Sans-Bold', fontSize=10.5, leading=14, spaceBefore=9, spaceAfter=4)}
W = A4[0] - 3.6 * cm


def inline(t):
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    t = re.sub(r'`([^`]+)`', lambda m: '<font name="Mono" size="0.92em">%s</font>' % m.group(1).replace('*', '&#42;'), t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<font color="#2a78d6">\1</font>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', t)
    t = t.replace('size="0.92em"', 'size="8"')
    return t


def table(rows):
    rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    head, data = rows[0], [r for r in rows[2:]]
    n = len(head)
    data = [r + [''] * (n - len(r)) for r in data]
    lens = [max(len(re.sub(r'[*`]', '', r[i])) for r in [head] + data) for i in range(n)]
    words = [max(len(wd) for r in [head] + data for wd in re.sub(r'[*`]', '', r[i]).split() or ['']) for i in range(n)]
    w = [min(max(l, 6, 1.45 * wd + 3), 60) for l, wd in zip(lens, words)]
    w = [W * x / sum(w) for x in w]
    cells = [[Paragraph(inline(c), cell_h) for c in head]] + [[Paragraph(inline(c), cell) for c in r] for r in data]
    t = Table(cells, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), HEAD_BG), ('LINEBELOW', (0, 0), (-1, 0), 0.8, INK2),
                           ('LINEBELOW', (0, 1), (-1, -1), 0.3, RULE), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                           ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
    return [t, Spacer(1, 7)]


def image(caption, path):
    p = SRC.parent / path   # images are relative to the Markdown file
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    w = min(W, iw / 200 * 72 * 1.0)   # figures are saved at 200 dpi
    w = W if w > 0.75 * W else w
    return [KeepTogether([Spacer(1, 4), Image(str(p), width=w, height=w * ih / iw), Paragraph(inline(caption), cap)])]


def lists(lines):
    """Nested bullet / numbered lists from lines with their indentation."""
    def build(items, start):
        out, i = [], start
        base = items[start][0]
        kind = items[start][1]
        while i < len(items) and items[i][0] >= base:
            ind, k, text, _ = items[i]
            if ind > base:
                sub, i = build(items, i)
                out[-1].append(sub)
                continue
            out.append([Paragraph(inline(text), ParagraphStyle('li', parent=body, spaceAfter=2))])
            i += 1
        fl = ListFlowable([ListItem(x if len(x) > 1 else x[0], leftIndent=12) if len(x) == 1 else
                           ListItem(x, leftIndent=12) for x in out],
                          bulletType='1' if kind == 'num' else 'bullet', start=str(items[start][3]) if kind == 'num' else None,
                          bulletFormat='%s.' if kind == 'num' else None,
                          bulletFontName='Sans', bulletFontSize=8 if kind == 'bullet' else 8.5, leftIndent=12,
                          bulletColor=INK2, spaceAfter=3)
        return fl, i
    items = []
    for l in lines:
        m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', l)
        if m:
            items.append([len(m.group(1)), 'num' if m.group(2)[0].isdigit() else 'bullet', m.group(3),
                          int(m.group(2)[:-1]) if m.group(2)[0].isdigit() else 1])
        else:   # continuation line
            items[-1][2] += ' ' + l.strip()
    fl, _ = build(items, 0)
    return [fl, Spacer(1, 3)]


def parse(md):
    story, lines, i = [], md.split('\n'), 0
    para = []

    def flush():
        if para:
            story.append(Paragraph(inline(' '.join(para)), body))
            para.clear()
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush(); i += 1; continue
        m = re.match(r'^(#{1,3})\s+(.*)', l)
        if m:
            flush()
            lvl = len(m.group(1))
            if lvl == 2 and story:
                story.append(Spacer(1, 4))
            story.append(Paragraph(inline(m.group(2)), H[lvl]))
            i += 1; continue
        if re.match(r'^---+\s*$', l):
            flush(); story.append(HRFlowable(width='100%', color=RULE, thickness=0.6, spaceBefore=4, spaceAfter=6))
            i += 1; continue
        m = re.match(r'^!\[(.*)\]\((.*)\)\s*$', l)
        if m:
            flush(); story += image(m.group(1), m.group(2)); i += 1; continue
        if l.lstrip().startswith('|'):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                rows.append(lines[i]); i += 1
            story += table(rows); continue
        if re.match(r'^\s*([-*]|\d+\.)\s+', l):
            flush()
            block = []
            while i < len(lines) and lines[i].strip() and (re.match(r'^\s*([-*]|\d+\.)\s+', lines[i])
                                                            or lines[i].startswith('   ')):
                block.append(lines[i]); i += 1
            story += lists(block); continue
        para.append(l.strip()); i += 1
    flush()
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Sans', 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(1.8 * cm, 1.1 * cm, FOOTER)
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


if __name__ == '__main__':
    doc = SimpleDocTemplate(str(DST), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm,
                            bottomMargin=1.8 * cm, title='Universal perturbations and VLMs' if EN else 'Perturbations universelles et VLM', author='')
    doc.build(parse(SRC.read_text()), onFirstPage=footer, onLaterPages=footer)
    print('>> Saved', DST)
