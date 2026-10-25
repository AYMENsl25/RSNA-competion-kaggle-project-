"""Build the technical report from public aggregate tables, without data access.

Requires reportlab. The optional --check mode additionally requires pypdf.
There are no network, training, model-loading or patient-data operations.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

from reportlab import __version__ as reportlab_version
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                               Spacer, Table, TableStyle, Preformatted, KeepTogether)

HERE = Path(__file__).resolve().parent
BLUE = colors.HexColor('#245a81')
ORANGE = colors.HexColor('#a85027')
INK = colors.HexColor('#182b3a')
MUTED = colors.HexColor('#526775')
LIGHT = colors.HexColor('#edf3f6')
WIDTH = A4[0] - 100


def architecture():
    d = Drawing(495, 264)
    columns = [15, 180, 345]
    titles = ['Independent MRI reader', 'Frozen-feature study', 'Public-weight track']
    rows = [
        ['DICOM geometry + cache', 'Canonical six-slot cache', 'Released knee checkpoints'],
        ['Adjacent-slice windows', 'Frozen DINOv2 features', 'Raptor 384 / 224 + R6'],
        ['ResNet18 / ConvNeXt-T', 'Projection + series context', 'Upstream small reader'],
        ['Max / finding attention', 'Geometry + finding attention', 'Original / altered fusion'],
        ['12 logits; Ens4 rank blend', '12 logits; proxy evaluation', 'Separate score attribution'],
    ]
    for c, x in enumerate(columns):
        d.add(String(x + 67, 245, titles[c], fontName='Helvetica-Bold', fontSize=9,
                     textAnchor='middle', fillColor=INK))
        for r, row in enumerate(rows):
            y = 204 - r * 40
            d.add(Rect(x, y, 136, 29, rx=4, ry=4, fillColor=LIGHT,
                       strokeColor=BLUE if c != 2 else MUTED, strokeWidth=.7))
            d.add(String(x + 68, y + 11, row[c], fontName='Helvetica', fontSize=7.2,
                         textAnchor='middle', fillColor=INK))
            if r < len(rows) - 1:
                d.add(Line(x + 68, y, x + 68, y - 11, strokeColor=MUTED, strokeWidth=.7))
    d.add(String(247, 8, 'Separate pipelines and endpoints; schematic, not patient imagery.',
                 fontSize=8, textAnchor='middle', fillColor=MUTED))
    return d


def effects(data):
    rows = data['paired_effects']
    d = Drawing(495, 246)
    left, right, lo, hi = 225, 470, -.02, .03
    def x(v): return left + (v - lo) / (hi - lo) * (right - left)
    for tick in [-.02, -.01, 0, .01, .02, .03]:
        d.add(Line(x(tick), 42, x(tick), 223, strokeColor=colors.HexColor('#dbe4e9'), strokeWidth=.6))
        d.add(String(x(tick), 25, f'{tick:+.2f}', fontSize=8, textAnchor='middle', fillColor=MUTED))
    d.add(Line(x(0), 42, x(0), 223, strokeColor=MUTED, strokeWidth=1))
    for i, row in enumerate(rows):
        y = 208 - i * 29
        color = BLUE if row['delta'] >= 0 else ORANGE
        d.add(String(8, y - 3, row['label'], fontSize=8.2, fillColor=INK))
        a, b = row['ci95']
        d.add(Line(x(a), y, x(b), y, strokeWidth=2, strokeColor=color))
        d.add(Line(x(a), y - 3, x(a), y + 3, strokeColor=color))
        d.add(Line(x(b), y - 3, x(b), y + 3, strokeColor=color))
        d.add(Circle(x(row['delta']), y, 3, fillColor=color, strokeColor=color))
    d.add(String(348, 8, 'Candidate minus its own control (proxy AUC)',
                 fontSize=8, textAnchor='middle', fillColor=MUTED))
    return d


def markup(text):
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', r'<link href="\2" color="#245a81">\1</link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    return text


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=24,
                         leading=29, textColor=INK, spaceAfter=16))
    s.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=9.3,
                         leading=14, textColor=INK, spaceAfter=8, alignment=TA_LEFT))
    s.add(ParagraphStyle(name='ReportH1', fontName='Helvetica-Bold', fontSize=14,
                         leading=19, textColor=BLUE, spaceBefore=15, spaceAfter=8,
                         keepWithNext=True))
    s.add(ParagraphStyle(name='ReportH2', fontName='Helvetica-Bold', fontSize=11,
                         leading=15, textColor=INK, spaceBefore=11, spaceAfter=6,
                         keepWithNext=True))
    s.add(ParagraphStyle(name='ReportCell', fontName='Helvetica', fontSize=7.3,
                         leading=10, textColor=INK))
    s.add(ParagraphStyle(name='ReportCode', fontName='Courier', fontSize=7.5,
                         leading=11, textColor=INK, backColor=LIGHT, borderPadding=7,
                         spaceBefore=5, spaceAfter=10))
    return s


def parse_markdown(text, drawings):
    s = styles()
    lines = text.splitlines()
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            flow.append(Preformatted('\n'.join(code), s['ReportCode']))
        elif line.startswith('!['):
            name = re.search(r'\(([^)]+)\)', line).group(1)
            flow.extend([Spacer(1, 8), drawings[name], Spacer(1, 5)])
        elif line.startswith('|'):
            matrix = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-]+', c.replace(' ', '')) for c in cells):
                    matrix.append([Paragraph(markup(c), s['ReportCell']) for c in cells])
                i += 1
            n = len(matrix[0])
            if n == 5:
                widths = [WIDTH * f for f in [.28, .13, .13, .16, .30]]
            elif n == 4:
                widths = [WIDTH * f for f in [.34, .18, .18, .30]]
            elif n == 3:
                widths = [WIDTH * f for f in [.38, .20, .42]]
            else:
                widths = [WIDTH / n] * n
            t = Table(matrix, colWidths=widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, 0), .7, BLUE),
                ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d9e2e8')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            flow.extend([t, Spacer(1, 9)])
            continue
        elif line.startswith('# '):
            flow.append(Paragraph(markup(line[2:]), s['ReportTitle']))
        elif line.startswith('## '):
            flow.append(Paragraph(markup(line[3:]), s['ReportH1']))
        elif line.startswith('### '):
            flow.append(Paragraph(markup(line[4:]), s['ReportH2']))
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '```', '![')):
                para.append(lines[i].strip())
                i += 1
            flow.append(Paragraph(markup(' '.join(para)), s['ReportBody']))
            continue
        i += 1
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbd8df'))
    canvas.line(50, 42, A4[0] - 50, 42)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(50, 29, 'RSNA Knee MRI | Technical report v1.0 | 25 October 2026')
    canvas.drawRightString(A4[0] - 50, 29, str(doc.page))
    canvas.restoreState()


def check():
    from pypdf import PdfReader
    data = json.loads((HERE / 'data/aggregate_results.json').read_text())
    reader = PdfReader(HERE / 'rsna-knee-technical-report.pdf')
    text = '\n'.join(p.extract_text() for p in reader.pages)
    for token in ['Abstract', '0.837', '0.942', '0.807809', 'References', 'Appendix A']:
        if token not in text:
            raise ValueError(f'missing PDF text: {token}')
    if len(reader.pages) < 5 or any(len(p.extract_text().strip()) < 120 for p in reader.pages):
        raise ValueError('unexpected short/empty PDF page')
    for row in data['paired_effects']:
        if not row['ci95'][0] <= row['delta'] <= row['ci95'][1]:
            raise ValueError('invalid interval')
    report = (HERE / 'REPORT.md').read_text()
    for row in data['frozen_feature_proxy'] + data['early_proxy']:
        if f"{row['auc']:.6f}" not in report:
            raise ValueError(f"aggregate score missing from report: {row['id']}")
    manifest = json.loads((HERE / 'artifact_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'artifact checksum mismatch: {name}')
    for md in [HERE / 'REPORT.md', HERE / 'README.md', HERE / 'BUILD.md']:
        for target in re.findall(r'\]\(([^)]+)\)', md.read_text()):
            if not target.startswith(('http://', 'https://', '#')) and not (md.parent / target).is_file():
                raise ValueError(f'broken local link: {target}')
    forbidden = re.compile(r'\b(?:\d+\.){4,}\d+\b|C:\\\\Users|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}')
    for p in HERE.rglob('*'):
        if p.is_file() and p.suffix in {'.md', '.json', '.svg'} and forbidden.search(p.read_text(encoding='utf-8')):
            raise ValueError(f'possible private identifier/path/secret in {p.name}')
    return {'status': 'passed', 'pages': len(reader.pages), 'pdf_text_characters': len(text),
            'aggregate_only': True, 'relative_links_checked': True,
            'interval_order_checked': True, 'text_content_scan': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate existing outputs, without building')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), indent=2))
        return
    data = json.loads((HERE / 'data/aggregate_results.json').read_text())
    drawings = {'figures/architecture.svg': architecture(), 'figures/paired-effects.svg': effects(data)}
    (HERE / 'figures').mkdir(exist_ok=True)
    for name, drawing in drawings.items():
        renderSVG.drawToFile(drawing, str(HERE / name))
    doc = BaseDocTemplate(str(HERE / 'rsna-knee-technical-report.pdf'), pagesize=A4,
                          leftMargin=50, rightMargin=50, topMargin=48, bottomMargin=57,
                          title='Budgeted Model Selection for Weakly Supervised Knee MRI',
                          author='RSNA Knee Research Project', invariant=1)
    frame = Frame(50, 57, WIDTH, A4[1] - 105, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='report', frames=[frame], onPage=footer))
    doc.build(parse_markdown((HERE / 'REPORT.md').read_text(), drawings))
    manifest = {'report_version': '1.0', 'report_date': '2026-10-25',
                'reportlab_version': reportlab_version, 'invariant_pdf': True,
                'reproduction_scope': 'report and figures from aggregate data; no training reproduction',
                'files': {}}
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts:
            manifest['files'][p.relative_to(HERE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status':'built', 'pdf':doc.filename, 'artifacts':len(manifest['files'])}, indent=2))


if __name__ == '__main__':
    main()
