"""Build an illustrated competition write-up from aggregate report results.

Uses ReportLab for diagrams/charts, Poppler for PNG rasterization, and Pillow
for PNG validation. No network, training, labels or model artifacts are read.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.lib import colors

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'technical-report/data/aggregate_results.json'
BLUE = colors.HexColor('#187ea1')
ORANGE = colors.HexColor('#bd663d')
NAVY = colors.HexColor('#142b3d')
MUTED = colors.HexColor('#556a78')
LIGHT = colors.HexColor('#eef4f7')
GRID = colors.HexColor('#d7e2e8')


def text(d, x, y, value, size=16, color=NAVY, bold=False, anchor='start'):
    d.add(String(x, y, value, fontName='Helvetica-Bold' if bold else 'Helvetica',
                 fontSize=size, fillColor=color, textAnchor=anchor))


def background(width, height, title, subtitle):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=None))
    d.add(Rect(0, height - 10, width, 10, fillColor=BLUE, strokeColor=None))
    text(d, 38, height - 54, title, 25, bold=True)
    text(d, 38, height - 84, subtitle, 13, MUTED)
    return d


def card(d, x, y, width, height, title, lines, dark=False):
    d.add(Rect(x, y, width, height, rx=10, ry=10,
               fillColor=NAVY if dark else LIGHT, strokeColor=None))
    text(d, x + 18, y + height - 28, title, 16, colors.white if dark else NAVY, True)
    for i, line in enumerate(lines):
        text(d, x + 18, y + height - 55 - i * 22, line, 12,
             colors.HexColor('#dcebf1') if dark else MUTED)


def arrow(d, x1, y, x2):
    d.add(Line(x1, y, x2, y, strokeColor=BLUE, strokeWidth=2))
    d.add(Polygon([x2, y, x2 - 8, y + 4, x2 - 8, y - 4], fillColor=BLUE, strokeColor=None))


def overview(data):
    d = Drawing(1080, 450)
    d.add(Rect(0, 0, 1080, 450, fillColor=NAVY, strokeColor=None))
    d.add(Rect(38, 392, 54, 5, fillColor=colors.HexColor('#49bdd9'), strokeColor=None))
    text(d, 38, 355, 'RSNA KNEE MRI', 17, colors.HexColor('#8ed9e8'), True)
    text(d, 38, 304, 'What worked. What failed.', 36, colors.white, True)
    text(d, 38, 260, 'What we learned.', 36, colors.white, True)
    for x, value, label, detail in [(38,'0.837','Our trained ensemble','Ens4 - private AUC'),
                                   (386,'0.942','Public-stack reproduction','F0 - private AUC'),
                                   (734,'12','Study-level findings','MRI-only inference')]:
        d.add(Rect(x, 59, 308, 150, rx=9, ry=9, fillColor=colors.HexColor('#203f54'), strokeColor=None))
        text(d, x + 20, 151, value, 37, colors.white, True)
        text(d, x + 20, 112, label, 15, colors.white, True)
        text(d, x + 20, 84, detail, 12, colors.HexColor('#b8d0dd'))
    text(d, 38, 25, 'A project write-up with controlled ablations, negative results and clear source attribution.',
         12, colors.HexColor('#b8d0dd'))
    return d


def pipeline(data):
    d = background(1080, 570, 'From MRI series to twelve predictions',
                   'Independent reader and Ens4; original schematic with no patient imagery')
    stages = [
        ('01  DICOM series', ['Sagittal / coronal / axial', 'Order by physical geometry']),
        ('02  Canonical cache', ['160-mm field of view', '256-px uint8 volumes']),
        ('03  Shared encoder', ['Three adjacent slices', '224-px ResNet / ConvNeXt']),
        ('04  Study readout', ['Max or finding attention', 'Masked windows; 12 logits'])]
    for i, (title, lines) in enumerate(stages):
        x = 38 + i * 258
        card(d, x, 310, 230, 143, title, lines)
        if i < 3:
            arrow(d, x + 233, 379, x + 253)
    text(d, 38, 266, 'Four independently trained members', 19, bold=True)
    members = [('E4+E5','ResNet18 / max'), ('E6','Augmented / averaged'),
               ('E2','Slabs / attention'), ('E8','ConvNeXt-T / max')]
    for i, (title, line) in enumerate(members):
        x = 38 + i * 258
        card(d, x, 139, 230, 92, title, [line])
        d.add(Line(x + 115, 139, x + 115, 112, strokeColor=BLUE, strokeWidth=1.3))
    d.add(Line(153, 112, 927, 112, strokeColor=BLUE, strokeWidth=1.3))
    d.add(Line(540, 112, 540, 94, strokeColor=BLUE, strokeWidth=1.3))
    d.add(Rect(180, 39, 720, 53, rx=8, ry=8, fillColor=NAVY, strokeColor=None))
    text(d, 540, 59, 'Equal-weight per-finding rank average -> 12 study scores',
         17, colors.white, True, 'middle')
    return d


def add_bars(d, records, value_keys, labels, x, y, width, height, axis_max=1):
    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = x, y, width, height
    chart.data = [tuple(r[key] for r in reversed(records)) for key in value_keys]
    chart.categoryAxis.categoryNames = list(reversed(labels))
    chart.categoryAxis.labels.fontName = 'Helvetica'
    chart.categoryAxis.labels.fontSize = 13
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = axis_max
    chart.valueAxis.valueStep = .2 if axis_max == 1 else 1
    chart.valueAxis.labels.fontSize = 11
    chart.valueAxis.strokeColor = GRID
    chart.valueAxis.gridStrokeColor = GRID
    chart.valueAxis.visibleGrid = True
    chart.categoryAxis.strokeColor = GRID
    chart.barWidth = 12
    chart.groupSpacing = 15
    chart.barSpacing = 4
    chart.bars[0].fillColor = BLUE
    if len(value_keys) > 1:
        chart.bars[1].fillColor = ORANGE
    chart.barLabelFormat = '%.3f' if axis_max == 1 else '%.2f'
    chart.barLabels.fontName = 'Helvetica'
    chart.barLabels.fontSize = 11
    chart.barLabels.boxAnchor = 'w'
    chart.barLabels.dx = 5
    d.add(chart)


def progress(data):
    d = background(1080, 560, 'Our independently trained submissions',
                   'Public and private competition AUC; historical packages, not isolated component effects')
    rows = [r for r in data['leaderboard'] if r['track'] == 'independent']
    add_bars(d, rows, ['public_auc','private_auc'], [r['id'] for r in rows],
             140, 105, 850, 320)
    for x, color, label in [(140, BLUE, 'Public'), (285, ORANGE, 'Private')]:
        d.add(Rect(x, 457, 18, 12, fillColor=color, strokeColor=None))
        text(d, x + 29, 457, label, 14)
    text(d, 140, 46, 'AUC axis starts at zero. Public-stack F0 is a separate track.', 13, MUTED)
    return d


def coverage(data):
    d = background(1080, 550, 'More coverage was not automatically better',
                   'Same three-epoch early protocol, 863 proxy-validation studies; separate AUC and cost axes')
    rows = data['early_proxy']
    labels = ['M00: random init','P00: ImageNet','D01: more windows','S01: two series']
    text(d, 190, 415, 'Report-proxy macro AUC', 16, bold=True)
    text(d, 700, 415, 'Measured script hours', 16, bold=True)
    add_bars(d, rows, ['auc'], labels, 200, 120, 320, 260)
    add_bars(d, rows, ['hours'], ['M00','P00','D01','S01'], 695, 120, 290, 260, 5)
    text(d, 38, 55, 'S01 retained the encoder window count but increased selected-series decoding work.', 13, MUTED)
    text(d, 38, 30, 'This comparison does not rule out stronger multi-series models or longer training.', 13, MUTED)
    return d


def replication(data):
    d = background(1080, 430, 'The attention gain did not replicate',
                   'Candidate minus matched max-pool control; fixed endpoint on the same proxy reference')
    left, right, low, high = 385, 985, -.012, .022
    def scale(v): return left + (v - low) / (high - low) * (right - left)
    for tick in [-.01, 0, .01, .02]:
        d.add(Line(scale(tick), 135, scale(tick), 296, strokeColor=GRID, strokeWidth=1))
        text(d, scale(tick), 112, f'{tick:+.3f}', 13, MUTED, anchor='middle')
    d.add(Line(scale(0), 135, scale(0), 296, strokeColor=MUTED, strokeWidth=1.5))
    for i, r in enumerate(data['paired_effects'][:2]):
        y=264-i*83
        text(d, 38, y+8, ['Seed 0 / laptop','Seed 1 / Kaggle'][i], 19, bold=True)
        text(d, 38, y-18, f"Difference: {r['delta']:+.4f}", 14, MUTED)
        a,b=r['ci95']; color=BLUE if i==0 else ORANGE
        d.add(Line(scale(a),y,scale(b),y,strokeColor=color,strokeWidth=4))
        d.add(Circle(scale(r['delta']),y,6,fillColor=color,strokeColor=None))
    text(d, 38, 63, 'Intervals resample known groups for a fitted run; they do not measure training-seed variance.', 13, MUTED)
    text(d, 38, 35, 'Each pair is internally hardware-matched; cross-pair differences also include environment changes.', 13, MUTED)
    return d


def frozen(data):
    d = background(1080, 600, 'A cheap frozen-feature follow-up',
                   'DINOv2 ViT-S/14 features -> small CPU heads; a separate exploratory proxy experiment')
    steps=[('Frozen image encoder',['224-px slice triplets','Cache 384-d features']),
           ('Contextual head B',['Within-series attention','Slot + geometry inputs']),
           ('Per-finding pooling',['12 weighted study features','12 output logits'])]
    for i,(title,lines) in enumerate(steps):
        card(d,38+i*350,350,310,132,title,lines)
        if i<2: arrow(d,354+i*350,415,379+i*350)
    rows=data['frozen_feature_proxy']
    labels=['A: simple attention','B: context + geometry','C: B, zero geometry','B + ranking loss','B + series dropout']
    for i,(r,label) in enumerate(zip(rows,labels)):
        y=305-i*43
        text(d,48,y,label,17,bold=i==3)
        text(d,485,y,f"{r['auc']:.6f}",20,BLUE if i==3 else NAVY,True)
    card(d,688,118,350,205,'Interpret the controls',
         ['A -> B changes several things.','B vs C: geometry adds ~0.0014.',
          'Ranking: +0.0089 vs B.','No independent test confirmation.'])
    text(d,38,47,'Original GPU feature extraction: ~7.4 minutes. Two F4 CPU workers: 84 seconds wall time.',13,MUTED)
    text(d,38,23,'The geometry control was post hoc. These values are not competition leaderboard scores.',13,MUTED)
    return d


def inline(s):
    s=html.escape(s, quote=False)
    s=re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)',r'<a href="\2">\1</a>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',s)
    return s


def render_markdown(markdown):
    lines=markdown.splitlines(); out=[]; i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s: i+=1; continue
        if s.startswith('!['):
            alt, name=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',s).groups()
            encoded=base64.b64encode((HERE/name).read_bytes()).decode()
            out.append(f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}"></figure>')
        elif s.startswith('```'):
            code=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]); i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>')
        elif s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[: -]+',c) for c in cells): rows.append(cells)
                i+=1
            out.append('<div class="table-wrap"><table>')
            for k,row in enumerate(rows):
                tag='th' if k==0 else 'td'; out.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row)+'</tr>')
            out.append('</table></div>'); continue
        elif s.startswith('#'):
            m=re.match(r'^(#{1,3}) (.*)',s)
            level=len(m[1]); slug=re.sub('[^a-z0-9]+','-',m[2].lower()).strip('-')
            out.append(f'<h{level} id="{slug}">{inline(m[2])}</h{level}>')
        elif s.startswith('- ') or re.match(r'^\d+\. ',s):
            ordered=not s.startswith('- '); tag='ol' if ordered else 'ul'; items=[]
            while i<len(lines) and (re.match(r'^\d+\. ',lines[i]) if ordered else lines[i].startswith('- ')):
                item=re.sub(r'^(?:\d+\. |- )','',lines[i]); i+=1
                while i<len(lines) and lines[i].startswith('  '): item+=' '+lines[i].strip(); i+=1
                items.append('<li>'+inline(item)+'</li>')
            out.append(f'<{tag}>'+''.join(items)+f'</{tag}>'); continue
        else:
            p=[s]; i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','```','- ')):
                p.append(lines[i].strip()); i+=1
            out.append('<p>'+inline(' '.join(p))+'</p>'); continue
        i+=1
    return '\n'.join(out)


def validate():
    from PIL import Image
    md=(HERE/'WRITEUP.md').read_text()
    images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
    assert len(images)==6
    for name in images:
        with Image.open(HERE/name) as im:
            assert im.width>=1500 and im.height>=500
            im.verify()
    page=(HERE/'writeup.html').read_text()
    assert page.count('data:image/png;base64,')==6
    assert '<script' not in page and 'src="http' not in page
    link_text=re.sub(r'```.*?```','',md,flags=re.S)
    for target in re.findall(r'\]\(([^)]+)\)',link_text):
        assert target.startswith(('http://','https://','#')) or (HERE/target).is_file(), target
    for value in ['0.837','0.942','0.807809','0.739719']:
        assert value in md and value in page
    manifest=json.loads((HERE/'manifest.json').read_text())
    for path,value in manifest['files'].items():
        assert hashlib.sha256((HERE/path).read_bytes()).hexdigest()==value, path
    return {'status':'passed','images':len(images),'standalone_html':True,
            'relative_links':'passed','checksums':'passed','patient_images':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdftoppm',default=shutil.which('pdftoppm'))
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if args.check: print(json.dumps(validate(),indent=2)); return
    if not args.pdftoppm: raise SystemExit('Provide --pdftoppm /absolute/path/to/pdftoppm')
    data=json.loads(DATA.read_text())
    figures=[('01-overview',overview),('02-pipeline',pipeline),('03-submission-progress',progress),
             ('04-coverage-and-cost',coverage),('05-attention-replication',replication),('06-frozen-feature-study',frozen)]
    (HERE/'assets').mkdir(exist_ok=True)
    temporary=HERE.parents[1]/'tmp/kaggle-writeup-build'; temporary.mkdir(parents=True,exist_ok=True)
    for name,draw in figures:
        pdf=temporary/f'{name}.pdf'; renderPDF.drawToFile(draw(data),str(pdf))
        subprocess.run([args.pdftoppm,'-singlefile','-scale-to','1800','-png',str(pdf),str(HERE/'assets'/name)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    body=render_markdown((HERE/'WRITEUP.md').read_text())
    css='''*{box-sizing:border-box}body{margin:0;background:#f0f4f7;color:#182d3c;font:17px/1.75 system-ui,-apple-system,Segoe UI,sans-serif}article{max-width:1000px;margin:40px auto;padding:44px 60px;background:white;border:1px solid #dce5eb;border-radius:16px}h1{font-size:38px;line-height:1.2;letter-spacing:-1px;margin:12px 0 24px}h2{font-size:27px;line-height:1.35;margin-top:56px;border-top:1px solid #dce5eb;padding-top:25px}h3{font-size:22px;margin-top:35px}a{color:#087d9d}p,li{max-width:78ch}img{display:block;width:100%;height:auto;border-radius:10px}figure{margin:28px -15px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px 14px;border-bottom:1px solid #dce5eb;text-align:left;vertical-align:top}th{background:#eef4f7}pre{white-space:pre-wrap;padding:20px;background:#f0f5f8;font-size:13px;border-radius:8px}code{font-family:ui-monospace,monospace}footer{margin-top:50px;color:#5a6b78;font-size:13px}.table-wrap{overflow-x:auto}em{color:#596b78}@media(max-width:700px){article{margin:0;padding:24px 20px;border:0;border-radius:0}h1{font-size:30px}h2{font-size:23px}figure{margin:24px -5px}body{font-size:16px}}'''
    page=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Our RSNA Knee MRI Solution</title><style>{css}</style></head><body><article>{body}<footer>Illustrated project write-up - 25 October 2026. Offline preview; no external scripts or image requests.</footer></article></body></html>'
    (HERE/'writeup.html').write_text(page,encoding='utf-8')
    manifest={'version':'1.0','date':'2026-10-25','aggregate_source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'scope':'Aggregate-only illustrations and write-up; not uploaded to Kaggle','files':{}}
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f.name!='manifest.json' and '__pycache__' not in f.parts:
            manifest['files'][f.relative_to(HERE).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(validate(),indent=2))


if __name__=='__main__': main()
