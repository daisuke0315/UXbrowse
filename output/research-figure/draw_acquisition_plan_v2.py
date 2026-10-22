"""New vector acquisition schematic; positions are illustrative, not measured.

All camera bodies, lens fronts, arrows and axes share one geometric direction.
No experimental photograph is altered or used in this drawing.
"""
import math
from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H, SCALE = 1560, 1150, 3
O = (780.0, 520.0)
HALF = 145.0
im = Image.new('RGB', (W*SCALE, H*SCALE), 'white')
d = ImageDraw.Draw(im)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="white"/>']
fontroot = Path('/System/Library/Fonts/Supplemental')

def pts(ps):
    return [(round(x*SCALE), round(y*SCALE)) for x,y in ps]

def line(ps, color='#41464d', width=3):
    d.line(pts(ps), fill=color, width=round(width*SCALE))
    svg.append(f'<polyline points="{" ".join(f"{x:.4f},{y:.4f}" for x,y in ps)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

def polygon(ps, fill, stroke='#41464d', width=3):
    d.polygon(pts(ps), fill=fill)
    if stroke:
        d.line(pts(ps+[ps[0]]), fill=stroke, width=round(width*SCALE), joint='curve')
    svg.append(f'<polygon points="{" ".join(f"{x:.4f},{y:.4f}" for x,y in ps)}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')

def text(x, y, s, size=29, bold=False, color='#252b32'):
    font = ImageFont.truetype(str(fontroot/('Arial Bold.ttf' if bold else 'Arial.ttf')), round(size*SCALE))
    d.text((x*SCALE,y*SCALE), s, font=font, anchor='mm', fill=color)
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(s)}</text>')

def dash(a, b, color='#bd574b', width=2.3, on=10, off=8):
    vx,vy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(vx,vy)
    ux,uy=vx/length,vy/length
    for i in range(math.ceil(length/(on+off))):
        s=i*(on+off)
        e=min(s+on,length)
        if s<length: line([(a[0]+ux*s,a[1]+uy*s),(a[0]+ux*e,a[1]+uy*e)],color,width)

# Orthographic plan view of the phantom. No marker positions are fabricated.
polygon([(O[0]-HALF,O[1]-HALF),(O[0]+HALF,O[1]-HALF),(O[0]+HALF,O[1]+HALF),(O[0]-HALF,O[1]+HALF)], '#f3e4bf', '#736b59', 3)
line([(O[0]-HALF,O[1]),(O[0]+HALF,O[1])],'#c9ba95',1.5)
line([(O[0],O[1]-HALF),(O[0],O[1]+HALF)],'#c9ba95',1.5)

cameras = [
    ((195,555),(180,665),'View 1'),
    ((495,170),(465,65),'View 2'),
    ((1260,215),(1355,120),'View 3'),
    ((1195,875),(1320,970),'View 4'),
]
checks=[]
for C,label,name in cameras:
    dx,dy=O[0]-C[0],O[1]-C[1]
    L=math.hypot(dx,dy)
    u=(dx/L,dy/L)
    v=(-u[1],u[0])
    def local(a,b): return (C[0]+a*u[0]+b*v[0],C[1]+a*u[1]+b*v[1])
    lens=local(46,0)
    # Square intersection of a center-to-camera ray, then stop arrow outside it.
    r=HALF/max(abs(u[0]),abs(u[1]))
    tip=(O[0]-(r+27)*u[0],O[1]-(r+27)*u[1])
    dash(lens,O)
    line([lens,tip],'#bd574b',4)
    back=(tip[0]-19*u[0],tip[1]-19*u[1])
    polygon([tip,(back[0]+8*v[0],back[1]+8*v[1]),(back[0]-8*v[0],back[1]-8*v[1])],'#bd574b',None)
    # Camera body long dimension is perpendicular to the optical axis.
    polygon([local(-27,-54),local(27,-54),local(27,54),local(-27,54)], '#fafbfc','#343a41',3)
    polygon([local(27,-18),local(46,-18),local(46,18),local(27,18)], '#e7ebee','#343a41',3)
    line([local(46,-18),local(46,18)],'#343a41',4)
    text(label[0],label[1],name,33,True)
    text(label[0],label[1]+40,'3 images',27)
    cross=(lens[0]-O[0])*u[1]-(lens[1]-O[1])*u[0]
    assert abs(cross)<1e-9
    assert (O[0]-lens[0])*u[0]+(O[1]-lens[1])*u[1]>0
    checks.append(f'{name}: axis passes through phantom center; residual={abs(cross):.2e}')

line([(O[0]-10,O[1]),(O[0]+10,O[1])],'#343a41',2)
line([(O[0],O[1]-10),(O[0],O[1]+10)],'#343a41',2)
text(780,720,'STI QC Phantom',31,True)
text(780,760,'Top view',25,color='#606973')
line([(590,850),(650,850)],'#bd574b',3)
text(790,850,'Optical axis',25,color='#606973')
text(780,1050,'One iPhone 15 Pro RGB camera · 4 views × 3 images = 12 images',29)
text(780,1100,'Schematic arrangement; angles and distances are not to scale.',23,color='#606973')
svg.append('</svg>')
(OUT/'CT-AR_acquisition_plan_v2.svg').write_text('\n'.join(svg))
im.save(OUT/'CT-AR_acquisition_plan_v2.png',dpi=(600,600))
(OUT/'CT-AR_acquisition_plan_v2_notes.txt').write_text(
    'New coordinate-based drawing, not an edit of the generated photograph.\n'
    'All positions are schematic and follow the qualitative arrangement supplied by the user.\n'
    'No actual camera azimuth, elevation or distance is asserted.\n\n'
    +'\n'.join(checks)+'\n\nSuggested caption:\n'
    'Schematic plan view of image acquisition. A single iPhone 15 Pro RGB camera was moved '
    'to four oblique viewpoints around the phantom. Three images were acquired at each viewpoint '
    '(12 images in total). Arrows indicate the projected optical axes; viewpoint angles '
    'and distances are schematic.\n')
print('\n'.join(checks))
print(OUT/'CT-AR_acquisition_plan_v2.svg')
print(OUT/'CT-AR_acquisition_plan_v2.png')
