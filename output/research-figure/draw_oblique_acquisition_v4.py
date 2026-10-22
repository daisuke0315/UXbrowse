"""Orthographic projection of a 3-D acquisition schematic, not measured poses."""
from pathlib import Path
from math import sqrt, sin, cos, pi
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
W,H,S=1800,1500,3
im=Image.new('RGB',(W*S,H*S),'white'); dr=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>']
F=Path('/System/Library/Fonts/Supplemental')
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def mul(a,k):return tuple(x*k for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def unit(a):return mul(a,1/sqrt(dot(a,a)))
def mean(ps):return tuple(sum(p[i] for p in ps)/len(ps) for i in range(3))
def pix(ps):return [(round(x*S),round(y*S)) for x,y in ps]
def poly(ps,fill,stroke='#454b52',width=2):
    dr.polygon(pix(ps),fill=fill)
    if stroke:dr.line(pix(ps+[ps[0]]),fill=stroke,width=round(width*S),joint='curve')
    svg.append(f'<polygon points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in ps)}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}" stroke-linejoin="round"/>')
def line(ps,color='#454b52',width=2):
    dr.line(pix(ps),fill=color,width=round(width*S))
    svg.append(f'<polyline points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in ps)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')
def text(x,y,s,size=31,bold=False,color='#252c33'):
    font=ImageFont.truetype(str(F/('Arial Bold.ttf' if bold else 'Arial.ttf')),round(size*S))
    dr.text((x*S,y*S),s,font=font,anchor='mm',fill=color)
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(s)}</text>')

TARGET=(0,0,.8)
class Scene:
    def __init__(self,eye,origin,scale):
        self.e=unit(eye);self.r=unit(cross((0,0,1),self.e));self.up=cross(self.e,self.r)
        self.o=origin;self.scale=scale;self.items=[]
    def proj(self,p):
        q=sub(p,TARGET)
        return (self.o[0]+self.scale*dot(q,self.r),self.o[1]-self.scale*dot(q,self.up))
    def face(self,ps,color,stroke='#555f67',width=1.7):
        self.items.append((dot(mean(ps),self.e),0,ps,color,stroke,width))
    def segment(self,a,b,color='#555f67',width=1.5):
        self.items.append((dot(mul(add(a,b),.5),self.e)+1e-5,1,[a,b],None,color,width))
    def render(self):
        for _,kind,ps,fill,stroke,width in sorted(self.items,key=lambda x:x[0]):
            points=[self.proj(p) for p in ps]
            if kind:line(points,stroke,width)
            else:poly(points,fill,stroke,width)

def cube(scene,mat=True):
    if mat:scene.face([(-1.16,-1.16,-.035),(1.16,-1.16,-.035),(1.16,1.16,-.035),(-1.16,1.16,-.035)],'#42484d','#42484d',1)
    a=.8
    faces=[
        ((0,0,1),[(-a,-a,1.6),(a,-a,1.6),(a,a,1.6),(-a,a,1.6)],'#f5e7c4'),
        ((0,-1,0),[(-a,-a,0),(a,-a,0),(a,-a,1.6),(-a,-a,1.6)],'#e6c78f'),
        ((1,0,0),[(a,-a,0),(a,a,0),(a,a,1.6),(a,-a,1.6)],'#edd9ad'),
        ((0,1,0),[(a,a,0),(-a,a,0),(-a,a,1.6),(a,a,1.6)],'#d7b982'),
        ((-1,0,0),[(-a,a,0),(-a,-a,0),(-a,-a,1.6),(-a,a,1.6)],'#ead2a3'),
    ]
    visible=[]
    for n,ps,color in faces:
        if dot(n,scene.e)>0:
            scene.face(ps,color,'#766951',2)
            # Surface seams suggest construction, without fabricating marker measurements.
            u=sub(ps[1],ps[0]);v=sub(ps[3],ps[0]);offset=mul(n,.001)
            scene.segment(add(add(ps[0],mul(u,.5)),offset),add(add(ps[3],mul(u,.5)),offset),'#b5a481',1.05)
            scene.segment(add(add(ps[0],mul(v,.5)),offset),add(add(ps[1],mul(v,.5)),offset),'#b5a481',1.05)
            visible.append(n)
    assert len(visible)==3,visible

def phone(scene,C):
    u=unit(sub(TARGET,C));v=unit(cross(u,(0,0,1)));w=cross(v,u)
    B=add(C,add(mul(v,.19),mul(w,-.37)))
    def p(x,y,z=0):return add(B,add(add(mul(v,x),mul(w,y)),mul(u,z)))
    back=[p(-.36,-.63),p(.36,-.63),p(.36,.63),p(-.36,.63)]
    front=[p(-.36,-.63,-.09),p(.36,-.63,-.09),p(.36,.63,-.09),p(-.36,.63,-.09)]
    scene.face(back,'#727a80','#373e44',1.7)
    scene.face(front,'#202a32','#373e44',1.7)
    for i in range(4):
        j=(i+1)%4
        scene.face([back[i],back[j],front[j],front[i]],'#a8afb4','#454b52',1.2)
    if dot(u,scene.e)>0:
        scene.face([p(-.31,.14,.004),p(.08,.14,.004),p(.08,.55,.004),p(-.31,.55,.004)],'#949b9f','#404950',.9)
        for x,y in [(-.2,.39),(-.01,.39),(-.105,.22)]:
            circle=[p(x+.069*cos(t*2*pi/24),y+.069*sin(t*2*pi/24),.008) for t in range(24)]
            scene.face(circle,'#172834','#bac3c9',.8)
    else:
        # The imaging lens is correctly hidden on the far side for near cameras.
        scene.face([p(-.30,-.55,-.095),p(.30,-.55,-.095),p(.30,.55,-.095),p(-.30,.55,-.095)],'#eaf0f3','#4b5963',1)
        scene.segment(p(-.10,.49,-.096),p(.10,.49,-.096),'#7d8992',2)
    return u

scene=Scene((3,-10,7),(900,630),130)
cube(scene)
positions=[(-4,-4,2.4),(-4,4,2.4),(4,4,2.4),(4,-4,2.4)]
labels=[(120,840),(330,75),(1640,120),(1450,900)]
checks=[]
for i,C in enumerate(positions,1):
    u=phone(scene,C)
    start=add(C,mul(u,.05))
    ray=sub(C,TARGET)
    # Ray-box intersection; stop the arrow in free space before the phantom.
    t=min(.8/abs(ray[k]) for k in range(3) if abs(ray[k])>1e-12)
    surface=add(TARGET,mul(ray,t))
    tip=add(surface,mul(unit(ray),.16))
    for k in range(45):
        a=add(start,mul(sub(tip,start),k/45));b=add(start,mul(sub(tip,start),(k+1)/45))
        scene.segment(a,b,'#bc5547',3.6)
    # A triangle in a camera-observer plane remains legible in projection.
    base=sub(tip,mul(u,.21));v=unit(cross(u,scene.e))
    scene.face([tip,add(base,mul(v,.080)),sub(base,mul(v,.080))],'#bc5547',None,0)
    assert C[2]>1.6 and abs(C[0])>.8 and abs(C[1])>.8
    residual=sqrt(dot(cross(sub(TARGET,C),u),cross(sub(TARGET,C),u)))
    assert residual<1e-10
    checks.append(f'View {i}: elevated corner viewpoint; three visible faces; axis residual {residual:.2e}')
scene.render()
for i,(x,y) in enumerate(labels,1):
    text(x,y,f'View {i}',34,True);text(x,y+44,'3 images',29)
text(900,865,'STI QC Phantom',35,True)
line([(130,1040),(1670,1040)],'#dde1e4',1.3)
text(900,1090,'Each camera view: top + two adjacent side faces',32,True)
for i,(C,x) in enumerate(zip(positions,[270,690,1110,1530]),1):
    view=Scene(sub(C,TARGET),(x,1235),78)
    cube(view,False);view.render()
    text(x,1370,f'View {i}',30,True)
text(900,1440,'One iPhone 15 Pro RGB camera · 4 viewpoints × 3 images = 12 images',29)
svg.append('</svg>')
(OUT/'CT-AR_oblique_acquisition_v4.svg').write_text('\n'.join(svg))
im.save(OUT/'CT-AR_oblique_acquisition_v4.png',dpi=(600,600))
(OUT/'CT-AR_oblique_acquisition_v4_notes.txt').write_text(
    'Coordinate-projected 3-D scientific schematic. No experimental image was modified.\n'
    'Positions and distances illustrate the described elevated corner acquisition, not measured camera extrinsics.\n'
    'Each phone imaging plane is perpendicular to its inward/downward optical axis.\n'
    'Near phones correctly show their screens because their imaging lenses face away from the illustration observer.\n'
    'The lower four cube projections use the same corresponding camera directions.\n\n'
    +'\n'.join(checks)+'\n\nCaption:\n'
    'Schematic of image acquisition from four elevated oblique viewpoints around the phantom. '
    'At each viewpoint, the top face and two adjacent side faces were visible. A single iPhone 15 Pro RGB camera '
    'was moved sequentially between viewpoints, and three images were acquired at each viewpoint (12 in total). '
    'The lower panels illustrate the corresponding viewing directions. Camera positions, distances, and proportions are schematic.\n')
print('\n'.join(checks))
