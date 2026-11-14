"""Simple dimensioned plan-view *illustration* of included OBJ geometry; not a Blender render."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from blender.geometry import shapes
p=shapes(5.0,'diagonal',.06)
S=900
im=Image.new('RGB',(S,S),(235,234,230))
d=ImageDraw.Draw(im)
scale=145
ox=oy=450
xy=lambda pts:[(round(ox+x*scale),round(oy-y*scale)) for x,y in pts]
d.polygon(xy(p['square']),fill=(244,244,238),outline=(80,80,80),width=2)
for poly in p['strokes']:d.polygon(xy(poly),fill=(12,163,69))
# Center shows color overlap but stays at geometric origin.
d.ellipse((ox-4,oy-4,ox+4,oy+4),outline=(5,35,22),width=2)
d.text((30,30),'C-UASC green X — model plan view',fill=(33,43,38))
d.text((30,55),'Nominal 5.0 m diagonal (ASSUMPTION)',fill=(65,65,65))
d.text((30,75),f"Square side: {p['side_m']:.3f} m (derived)",fill=(65,65,65))
d.text((30,95),'Green stripe width: 0.06 m (ASSUMPTION)',fill=(65,65,65))
d.text((30,845),'Actual square dimension / stripe width not specified in 2026 rules',fill=(95,95,95))
im.save('/mnt/data/cuasc_green_x/examples/green_x_plan_view.png')
