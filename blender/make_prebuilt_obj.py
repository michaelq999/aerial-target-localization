"""Generate a true meter-scale OBJ+MTL asset without needing Blender installed."""
import json
import math
from pathlib import Path
from geometry import shapes,prism

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'model'
OUT.mkdir(exist_ok=True)
span=5.0
mode='diagonal'
width=.06
parts=shapes(span,mode,width)
items=[
 ('White_Target_Board',parts['square'],0.003,0.013,'White_Matte'),
 ('Green_Diagonal_A',parts['strokes'][0],0.0132,0.0143,'Green_Matte'),
 ('Green_Diagonal_B',parts['strokes'][1],0.0143,0.0154,'Green_Matte'),
]
mtl='''newmtl White_Matte
Ka 0.2 0.2 0.2
Kd 0.92 0.91 0.87
Ks 0 0 0
Ns 10
illum 2

newmtl Green_Matte
Ka 0.03 0.1 0.03
Kd 0.025 0.50 0.10
Ks 0 0 0
Ns 10
illum 2
'''
(OUT/'green_x.mtl').write_text(mtl)
lines=['# C-UASC 2026-inspired green X on white square. Units: meters.', '# 5-meter diagonal / 0.06-meter strokes are configurable ASSUMPTIONS.','mtllib green_x.mtl']
vbase=1
for name,poly,z0,z1,mat in items:
    vs,faces=prism(poly,z0,z1)
    lines+=['o '+name,'usemtl '+mat]
    lines += ['v {:.6f} {:.6f} {:.6f}'.format(*v) for v in vs]
    for face in faces:
        lines.append('f '+' '.join(str(vbase+j) for j in face))
    vbase+=len(vs)
(OUT/'green_x.obj').write_text('\n'.join(lines)+'\n')
spec={
 'source':'C-UASC 2026 official rules (Rev 3, 2026-02-23), p.13',
 'official':'Target approximately 5 meters in diameter with white background and green X.',
 'geometry_note':'Square side and stroke width not specified by competition. Change Blender generator arguments if clarified.',
 'span_m':span,'size_mode':mode,'square_side_m':parts['side_m'],
 'stroke_width_m_assumed':width,
 'board_thickness_m_assumed':0.010,
 'model_units':'meters',
 'target_center_world_m':[0,0,0.0154],
}
(OUT/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
print('Wrote',OUT/'green_x.obj','side_m:',parts['side_m'])
