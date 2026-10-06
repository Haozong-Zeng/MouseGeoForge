"""Assemble scientific comparison sheets and fixed same-geometry luminance diagnostics."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v040_light_refinement';font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',20)
sheet=Image.new('RGB',(2160,1265),'#182331');draw=ImageDraw.Draw(sheet);stats={}
for col,(tag,label) in enumerate([('control','Control v033'),('a_small_sources','Small background sources'),('b_rebalance','Half key/rim;1.5x bottom'),('c_combined','Combined')]):
    draw.text((col*540+8,8),label,font=font,fill='white')
    for row,c in enumerate('FA'):
        p=(base/'v033_dof/b_radius0p6_camF.png' if c=='F' else base/'v033_dof/native128/camA_denoised.png') if tag=='control' else out/f'{tag}_cam{c}.png';im=Image.open(p).convert('RGB');sheet.paste(im.resize((540,600)),(col*540,35+row*615))
        if c=='F':
            a=np.asarray(im.convert('L'));stats[tag]=dict(same_geometry_fur_roi_median=float(np.median(a[570:800,730:835])),roi_xyxy=[730,570,835,800],clipped_pixels_whole_frame=int((a==255).sum()))
sheet.save(out/'comparison.jpg',quality=94);(out/'pixel_diagnostics.json').write_text(json.dumps(dict(note='Same geometry F-image ROI only. Not segmentation or realism score. Whole-frame clipping includes all objects/background.',values=stats),indent=2));print(json.dumps(stats,indent=2))
