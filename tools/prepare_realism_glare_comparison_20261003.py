"""Scientific optical-ablation and backdrop comparison sheets, no image retouching."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';diag=base/'v043_glare_diagnostic';out=base/'v044_backdrop';font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',17)
tags=[('control','Control'),('a_no_cylinder','No cylinder'),('b_no_floor','No floor'),('c_no_caustics','No caustics'),('d_pinhole','Pinhole'),('e_pinhole_small_sources','Pinhole/small lamps')];sheet=Image.new('RGB',(2160,860),'#182331');draw=ImageDraw.Draw(sheet)
for col,(tag,title) in enumerate(tags):
    draw.text((col*360+8,8),title,font=font,fill='white')
    for row,c in enumerate('FA'):
        p=(base/'v040_light_refinement/b_rebalance_camF.png' if c=='F' else base/'v040_light_refinement/native128/camA_denoised.png') if tag=='control' else diag/f'{tag}_cam{c}.png';sheet.paste(Image.open(p).convert('RGB').resize((360,400)),(col*360,36+row*412))
sheet.save(diag/'comparison.jpg',quality=94)
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',21);sheet=Image.new('RGB',(1620,1280),'#182331');draw=ImageDraw.Draw(sheet);stats={}
for col,(tag,title) in enumerate([('control','Original linked roughness'),('a_diffuse','Diffuse diagnostic'),('b_matte_principled','Matte Principled candidate')]):
    draw.text((col*540+8,8),title,font=font,fill='white')
    for row,c in enumerate('FA'):
        p=(base/'v040_light_refinement/b_rebalance_camF.png' if c=='F' else base/'v040_light_refinement/native128/camA_denoised.png') if tag=='control' else out/f'{tag}_cam{c}.png';im=Image.open(p).convert('RGB');sheet.paste(im.resize((540,600)),(col*540,40+row*620))
        if c=='F':
            a=np.asarray(im.convert('L'));stats[tag]=dict(clipped255_pixels=int((a==255).sum()),bright250_pixels=int((a>=250).sum()),same_fur_roi_median=float(np.median(a[570:800,730:835])))
sheet.save(out/'comparison.jpg',quality=94);(out/'pixel_diagnostics.json').write_text(json.dumps(dict(note='Whole-frame saturation and same-geometry fur ROI only, not a realism metric.',values=stats),indent=2));print(json.dumps(stats))
