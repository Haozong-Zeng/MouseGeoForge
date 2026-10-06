"""Regenerate all-view RGB/curve diagnostic sheets, no training promotion."""
import json,hashlib,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/surface_curve_cycle_20261003/render_staging';OUT=BASE.parent/'visual_review';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
manifest=json.loads((BASE/'manifest.json').read_text());assert len(manifest['poses'])==3
mask_mode='--masks' in sys.argv
rows=[]
for pose in manifest['poses']:
    assert len(pose['views'])==7
    sheet=Image.new('RGB',(4*500,2*500),'#142030')
    for i,v in enumerate(pose['views']):
        path=BASE/v['rgb'];im=Image.open(path).convert('RGB');assert im.size==(1080,1200)
        raw=im.copy();draw=ImageDraw.Draw(im);uv=np.array(v['keypoint_uv']);curve=np.array(v['curve_uv'])
        # Diagnostic projections include occluded points, explicitly not visibility labels.
        draw.line([tuple(p) for p in curve],fill='#ff9c20',width=4)
        for k in [5,6,14,15,16,17,22]:
            x,y=uv[k];draw.ellipse((x-7,y-7,x+7,y+7),outline='#ff40e0',width=3);draw.text((x+8,y),str(k+1),fill='white',font=font,stroke_fill='black',stroke_width=2)
        if mask_mode:im=Image.open(BASE/f"frame_{pose['frame_index']:06d}"/'annotations'/f"{i:08d}_mask_curve.png").convert('RGB')
        xy=np.r_[uv,curve];lo=np.maximum(xy.min(0)-65,0);hi=np.minimum(xy.max(0)+65,[1080,1200]);box=tuple(np.r_[lo,hi].astype(int))
        im=im.crop(box);im.thumbnail((490,440));panel=Image.new('RGB',(500,500),'#142030');panel.paste(im,((500-im.width)//2,55));d=ImageDraw.Draw(panel);d.text((8,8),pose['pose']+' cam'+v['camera'],font=font,fill='white');d.text((8,30),'Skin mask + visible surface line' if mask_mode else 'Projected surface line (incl. occluded)',font=font,fill='#ffba66')
        sheet.paste(panel,((i%4)*500,(i//4)*500))
        raw.thumbnail((540,600));raw.save(OUT/f"{pose['pose']}_cam{v['camera']}_raw.jpg",quality=94)
        rows.append(dict(pose=pose['pose'],camera=v['camera'],image=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=[1080,1200],projection_max_error_px=v['projection_max_error_px']))
    sheet.save(OUT/(pose['pose']+('_mask_visible_curve.jpg' if mask_mode else '_curve_projections.jpg')),quality=95)
(OUT/'rgb_verification.json').write_text(json.dumps(dict(images=len(rows),all_dimensions_verified=True,visibility_not_yet_verified=True,rows=rows),indent=2))
print('RGB_QA_ARTIFACTS',len(rows))
