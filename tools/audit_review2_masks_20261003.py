import json,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_review_return2_20261003 import OUT,FONT,write
r=json.loads((OUT/'summary.json').read_text());old=next((OUT/'sent_package').iterdir());new=Path(r['extracted']);records=[];issues=[]
for row in r['files']:
    path=Path(row['annotation']);a=json.loads((old/path.relative_to(new)).read_text());b=json.loads(path.read_text())
    get=lambda ann:[(s['shape_type'],s['label'],s['points']) for s in ann['shapes'] if s['shape_type']!='point']
    aa,bb=get(a),get(b)
    for typ,label,coords in bb:
        q=np.array(coords)
        if typ!='polygon' or label!='mouse' or len(q)<3 or not np.isfinite(q).all() or (q<0).any() or (q[:,0]>=1080).any() or (q[:,1]>=1200).any():issues.append(dict(group=row['group'],camera=row['camera'],type=typ,label=label))
    if aa==bb:continue
    masks=[];panels=[]
    for shapes,label in [(aa,'返修前'),(bb,'返修后')]:
        mask=Image.new('1',(1080,1200));draw=ImageDraw.Draw(mask);im=Image.open(path.with_suffix('.png')).convert('RGB');pen=ImageDraw.Draw(im)
        for typ,_,coords in shapes:
            if typ=='polygon':draw.polygon([tuple(x) for x in coords],fill=1);pen.line([tuple(x) for x in coords]+[tuple(coords[0])],fill='#00ffff',width=3)
        masks.append(np.asarray(mask));im.thumbnail((450,500));p=Image.new('RGB',(460,540),'#18212d');p.paste(im,((460-im.width)//2,40));ImageDraw.Draw(p).text((8,8),f"{row['group']} cam{row['camera']} {label}",font=ImageFont.truetype(FONT,20),fill='white');panels.append(p)
    intersection=np.logical_and(*masks).sum();union=np.logical_or(*masks).sum();record=dict(group=row['group'],camera=row['camera'],mask_iou_before_after=float(intersection/max(union,1)),before_area=int(masks[0].sum()),after_area=int(masks[1].sum()))
    records.append(record);board=Image.new('RGB',(920,540));board.paste(panels[0],(0,0));board.paste(panels[1],(460,0));board.save(OUT/f"mask_{row['group']}_cam{row['camera']}.jpg",quality=95)
    if (row['group'],row['camera']) in [('R12','B'),('R07','B')]:
        raw=np.array(Image.open(path.with_suffix('.png')).convert('RGB'));added=masks[1]&~masks[0];removed=masks[0]&~masks[1]
        raw[added]=(.65*raw[added]+.35*np.array([255,50,150])).astype('uint8');raw[removed]=(.65*raw[removed]+.35*np.array([0,255,255])).astype('uint8')
        ys,xs=np.nonzero(masks[0]|masks[1]);box=(max(0,int(xs.min())-30),max(0,int(ys.min())-30),min(1080,int(xs.max())+31),min(1200,int(ys.max())+31))
        Image.fromarray(raw).crop(box).save(OUT/f"mask_change_detail_{row['group']}_cam{row['camera']}.png")
write(OUT/'mask_changes.json',dict(changed_images=len(records),issues=issues,records=records,caveat='IoU measures change, not improvement against unknown true mask. Coordinates checked; no claim of complete polygon topology validation.'))
print(json.dumps(dict(changed_images=len(records),issues=issues,largest_changes=sorted(records,key=lambda x:x['mask_iou_before_after'])[:6]),indent=2))
