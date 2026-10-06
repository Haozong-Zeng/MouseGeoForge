"""Scientific crops of the six boundary-tail labels in the 56-image return."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/annotation_review_20260914_real_images_56'
a=json.loads((OUT/'audit_56.json').read_text());source=Path(a['extract_root'])
font=lambda n:ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',n)
rows=[r for r in a['views'] if r['tail_tip_edge_distance_px'] is not None and r['tail_tip_edge_distance_px']<10]
assert len(rows)==6
im=Image.new('RGB',(1100,1390),'#17212c');d=ImageDraw.Draw(im)
d.text((20,15),'56张回标：6处 tail_tip 仍落在图像截断位置',font=font(27),fill='white')
d.text((20,60),'原图局部放大；红圈 = 14号标注，黄色 = 原图边界。未修改原始标签。',font=font(20),fill='#c7d3df')
for i,r in enumerate(rows):
    raw=Image.open(source/f"frame_{r['frame']:06d}"/(r['camera']+'.png')).convert('RGB')
    x,y=r['tail_tip'];w,h=raw.size
    x0=max(0,min(w-240,int(x)-120));y0=max(0,min(h-240,int(y)-120))
    crop=raw.crop((x0,y0,x0+240,y0+240)).resize((288,288))
    ox=(i%2)*550+100;oy=120+(i//2)*410
    d.text(((i%2)*550+20,oy),f"frame {r['frame']} · {r['camera']} · 距边界 {r['tail_tip_edge_distance_px']:.3f}px",font=font(22),fill='white')
    cy=oy+42;im.paste(crop,(ox,cy))
    if x0==0:d.line((ox,cy,ox,cy+288),fill='#ffce47',width=4)
    if x0+240==w:d.line((ox+287,cy,ox+287,cy+288),fill='#ffce47',width=4)
    if y0==0:d.line((ox,cy,ox+288,cy),fill='#ffce47',width=4)
    if y0+240==h:d.line((ox,cy+287,ox+288,cy+287),fill='#ffce47',width=4)
    px=ox+(x-x0)*1.2;py=cy+(y-y0)*1.2
    d.ellipse((px-10,py-10,px+10,py+10),outline='#ff6268',width=3)
    d.text(((i%2)*550+20,cy+300),f'14号坐标：({x:.3f}, {y:.3f})',font=font(20),fill='#ff969a')
d.text((20,1350),'只标可见点时，应省略画外真尾尖；不要把边界上的可见尾段末端改叫 tail_tip。',font=font(22),fill='#91e5af')
im.save(OUT/'tail_tip_boundary_flags.jpg',quality=96)
print(OUT/'tail_tip_boundary_flags.jpg')
