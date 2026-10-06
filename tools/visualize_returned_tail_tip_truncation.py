"""Evidence figure from returned RGB/point coordinates; no label modification."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260914_real_development_corrected/real_development---ok'
OUT=ROOT/'output/annotation_review_20260914_real_development_corrected/tail_tip_boundary_examples.jpg'
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
def font(size):return ImageFont.truetype(FONT,size)


def main():
    sheet=Image.new('RGB',(1280,1540),'#17212c');d=ImageDraw.Draw(sheet)
    d.text((25,16),'尾尖检查：红圈是回标的 14 号点，黄线是真实图像边界',font=font(30),fill='white')
    d.text((25,64),'仅裁切放大原始 RGB；未生成或补画画外内容，未更改原标注。',font=font(21),fill='#c7d3df')
    entries=[(0,'A',(0,640,205,955),'left'),(0,'D',(165,970,510,1200),'bottom'),(64,'A',(0,640,205,955),'left')]
    for row,(frame,cam,box,side) in enumerate(entries):
        folder=SOURCE/f'frame_{frame:06d}'
        im=Image.open(folder/f'cam{cam}.png').convert('RGB')
        a=json.loads((folder/f'cam{cam}.json').read_text())
        xy=next(s['points'][0] for s in a['shapes'] if s['label']=='14tail_tip')
        top=115+row*440
        d.text((25,top),f'源帧 {frame} · cam{cam}',font=font(26),fill='white')
        # Original image overview, original coordinate limits unchanged.
        overview=im.resize((324,360));ox=25;oy=top+48;sheet.paste(overview,(ox,oy))
        s=.3
        d.rectangle((ox+box[0]*s,oy+box[1]*s,ox+box[2]*s-1,oy+box[3]*s-1),outline='#ffd256',width=2)
        d.text((375,top),'原始边界附近放大',font=font(24),fill='#c7d3df')
        crop=im.crop(box);scale=min(470/crop.width,(280 if side=='bottom' else 335)/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
        cx=450;cy=top+48;sheet.paste(crop,(cx,cy))
        if side=='left':
            d.line((cx,cy,cx,cy+crop.height),fill='#ffd256',width=3)
            d.text((365,cy+10),'画外',font=font(23),fill='#98a9ba')
            d.text((740,cy+235),'左边界：x = 0',font=font(23),fill='#ffd256')
        else:
            d.line((cx,cy+crop.height-1,cx+crop.width,cy+crop.height-1),fill='#ffd256',width=3)
            d.text((cx+10,cy+crop.height+8),'画外（图像到此结束）',font=font(21),fill='#98a9ba')
            d.text((950,cy+240),'底边界',font=font(23),fill='#ffd256')
        px=cx+(xy[0]-box[0])*scale;py=cy+(xy[1]-box[1])*scale
        d.ellipse((px-12,py-12,px+12,py+12),outline='#ff6268',width=3)
        tx=760 if side=='left' else 960;ty=cy+60
        d.text((tx,ty),'当前 14 号点',font=font(26),fill='#ff6268')
        d.text((tx,ty+40),f'({xy[0]:.3f}, {xy[1]:.2f})',font=font(20),fill='#dce4ec')
        distance=xy[0] if side=='left' else im.height-1-xy[1]
        d.text((tx,ty+76),f'距边界 {distance:.3f} px',font=font(20),fill='#dce4ec')
        d.line((tx-7,ty+18,px+12,py),fill='#ff6268',width=2)
        if side=='left':
            d.text((740,cy+285),'尾巴仍有宽度地穿过边界，',font=font(22),fill='white')
            d.text((740,cy+320),'图中没有显示真实末端。',font=font(22),fill='white')
        else:
            d.text((450,cy+330),'尾段延伸到画面外；不能把图像截断处当解剖尾尖。',font=font(21),fill='white')
        d.line((25,top+425,1255,top+425),fill='#455465',width=1)
    d.text((25,1465),'建议：只标可见点时，此处不标 tail_tip；若记录状态则用 out_of_frame。',font=font(25),fill='#91e5af')
    d.text((25,1505),'“未看到真尾尖”不代表尾巴断了；只是原始相机画面未包含它。',font=font(20),fill='#c7d3df')
    sheet.save(OUT,quality=96)
    print(OUT)


if __name__=='__main__':main()
