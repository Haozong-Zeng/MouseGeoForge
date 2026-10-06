"""Fixed-coordinate scientific crop comparison; original render files untouched."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v051_closed_mouth';font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',22)
for cam,box in [('A',(430,890,730,1080)),('B',(190,740,480,990))]:
    width,height=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(width*6,height*2+40),'#182331');d=ImageDraw.Draw(sheet)
    for i,(name,p) in enumerate([('Control',base/'v046_pose_appearance/quadruped_r1'/f'cam{cam}.png'),('Half lift',out/f'a_half_cam{cam}.png'),('Full lift',out/f'b_full_cam{cam}.png')]):
        im=Image.open(p).convert('RGB').crop(box).resize((width*2,height*2));sheet.paste(im,(i*width*2,40));d.text((i*width*2+8,8),name,font=font,fill='white')
    sheet.save(out/f'comparison_{cam}.jpg',quality=97)
