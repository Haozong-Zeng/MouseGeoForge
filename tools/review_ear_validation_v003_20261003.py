import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003';out=base/'validate_v003';manifest=json.loads((out/'manifest.json').read_text());assert manifest['complete']
old=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY/renders';refs=root/'output/pymousestudio_realism_cycle_20261002/references'
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',21)
sheet=Image.new('RGB',(1440,1320),'#182331');d=ImageDraw.Draw(sheet)
for row,pose in enumerate(['climb','quadruped','headraise']):
    board=Image.new('RGB',(1440,640),'#182331');draw=ImageDraw.Draw(board)
    for col,c in enumerate('ACEF'):
        roi=manifest['poses'][pose]['ear_rois'][c]
        for rr,(label,path) in enumerate([('OLD .12',old/pose/f'cam{c}.png'),('NEW .60',out/pose/f'cam{c}.png')]):
            im=Image.open(path).convert('RGB');crop=ImageOps.contain(im.crop(roi),(344,274));draw.text((col*360+8,rr*320+5),f'{pose} {c} {label}',font=font,fill='white');board.paste(crop,(col*360+(360-crop.width)//2,rr*320+38+(274-crop.height)//2))
        d.text((col*360+8,row*440+5),f'{pose} cam{c}',font=font,fill='white');sheet.paste(Image.open(out/pose/f'cam{c}.png').convert('RGB').resize((360,400)),(col*360,row*440+40))
    board.save(out/f'{pose}_ear_before_after.jpg',quality=97)
    group={'climb':'FR2/frame_000649','quadruped':'FR2/frame_001104','headraise':'AZR2L2/frame_000129'}[pose]
    realboard=Image.new('RGB',(1080,1280),'#182331');dd=ImageDraw.Draw(realboard)
    for col,c in enumerate('CF'):
        for rr,(label,path) in enumerate([('REAL',refs/group/f'cam{c}.png'),('NEW .60',out/pose/f'cam{c}.png')]):
            dd.text((col*540+10,rr*640+5),f'{pose} {c} {label}',font=font,fill='white');realboard.paste(Image.open(path).convert('RGB').resize((540,600)),(col*540,rr*640+40))
    realboard.save(out/f'{pose}_real_comparison.jpg',quality=97)
sheet.save(out/'all_pose_views.jpg',quality=97)
print(out)
