import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003';out=base/'probe_v002'
a=json.loads((base/'audit_baseline/audit.json').read_text());m=json.loads((out/'manifest.json').read_text());assert m['complete']
m['hypothesis']='Independent gray.45/.75 bracket around v001 gray.60; gray.60+SSS.20/scale.35/neutral radius and gray.60+15percent diffuse-translucent controls are compared with v001 gray.60.'
(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
realroot=root/'output/pymousestudio_realism_cycle_20261002/references/FR2/frame_000649';realboxes={'C':[255,242,399,345],'F':[773,266,980,395]}
export=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003')
for c in 'CF':
    sources=[('REAL (different pose)',realroot/f'cam{c}.png',realboxes[c]),('baseline gray .12',base/'audit_baseline'/f'cam{c}.png',a['ear_rois'][c]),('selected .60 (v001)',base/'probe_v001'/f'b_albedo060_cam{c}.png',a['ear_rois'][c])]+[(tag,out/f'{tag}_cam{c}.png',a['ear_rois'][c]) for tag in m['candidates']]
    board=Image.new('RGB',(280*len(sources),340),'#182331');draw=ImageDraw.Draw(board)
    for col,(label,path,roi) in enumerate(sources):
        crop=ImageOps.contain(Image.open(path).convert('RGB').crop(roi),(264,284));draw.text((col*280+8,8),label,font=font,fill='white');board.paste(crop,(col*280+(280-crop.width)//2,45+(284-crop.height)//2))
    board.save(out/f'ear_refine_cam{c}.jpg',quality=97);shutil.copy2(out/f'ear_refine_cam{c}.jpg',export/f'ear_refine_cam{c}.jpg')
print(out)
