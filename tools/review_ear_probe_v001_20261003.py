import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_ear_cycle_20261003';out=base/'probe_v001'
a=json.loads((base/'audit_baseline/audit.json').read_text());m=json.loads((out/'manifest.json').read_text());assert m['complete']
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18)
realroot=root/'output/pymousestudio_realism_cycle_20261002/references/FR2/frame_000649'
realboxes={'C':[255,242,399,345],'F':[773,266,980,395]}
for c in 'CF':
    sources=[('REAL (different pose)',realroot/f'cam{c}.png',realboxes[c]),('baseline gray .12',base/'audit_baseline'/f'cam{c}.png',a['ear_rois'][c])]+[(tag,out/f'{tag}_cam{c}.png',a['ear_rois'][c]) for tag in m['candidates']]
    board=Image.new('RGB',(280*len(sources),340),'#182331');draw=ImageDraw.Draw(board)
    for col,(label,path,roi) in enumerate(sources):
        crop=Image.open(path).convert('RGB').crop(roi);crop=ImageOps.contain(crop,(264,284))
        draw.text((col*280+8,8),label,font=font,fill='white');board.paste(crop,(col*280+(280-crop.width)//2,45+(284-crop.height)//2))
    board.save(out/f'ear_compare_cam{c}.jpg',quality=97)
export=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003');export.mkdir(exist_ok=True)
for c in 'CF':shutil.copy2(out/f'ear_compare_cam{c}.jpg',export/f'ear_compare_cam{c}.jpg')
(export/'index.html').write_text('<meta charset="utf-8"><title>耳部材质试验 · 首轮</title><style>body{font:18px/1.7 sans-serif;background:#182331;color:white;padding:24px}img{max-width:100%}</style><h1>耳部材质专项：首轮对照</h1><p>仅耳材质改变，原分辨率1080×1200、128采样OptiX。真实参考与合成姿态不同，局部裁切缩放仅用于外观判断，不是像素配准评分。左起：真实、基线、灰度.30、灰度.60、柔和高光、30%薄组织透光、次表面散射。</p><h2>C侧面</h2><img src="ear_compare_camC.jpg"><h2>F侧面</h2><img src="ear_compare_camF.jpg"><p>仍在实验，尚未选定最终材质。</p>',encoding='utf-8')
print(export)
