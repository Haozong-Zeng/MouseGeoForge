import json
import shutil
import datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
m=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
with np.load(OUT/'lp_seed17/predictions_source_pixels.npz') as z:p=dict(z)
dest=OUT/'lp_seed17/review';dest.mkdir(exist_ok=True)
print('pred medians',np.nanmedian(p['uv'],axis=0),'std',p['uv'].std(axis=0),'confidence',np.median(p['confidence'],axis=0))
font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',22)
for session,frame,cam in [('FR2',1299,'A'),('AZR2L2',259,'E'),('FR2',909,'F')]:
    i=next(i for i,e in enumerate(m['entries']) if (e['session'],e['frame'],e['camera'])==(session,frame,cam))
    im=Image.open(ROOT/m['entries'][i]['source']).convert('RGB');d=ImageDraw.Draw(im)
    for j,k in enumerate(m['ids_1based']):
        x,y=p['gt'][i,j];u,v=p['uv'][i,j]
        if np.isfinite([x,y]).all():
            d.ellipse((x-7,y-7,x+7,y+7),outline='lime',width=3);d.text((x+8,y),str(k),fill='lime',font=font)
            d.line((x,y,u,v),fill='magenta',width=2)
        if np.isfinite([u,v]).all():
            d.ellipse((u-7,v-7,u+7,v+7),outline='magenta',width=3);d.text((u+8,v),str(k),fill='magenta',font=font)
    path=dest/f'{session}_{frame:06d}_{cam}.jpg';im.save(path,quality=94);print(path)
report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),run='lp_seed17',
    observations='Nose and both dorsal channels predict nearly constant image center with very low confidence; visible in reviewed images. Paw channels track animals but retain semantic outliers.',
    prediction_median=np.median(p['uv'],axis=0).tolist(),prediction_std=p['uv'].std(axis=0).tolist(),
    confidence_median=np.median(p['confidence'],axis=0).tolist(),ids=m['ids_1based'],
    validation=json.loads((OUT/'lp_seed17/summary.json').read_text()),
    checks='LP dataset names, all coordinates, NaN masks and target peak locations verified; missing-label gradients are zero. No label remapping applied.',
    decision='Preserve failed MSE baseline. Run same seed, architecture, cohort, optimizer and 30 epochs with KL heatmap loss only; test partial heatmap-collapse hypothesis.',
    next='Evaluate lp_seed17_kl in source pixels, inspect all channels and compare to DLC70; do not ensemble flat predictions or export pseudo-GT.',
    pseudo_labels_released=False,locked_holdout_opened=False)
(OUT/'heartbeat_20261001_2259_decision.json').write_text(json.dumps(report,indent=2))
(dest/'README.txt').write_text('Green: accepted human labels. Magenta: LP MSE predictions.\nNose/back channels collapsed near image center; this is a failed pilot, NOT pseudo-GT.\nTarget coordinate order and target peaks verified; KL-only control launched.\n')
export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/lp_mse_failure_review')
shutil.copytree(dest,export,dirs_exist_ok=True)
shutil.copy2(OUT/'heartbeat_20261001_2259_decision.json',export/'failure_diagnosis.json')
