"""Read-only pilot dataset verification plus an input-label preview."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'
plan=json.loads((OUT/'plan.json').read_text());manifest=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
project=Path(manifest['project_config']).parent
df=pd.read_hdf(project/'labeled-data/camA/CollectedData_manualv3.h5')
excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
assert len(df)==1336 and len(manifest['train_indices'])==1043 and len(manifest['validation_indices'])==293
assert set(manifest['train_indices']).isdisjoint(manifest['validation_indices'])
assert sorted(manifest['train_indices']+manifest['validation_indices'])==list(range(1336))
for i,e in enumerate(manifest['entries']):
    assert tuple(e['index'])==tuple(df.index[i])
    assert (e['session'],e['frame']) not in excluded
    assert e['session'] not in plan['locked_holdout_sessions']
    assert e['session'] in plan['training_sessions' if e['role']=='train' else 'development_validation_sessions']
    assert hashlib.sha256((ROOT/e['source']).read_bytes()).hexdigest()==e['source_sha256']
    np.testing.assert_equal(np.isfinite(df.iloc[i].values.reshape(7,2)).all(-1),e['xy_observed'])
    with Image.open(ROOT/e['derived']) as im:assert im.size==(704,768);im.verify()
choices=[]
for role,cam in [('train','A'),('train','F'),('validation','A'),('validation','F')]:
    choices.append(next(i for i,e in enumerate(manifest['entries']) if e['role']==role and e['camera']==cam))
preview=Image.new('RGB',(1408,1600),'#17212c')
for j,i in enumerate(choices):
    e=manifest['entries'][i];im=Image.open(ROOT/e['derived']).convert('RGB');d=ImageDraw.Draw(im)
    for k,xy in zip(manifest['ids_1based'],df.iloc[i].values.reshape(7,2)):
        if not np.isfinite(xy).all():continue
        x,y=xy;d.ellipse((x-3,y-3,x+3,y+3),outline='#6fff70',width=2);d.text((x+5,y-12),str(k),fill='#6fff70',stroke_width=1,stroke_fill='black')
    ox=(j%2)*704;oy=(j//2)*800
    ImageDraw.Draw(preview).text((ox+12,oy+10),f"{e['role']} {e['session']} {e['frame']} cam{e['camera']}",fill='white')
    preview.paste(im,(ox,oy+32))
preview.save(OUT/'dataset_input_preview.jpg',quality=95)
result=dict(status='passed',train_images=1043,development_validation_images=293,
            source_hashes_verified=1336,source_separated=True,pending_groups_excluded=24,
            locked_holdout_read=False,nan_mask_preserved=True,input_wh=[704,768])
(OUT/'dataset_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
