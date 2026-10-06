"""Compare saved/reopened delivery renders at their stored PNG bit depth."""
import json,hashlib,datetime as dt
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v064_final_delivery';m=json.loads((out/'manifest.json').read_text());result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),render_job_complete=m['complete'],comparisons=[])
for tag,p in m['poses'].items():
    previous=base/'v059_camera_supports' if tag=='climb' else base/'v060_supports_common'/tag
    for r in p['renders']:
        name='cam'+r['camera']+'.png';old=previous/name;new=out/tag/name;a=cv2.imread(str(old),cv2.IMREAD_UNCHANGED);b=cv2.imread(str(new),cv2.IMREAD_UNCHANGED);assert a is not None and b is not None and a.shape==b.shape and a.dtype==b.dtype
        d=np.abs(a.astype(np.int32)-b.astype(np.int32));pixel=d[:,:,:3].max(axis=2);y,x=np.unravel_index(pixel.argmax(),pixel.shape);result['comparisons'].append(dict(pose=tag,camera=r['camera'],dtype=str(a.dtype),shape=list(a.shape),max_absolute_difference=int(d.max()),mean_absolute_difference=float(d.mean()),rgb_mean_difference_8bit_equivalent=float(d[:,:,:3].mean()/257),p99_pixel_difference_8bit_equivalent=float(np.quantile(pixel,.99)/257),pixels_over_1gray_8bit=int((pixel>257).sum()),pixels_over_5gray_8bit=int((pixel>1285).sum()),max_pixel_xy=[int(x),int(y)],different_values=int(np.count_nonzero(d)),pixel_identical=bool(np.array_equal(a,b))))
result['all_available_identical']=all(x['pixel_identical'] for x in result['comparisons']);result['twelve_views_verified']=m['complete'] and len(result['comparisons'])==12
plan=json.loads((base/'plan.json').read_text());result['original_source_unchanged']=hashlib.sha256(Path(plan['source_blend']).read_bytes()).hexdigest()==plan['source_sha256'];assert result['original_source_unchanged']
(out/'pixel_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
