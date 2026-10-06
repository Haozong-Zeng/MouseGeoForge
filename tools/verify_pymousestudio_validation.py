"""Verify original hashes, source split, native/rectified loader and review assets."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
from collections import defaultdict
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from import_pymousestudio_raw import sha256
from mousegeoforge.raw_validation import RawVideoValidation


def run():
    raw=ROOT/'dataset/PyMouseStudioRaw';out=ROOT/'output/pymousestudio_raw_delivery_v2'
    out.mkdir(exist_ok=False)
    imported=json.loads((raw/'_metadata/import.json').read_text());hash_count=0;video_hashes=defaultdict(list)
    for archive in imported['archives']:
        for file in archive['files']:
            assert sha256(raw/file['path'])==file['sha256'];hash_count+=1
            if '/videos/' in file['path']:video_hashes[file['sha256']].append(file['path'])
    duplicates=[v for v in video_hashes.values() if len(v)>1]
    assert not duplicates,'Duplicate source videos require source-group merge before validation'
    manifest=raw/'_validation_v2/sequences.jsonl';reader=RawVideoValidation(manifest,ROOT)
    groups=defaultdict(set)
    for r in reader.records.values():groups[r['split']].add(r['source_group'])
    assert not groups['real_development']&groups['real_locked_holdout']
    try:reader.read('pymousestudio_raw/UR2',0)
    except PermissionError:locked=True
    else:raise AssertionError('Holdout lock failed')
    checks=[]
    for cameras in (['A'],['D','E'],['D','E','A'],list('ABCDEFG')):
        x=reader.read('pymousestudio_raw/AZL3',780,cameras,undistort=True)
        assert x['images'].shape==(len(cameras),1200,1080,3)
        assert x['keypoints2d_source31'].shape==(len(cameras),1,31,2)
        assert not x['distortion'].any() and x['depth'] is None and not x['manual_ground_truth']
        if 'A' in cameras:
            a=cameras.index('A');assert not x['mask_valid'][a] and np.isnan(x['keypoints2d_source31'][a]).all()
        checks.append(dict(cameras=cameras,shape=list(x['images'].shape)))
    x=reader.read('pymousestudio_raw/AZR2L2',0,undistort=False);assert len(x['camera_names'])==5
    # Existing, bounded manifest/config/protocol audit; no full-history claim.
    pattern=re.compile(r'(?:AZL3|AZR2L2|AZR2|CR2|EL2L2|EL2|ER3L2|FR2|FR3|GR2|UR2|UR3)_20241122',re.I)
    paths=list((ROOT/'configs').rglob('*.json'))+list((ROOT/'configs').rglob('*.yaml'))
    paths+=list((ROOT/'docs').glob('*.md'))+list((ROOT/'output').glob('*/protocol.json'))+list((ROOT/'output').glob('*/manifest.json'))+list((ROOT/'output').glob('*/summary.json'))
    hits=[];scanned=[]
    for p in paths:
        if p.stat().st_size>15000000 or 'pymousestudio' in p.name.lower():continue
        matches=sorted(set(pattern.findall(p.read_text(errors='replace'))));scanned.append(str(p.relative_to(ROOT)))
        if matches:hits.append(dict(path=str(p.relative_to(ROOT)),matches=matches))
    tasks=[]
    for r in reader.records.values():
        for frame in r['diagnostic_frame_indices']:
            for camera in ('A','D','E'):
                if camera not in r['cameras']:continue
                tasks.append(dict(sequence_id=r['sequence_id'],source_group=r['source_group'],split=r['split'],
                    frame=frame,camera=camera,video=r['cameras'][camera]['video'],manual_review_status='pending',
                    required=['all visible animals and instance IDs','body-only and full-animal bbox definitions','visible/occluded/unannotated landmarks','foreground mask if feasible'],
                    model_predictions_shown=False))
    (out/'annotation_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    x=reader.read('pymousestudio_raw/AZL3',780,['B','D','F'],undistort=True)
    fig,axes=plt.subplots(1,3,figsize=(10,4),layout='constrained')
    for i,ax in enumerate(axes):
        ax.imshow(x['images'][i]);mask=x['foreground_mask_uint8'][i]>127
        overlay=np.zeros((*mask.shape,4));overlay[mask]=[0,1,0,.4];ax.imshow(overlay)
        ax.set_title(f'AZL3 {x["camera_names"][i]} existing mask');ax.axis('off')
    fig.savefig(out/'mask_review.png',dpi=120);plt.close(fig)
    report=dict(passed=True,verified_original_files=hash_count,archive_count=len(imported['archives']),duplicate_rgb_videos=duplicates,
        data_bytes=imported['uncompressed_bytes'],groups={k:sorted(v) for k,v in groups.items()},holdout_lock_passed=locked,
        loader_checks=checks,missing_camera_sequence_views=sum(7-len(r['cameras']) for r in reader.records.values()),prior_source_name_scan_files=scanned,prior_source_name_hits=hits,
        source_independence_scope='No exact project-date names in scanned previous configs/protocols; no proof about all historical training or upstream DLC/encoder pretraining.',
        annotation_tasks=len(tasks),manifest_sha256=sha256(manifest),
        schema='Validation input adapter, NOT existing Core23 training data. Missing/uncertain supervision explicitly absent.',
        code_sha256={p:sha256(ROOT/p) for p in ['mousegeoforge/raw_validation.py','tools/audit_pymousestudio_raw.py','tools/import_pymousestudio_raw.py','tools/verify_pymousestudio_validation.py']},
        model_inference_runs=0,training_runs=0)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('prior_source_name_scan_files','code_sha256')},indent=2))


if __name__=='__main__':run()
