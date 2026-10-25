"""CPU-only coverage and TRAIN input-sensitivity audits; no model selection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from mousegeoforge.models.calibrated_mouse_depth import CalibratedRayDepth
from mousegeoforge.semseg_training import digest, write_json

OUT=ROOT/'output/calibrated_mouse_10h_20261025'


def coverage():
    target=OUT/'coverage_audit_v1.json'
    if target.exists():raise FileExistsError(target)
    rows=json.loads((OUT/'semseg_items.json').read_text())['items']
    by_group=defaultdict(list); identities=defaultdict(lambda:defaultdict(list))
    for row in rows:
        path=ROOT/row['label'];assert digest(path)==row['label_sha256']
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as truth:label=truth['body'].astype('uint8')
        else:
            label=np.array(Image.open(path).convert('L'))
            if row.get('label_encoding')!='class_index_0_1_255':label=(label>0).astype('uint8')
        assert set(np.unique(label))<={0,1,255}
        mask=(label==1).astype('uint8');ys,xs=np.where(mask)
        n,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        large=int((stats[1:,cv2.CC_STAT_AREA]>=20).sum())
        item=dict(id=row['id'],empty=not len(xs),foreground_pixels=int(mask.sum()),
            unknown_pixels=int((label==255).sum()),foreground_fraction=float(mask.mean()),
            components_ge20px=large,components_are_not_instance_labels=True,
            boundary_touch=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
            bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None)
        key=f"{row['split']}/{row['domain']}/{'bottom' if row['bottom'] else 'other'}"
        by_group[key].append(item);identities[row['split']][row['image_sha256']].append(row['id'])
    overlap=sorted(set(identities['train'])&set(identities['dev']))
    summary={key:dict(images=len(items),empty=sum(x['empty'] for x in items),
        boundary_touch=sum(x['boundary_touch'] for x in items),
        median_foreground_fraction=float(np.median([x['foreground_fraction'] for x in items])),
        images_with_multiple_components_ge20px=sum(x['components_ge20px']>1 for x in items),
        unknown_pixels=sum(x['unknown_pixels'] for x in items)) for key,items in by_group.items()}
    write_json(target,dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)),items_sha256=digest(OUT/'semseg_items.json'),
        summary=summary,exact_train_dev_rgb_overlap_count=len(overlap),
        overlap=[dict(sha256=sha,train=identities['train'][sha],dev=identities['dev'][sha]) for sha in overlap],
        per_image=dict(by_group),
        instance_count_ground_truth_available=False,
        component_counts_do_not_establish_instance_counts=True,
        empty_background_fpr_evaluable=any(x['empty'] for key,items in by_group.items() if key.startswith('dev/') for x in items),
        no_dataset_or_training_changes=True))
    print(json.dumps(dict(summary=summary,exact_rgb_overlap=len(overlap)),indent=2))


def sensitivity():
    target=OUT/'depth_train_sensitivity_v1.json'
    if target.exists():raise FileExistsError(target)
    began=time.monotonic()
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    cache=json.loads((OUT/'cache_receipt.json').read_text())
    candidates=defaultdict(list)
    for i,row in enumerate(manifest['scenes']):
        if row['split']=='train':candidates[(row['source'],row['pose_frame'])].append(i)
    selected=[i for key in sorted(candidates) for i in candidates[key][:2]]
    results={}
    with torch.inference_mode():
        for arm in ['D1','D2']:
            checkpoint=OUT/f'depth_{arm}_6000_s17_v2/step_006000.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            for relative,pin in saved['protocol']['source_sha256'].items():assert digest(ROOT/relative)==pin
            model=CalibratedRayDepth(planes=saved['protocol']['planes']).eval().requires_grad_(False)
            model.load_state_dict(saved['ema'])
            bounds=torch.tensor(saved['protocol']['capture_aabb_mm'],dtype=torch.float32)
            rows=[]
            for index in selected:
                entry=cache['files'][index];assert digest(ROOT/entry['path'])==entry['sha256']
                item=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False)
                assert item['scene']['split']=='train'
                encoded=model.encode(item['features'].float());k=item['K'].float();e=item['E'].float()
                for reference in range(3):
                    ys,xs=np.where(item['body'][reference].numpy())
                    rng=np.random.default_rng(20261025+index*3+reference)
                    choice=rng.choice(len(xs),64,replace=len(xs)<64)
                    xy=torch.tensor(np.stack([xs[choice],ys[choice]],-1),dtype=torch.float32)
                    truth=item['depth'][reference,ys[choice],xs[choice]].float()
                    baseline,_,valid=model(encoded,xy,reference,k,e,bounds)
                    assert valid.all()
                    cases={}
                    for name in ['matching_disabled','source_camera_x_plus10mm']:
                        modified=e.clone()
                        if name=='source_camera_x_plus10mm':
                            for source in range(3):
                                if source!=reference:modified[source,0,3]+=10.
                        pred,_,again=model(encoded,xy,reference,k,modified,bounds,use_cost=name!='matching_disabled')
                        assert torch.equal(valid,again)
                        cases[name]=dict(mean_absolute_prediction_change_mm=float((pred-baseline).abs().mean()),
                            rmse_mm=float(((pred-truth)**2).mean().sqrt()))
                    rows.append(dict(scene=index,reference=reference,rays=len(xy),
                        baseline_rmse_mm=float(((baseline-truth)**2).mean().sqrt()),cases=cases))
            results[arm]=dict(checkpoint_sha256=digest(checkpoint),rows=rows,
                summary=dict(mean_probe_rmse_mm=float(np.mean([x['baseline_rmse_mm'] for x in rows])),
                    **{name:dict(mean_absolute_prediction_change_mm=float(np.mean([x['cases'][name]['mean_absolute_prediction_change_mm'] for x in rows])),
                        mean_probe_rmse_mm=float(np.mean([x['cases'][name]['rmse_mm'] for x in rows])))
                       for name in ['matching_disabled','source_camera_x_plus10mm']}))
    write_json(target,dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),arms=results,
        split='TRAIN only, 12 scenes x3views x64 fixed rays',device='CPU; no CUDA allocation',
        source_sha256=digest(Path(__file__)),seconds=time.monotonic()-began,
        primary_protocol_unchanged=True,dev_selection=False,
        interpretation='Input perturbation is out-of-distribution diagnostic, not a matched retrained control; v3 matching/reference pair remains required',
        no_geometric_generalization_claim=True))
    print(json.dumps({arm:result['summary'] for arm,result in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['coverage','sensitivity']);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    coverage() if args.command=='coverage' else sensitivity()
