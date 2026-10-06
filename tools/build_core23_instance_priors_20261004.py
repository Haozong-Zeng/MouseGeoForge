"""Prediction-only instance priors, with exact original full-readout check."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import torch
from torch import nn
from torch.nn import functional as F
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
OLD=ROOT/'output/surface_curve_cycle_20261003'
OUT=CYCLE/'instance_priors_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_prediction_only_instance_priors',active_pid=os.getpid(),active_job='instance_priors_v1');write_json(CYCLE/'state.json',state)
    model=SurfaceCurveDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval().requires_grad_(False)
    initial=OLD/'training_v1/pilot/initial_heads.pt';model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=True)
    manifest=json.loads((CYCLE/'features_v1/manifest.json').read_text());boxes=[];masks=[];audit=[];maxdelta=0.
    for n,r in enumerate(manifest['records']):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        assert fingerprint(ROOT/r['path'])==r['sha256']
        saved=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)
        item=torch.load(OLD/f'training_v1/pilot/cache/{r["key"]}.pt',map_location='cpu',weights_only=False)
        assert all(fingerprint(ROOT/p)==sha for p,sha in r['source_hashes'].items())
        with torch.no_grad():pred=model.forward_cached(item['batch']['images'].cuda(),[None if x is None else x.cuda() for x in item['cached']],item['patch_start'],coupling='off')
        for o in saved['observations']:
            m=o['metadata'];v,q=m['view_index'],m['query'];xy=pred['instance_keypoint_xy'][0,v,q].cpu()
            delta=float((xy-o['original_xy']).abs().max());maxdelta=max(maxdelta,delta);assert delta<2e-4
            box=pred['instance_boxes_cxcywh'][0,v,q].cpu();boxes.append(box)
            masks.append(F.interpolate(pred['instance_mask_logits'][0,v,q][None,None],size=(48,48),mode='bilinear',align_corners=False)[0,0].cpu().half())
            valid=o['valid'].clone()
            if m['instance_count']>1:valid[7:14]=False
            inside=((o['gt_xy']-box[:2]).abs()<=box[2:]*.625).all(-1)
            audit.append(dict(kind=item['record']['kind'],role=item['record']['role'],instance_count=m['instance_count'],valid=int(valid.sum()),inside_expanded_predicted_box=int((valid&inside).sum()),box_iou=m['box_iou']))
        write_json(OUT/'status.json',dict(stage='extracting',pid=os.getpid(),completed=n+1,total=len(manifest['records'])))
    path=OUT/'inputs.pt';torch.save(dict(boxes_cxcywh=torch.stack(boxes),mask_logits48=torch.stack(masks)),path)
    write_json(OUT/'manifest.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),source_manifest_sha256=fingerprint(CYCLE/'features_v1/manifest.json'),max_original_prediction_delta=maxdelta,observations=len(boxes),audit=audit,semantics='Frozen predicted boxes/masks for original query. GT used only to audit containment, never form forward priors. Query association remains evaluation assisted.'))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid(),observations=len(boxes)))
    state.update(stage='instance_priors_ready',active_pid=None,active_job=None,next='Run prediction-only soft expanded-box gate controls with same original tokens/data/seeds/steps. Inspect poor real-box containment; no hard GT masking.');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
