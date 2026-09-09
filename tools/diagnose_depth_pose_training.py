"""Read-only diagnostics for poor pilot pose accuracy; no checkpoint updates.

GT-keypoint-assisted matching is intentionally ORACLE DIAGNOSTIC ONLY, never
used as the headline evaluation or for selecting a model.
"""
import argparse
import json
from pathlib import Path
import sys
import math
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,load_batch,
    restore,cached_forward,write_json,instance_targets_from_batch,instance_pose_loss,InstancePoseLossWeights)
from training.instance_pose_losses import hungarian_match_instances
from mousegeoforge.pilot_metrics import evaluate_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    state=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    spatial.load_state_dict({k:v for k,v in state.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    del state
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval(); del spatial
    manifest=json.loads((a.output/'manifest.json').read_text())
    groups={}
    for r in manifest['records']: groups.setdefault((r['split'],r['source_group']),[]).append(r)
    chosen=[rs[i] for rs in groups.values() for i in np.linspace(0,len(rs)-1,4).round().astype(int)]
    data=[]
    for r in chosen:
        batch=load_batch(r)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached,start=model.aggregator(batch['images'].cuda())
        data.append(dict(record=r,batch=batch,patch_start=start,
                         cached=[x.cpu() if x is not None and i in (4,11,17,23) else None for i,x in enumerate(cached)]))
    records=[]
    for mode in ('off','local_depth_features'):
        state=torch.load(a.output/f'seed{a.seed}_{mode}/final_heads.pt',map_location='cpu',weights_only=True)
        restore(model,state['model']); del state
        with torch.no_grad():
            for item in data:
                batch,out=cached_forward(model,item,mode)
                target=instance_targets_from_batch(batch)
                primary,box_matches=evaluate_scene(out,batch,target)
                matches=hungarian_match_instances(out,target)[0]
                errors=[]; inside=[]; maxima=[]; entropies=[]; agreements=[]
                h,w=batch['images'].shape[-2:]; scale=out['depth'].new_tensor([w,h])
                for v,(ps,ts) in enumerate(matches):
                    bbox_map={t:q for q,t in zip(*box_matches[v])}
                    for q,t in zip(ps.tolist(),ts.tolist()):
                        agreements.append(float(bbox_map[t]==q))
                        valid=target['keypoint_valid'][0,v,t]
                        gt=target['keypoint_xy'][0,v,t]
                        pred=out['instance_keypoint_xy'][0,v,q]
                        coarse=out['instance_keypoint_coarse_xy'][0,v,q]
                        diag=(target['boxes_cxcywh'][0,v,t,2:]*scale).norm().clamp_min(1.)
                        errors.extend((((pred-gt)*scale).norm(dim=-1)/diag)[valid].cpu().tolist())
                        inside.extend(((gt-coarse).abs()<=.1).all(-1)[valid].float().cpu().tolist())
                        probs=out['instance_keypoint_local_heatmap_logits'][0,v,q].flatten(-2).softmax(-1)[valid]
                        maxima.extend(probs.max(-1).values.cpu().tolist())
                        entropies.extend((-(probs*probs.clamp_min(1e-8).log()).sum(-1)/math.log(probs.shape[-1])).cpu().tolist())
                _,logs=instance_pose_loss(out,target,InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.),image_size=(h,w))
                records.append(dict(mode=mode,split=item['record']['split'],source_group=item['record']['source_group'],
                    scene=item['record']['scene'],box_matched_nme=primary['box_matched_nme'],
                    oracle_keypoint_assisted_nme=float(np.mean(errors)),assignment_agreement=float(np.mean(agreements)),
                    gt_in_local_window=float(np.mean(inside)),local_heatmap_peak_probability=float(np.mean(maxima)),
                    normalized_heatmap_entropy=float(np.mean(entropies)),
                    loss_components={k:float(v) for k,v in logs.items() if k.startswith('loss_')}))
    numeric=['box_matched_nme','oracle_keypoint_assisted_nme','assignment_agreement','gt_in_local_window',
             'local_heatmap_peak_probability','normalized_heatmap_entropy']
    summary={}
    for mode in ('off','local_depth_features'):
        summary[mode]={}
        for split in ('train','val'):
            rows=[r for r in records if r['mode']==mode and r['split']==split]
            summary[mode][split]={k:float(np.mean([r[k] for r in rows])) for k in numeric}
    write_json(a.output/'pose_diagnostics.json',dict(seed=a.seed,summary=summary,records=records,
         theoretical_peak_probability_bound=math.exp(2)/(math.exp(2)+63),
         warning='GT-assisted matching is an oracle diagnostic, not headline accuracy. No training or tuning.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
