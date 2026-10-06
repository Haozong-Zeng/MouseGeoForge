"""CPU-only label scale audit, no training/camera/model change."""
import json
import numpy as np
import torch
from run_depth_pose_ablation import ROOT,write_json
from train_cylinder_view_balance import oriented_batch


def main():
    torch.set_num_threads(2);out=ROOT/'output/keypoint_pixel_scale_audit_v1';out.mkdir(exist_ok=False)
    train=json.loads((ROOT/'output/spatial_anchor_visible_heatmap_v1/protocol.json').read_text())
    dev=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
    items=[('train',train['records'][k]['source_group'],train['records'][k]) for k in train['train_keys']]
    items += [('dev',e['group'],dev['records'][e['key']]) for e in dev['evaluation']]
    body=list(range(7))+list(range(14,23));rows=[]
    for split,group,r in items:
        b=oriented_batch(r,num_views=len(r['views']));h,w=b['images'].shape[-2:]
        for v in range(len(r['views'])):
            for m in torch.where(b['instance_valid'][0,v])[0].tolist():
                valid=b['instance_keypoint_valid'][0,v,m].numpy();visible=(b['instance_keypoint_visible'][0,v,m]&b['instance_keypoint_visibility_valid'][0,v,m]).numpy()
                xy=b['instance_keypoint_xy'][0,v,m].numpy()*[w,h]
                ids=[k for k in body if valid[k] and visible[k]]
                if len(ids)<2:continue
                d=np.linalg.norm(xy[ids,None]-xy[None,ids],axis=-1);tri=d[np.triu_indices(len(ids),1)]
                unique=d.copy();unique[unique<1e-5]=np.inf;nearest=unique.min(-1);nearest=nearest[np.isfinite(nearest)]
                allbody=xy[[k for k in body if valid[k]]];span=np.ptp(allbody,axis=0)
                rows.append(dict(split=split,group=group,scene=r['scene'],view=v,role=r['physical_roles'][v],instance=m,
                    body_span_diagonal_px=float(np.linalg.norm(span)),box_diagonal_px=float(np.linalg.norm(b['boxes_cxcywh'][0,v,m,2:].numpy()*[w,h])),
                    visible_body_points=len(ids),coincident_visible_pairs=int((tri<1e-5).sum()),visible_pairs=len(tri),
                    nearest_distinct_visible_distances_px=nearest.tolist(),pairs_below8=int(((tri>1e-5)&(tri<8)).sum()),pairs_below16=int(((tri>1e-5)&(tri<16)).sum())))
    summary={}
    for split,group in dict.fromkeys((r['split'],r['group']) for r in rows):
        selected=[r for r in rows if r['split']==split and r['group']==group];dist=[d for r in selected for d in r['nearest_distinct_visible_distances_px']]
        summary[split+'/'+group]=dict(instance_views=len(selected),median_body_span_diagonal_px=float(np.median([r['body_span_diagonal_px'] for r in selected])),
            median_nearest_visible_px=float(np.median(dist)),nearest_visible_below8_fraction=float(np.mean(np.array(dist)<8)),
            nearest_visible_below16_fraction=float(np.mean(np.array(dist)<16)),coincident_pairs=sum(r['coincident_visible_pairs'] for r in selected))
    write_json(out/'records.json',rows);write_json(out/'summary.json',summary)
    write_json(out/'verification.json',dict(passed=True,scene_records=len(items),instance_views=len(rows),training_steps=0,real_accessed=False,
        interpretation='2D visible projected spacing only; coincident semantic points excluded from nearest DISTINCT distance and reported separately. Does not establish a resolution or Gaussian-sigma causal effect.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
