"""Read-only camera/visibility/projected-curvature audit from saved predictions."""
from collections import defaultdict
import json
import numpy as np
from train_variable_view_seed_control import ROOT, OUTPUT, torch, write_json
from mousegeoforge.pilot_metrics import box_iou


def main():
    rows=[]; cross_instance=[]
    for arm in ('initial','replay_seed29','replay_seed43'):
        for path in sorted((OUTPUT/arm).glob('prediction_*.pt')):
            item=torch.load(path,weights_only=True)
            if item['group']!='fixed_pair': continue
            h,w=item['record']['image_hw']; scale=torch.tensor([w,h])
            for v,(qs,ms) in enumerate(item['oracle_assignment']):
                for q,m in zip(qs,ms):
                    gt=item['targets']['keypoint_xy'][0,v,m,7:14]*scale
                    pr=item['predictions']['instance_keypoint_xy'][0,v,q,7:14]*scale
                    valid=item['targets']['keypoint_valid'][0,v,m,7:14]
                    visible=item['targets']['keypoint_visible'][0,v,m,7:14]
                    known=item['targets']['keypoint_visibility_valid'][0,v,m,7:14]
                    diag=(item['targets']['boxes_cxcywh'][0,v,m,2:]*scale).norm().clamp_min(1)
                    score=float(item['predictions']['instance_objectness_logits'][0,v,q].sigmoid())
                    overlap=float(box_iou(item['predictions']['instance_boxes_cxcywh'][0,v,q][None],
                                          item['targets']['boxes_cxcywh'][0,v,m][None])[0,0])
                    others=[other for other in ms if other!=m]
                    for other in others:
                        other_gt=item['targets']['keypoint_xy'][0,v,other,7:14]*scale
                        common=valid & item['targets']['keypoint_valid'][0,v,other,7:14]
                        if common.any():
                            body=torch.tensor([*range(7),*range(14,23)])
                            full_pred=item['predictions']['instance_keypoint_xy'][0,v,q]*scale
                            full_gt=item['targets']['keypoint_xy'][0,v,m]*scale
                            full_other=item['targets']['keypoint_xy'][0,v,other]*scale
                            body_valid=(item['targets']['keypoint_valid'][0,v,m] &
                                        item['targets']['keypoint_valid'][0,v,other])[body]
                            own_body=((full_pred-full_gt).norm(dim=-1)/diag)[body][body_valid]
                            other_body=((full_pred-full_other).norm(dim=-1)/diag)[body][body_valid]
                            cross_instance.append(dict(arm=arm,frame=item['record']['frame'],
                                camera=item['record']['views'][v],instance=m,other_instance=other,
                                query=q,score=score,box_iou=overlap,detected=score>=.5 and overlap>=.5,
                                own_tail_nme=float(((pr-gt).norm(dim=-1)/diag)[common].mean()),
                                other_tail_nme=float(((pr-other_gt).norm(dim=-1)/diag)[common].mean()),
                                own_body_nme=float(own_body.mean()) if own_body.numel() else None,
                                other_body_nme=float(other_body.mean()) if other_body.numel() else None))
                    ratio=float((gt[1:]-gt[:-1]).norm(dim=-1).sum()/(gt[-1]-gt[0]).norm().clamp_min(1)) if valid.all() else None
                    bend='unknown' if ratio is None else 'straight' if ratio<1.1 else 'moderate' if ratio<1.5 else 'strong'
                    for j in range(7):
                        if valid[j]:
                            rows.append(dict(arm=arm,frame=item['record']['frame'],camera=item['record']['views'][v],
                                instance=m,channel=j+7,nme=float((pr[j]-gt[j]).norm()/diag),
                                visibility='unknown' if not known[j] else 'visible' if visible[j] else 'occluded',
                                projected_bend=bend,projected_arc_chord=ratio))
    grouped={}
    for axis in ('camera','visibility','projected_bend','channel','frame'):
        buckets=defaultdict(list)
        for r in rows: buckets[(r['arm'],str(r[axis]))].append(r['nme'])
        grouped[axis]=[dict(arm=a,category=c,count=len(v),nme=float(np.mean(v))) for (a,c),v in sorted(buckets.items())]
    write_json(ROOT/'output/tail_supervision_v1/tail_diagnostic.json',
        dict(rows=rows,groups=grouped,cross_instance=cross_instance,
             warning='Fixed-pair reused development set, GT box assignment; bend is 2D arc/chord, not 3D curvature. Other-tail proximity is a diagnostic, not proof of identity swap.'))
    print(json.dumps(grouped,indent=2))


if __name__=='__main__': main()
