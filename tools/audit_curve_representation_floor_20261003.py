"""Analytic GT-assisted representation/box bounds, NOT learned prediction results."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
import torch
from mousegeoforge.surface_curve_supervision import points_to_polyline
BASE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'
def stats(a):return dict(mean=float(np.mean(a)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(np.max(a)))
def main():
    torch.set_num_threads(2);saved=torch.load(BASE/'frozen_inputs.pt',map_location='cpu',weights_only=False);data=saved['tensors'];meta=saved['metadata']
    knots=np.r_[np.zeros(4),np.arange(1,3)/3,np.ones(4)];basis=BSpline(knots,np.eye(6),3)(np.linspace(0,1,65));sample=BSpline(knots,np.eye(6),3)(np.linspace(0,1,33))
    records=[]
    for i,m in enumerate(meta):
        gt=data['gt'][i].numpy().astype(float);px=data['pixels'][i].numpy();box=data['boxes'][i].numpy();lo=box[:2]-box[2:]/2;hi=box[:2]+box[2:]/2
        controls=np.linalg.lstsq(basis,gt,rcond=None)[0]
        bounded=np.stack([lsq_linear(basis,gt[:,k],bounds=(lo[k],hi[k]),tol=1e-12,max_iter=200).x for k in range(2)],axis=-1)
        row=dict(index=i,**m)
        for name,c in [('unbounded',controls),('box_bounded',bounded)]:
            p=torch.tensor(sample@c*px,dtype=torch.float64);g=torch.tensor(gt*px,dtype=torch.float64)
            row[name]=dict(material_parameter_rmse_px=float(np.sqrt(np.mean(np.square((basis@c-gt)*px)))),
                pred_to_gt_px=float(points_to_polyline(p,g,torch.ones(65,dtype=torch.bool))[0].sqrt().mean()),
                gt_to_pred_px=float(points_to_polyline(g,p,torch.ones(33,dtype=torch.bool))[0].sqrt().mean()),
                endpoints_px=float((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).mean()))
        row['outside_predicted_box_fraction']=float(np.any((gt<lo)|(gt>hi),axis=-1).mean());records.append(row)
    summary={}
    for role in ['train','validation']:
        rr=[r for r in records if r['role']==role]
        summary[role]={mode:{key:stats([r[mode][key] for r in rr]) for key in ['pred_to_gt_px','gt_to_pred_px','material_parameter_rmse_px','endpoints_px']} for mode in ['unbounded','box_bounded']}
    payload=dict(method='Analytic per-sample GT-assisted cubic6control spline fits, known rest-material parameters, bounded or unbounded. Not a model result or heldout score.',
        purpose='Separates representational limitations and wrong-box floors from image-feature/optimization failures. No hyperparameter/checkpoint selected from these oracle fits.',summary=summary,records=records)
    (BASE/'representation_floor.json').write_text(json.dumps(payload,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
