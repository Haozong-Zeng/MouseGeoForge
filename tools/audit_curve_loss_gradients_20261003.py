"""CPU analytic-gradient diagnostic; no model optimization or heldout selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mousegeoforge.surface_curve_supervision import cubic_bspline,curve_loss
BASE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'
def main():
    torch.set_num_threads(2);saved=torch.load(BASE/'frozen_inputs.pt',map_location='cpu',weights_only=False);data=saved['tensors'];rows=[]
    for i,meta in enumerate(saved['metadata']):
        if meta['role']!='train':continue
        b=data['boxes'][i];control=(b[:2]+(data['base_logits'][i].sigmoid()-.5)*b[2:]).detach().requires_grad_(True)
        p=cubic_bspline(control,33)*data['pixels'][i];gt=data['gt'][i]*data['pixels'][i];valid=torch.ones(65,dtype=torch.bool)
        total=curve_loss(p,gt,valid,full_curve=True);geometry=curve_loss(p,gt,valid,full_curve=True,order_weight=0.)
        gg=torch.autograd.grad(geometry,control,retain_graph=True)[0];go=torch.autograd.grad(total-geometry,control)[0]
        rows.append(dict(index=i,**meta,geometry_loss=float(geometry.detach()),order_loss=float((total-geometry).detach()),geometry_grad_norm=float(gg.norm()),order_grad_norm=float(go.norm()),
            order_to_geometry_grad_ratio=float(go.norm()/gg.norm().clamp_min(1e-12)),gradient_cosine=float((gg*go).sum()/(gg.norm()*go.norm()).clamp_min(1e-12))))
    x=np.array([r['order_to_geometry_grad_ratio'] for r in rows]);summary=dict(train_count=len(rows),ratio_median=float(np.median(x)),ratio_p90=float(np.quantile(x,.9)),ratio_max=float(x.max()),order_dominates_count=int((x>1).sum()),order_over10x_count=int((x>10).sum()))
    (BASE/'gradient_audit.json').write_text(json.dumps(dict(summary=summary,rows=rows,interpretation='Gradient norms w.r.t6normalized control positions at identical initial curve; sourceTRAINonly. Diagnostic, not a trained-model score.'),indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
