"""Opt-in group-conditioned candidate refinement, with conservative fallback.

Groups contain predicted physical-camera/query correspondences only. Annotation
identities or the number of animals must not be supplied. No detector output is
modified; rejected/unassigned nodes remain exactly original.
"""
import numpy as np
import torch
from torch import nn
from ..candidate_selector_features import KP


class GroupedCandidateFusion(nn.Module):
    def __init__(self,refiner):
        super().__init__()
        self.refiner=refiner.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,predictions,groups=None,projections=None,physical_views=None,pixels=None,
                *,enabled=False,local_enabled=False,fusion_enabled=True,view_valid=None):
        original=predictions['xy']
        if original.ndim!=4 or original.shape[1:]!=(6,23,2):raise ValueError('Expected XY[V,6,23,2]')
        v=len(original);empty=torch.zeros(v,6,5,dtype=torch.bool,device=original.device)
        result=dict(xy=original,associated=empty.clone(),local_eligible=empty.clone(),
                    fusion_eligible=empty.clone(),changed=empty.clone(),accepted_groups=[],rejected_groups=[])
        if not enabled or not (local_enabled or fusion_enabled):return result
        groups=[] if groups is None else groups
        if projections is None or physical_views is None or pixels is None:
            result['rejected_groups']=[dict(index=i,reason='missing_calibration') for i in range(len(groups))];return result
        views=list(physical_views);p=np.asarray(projections,float);px=np.asarray(pixels,float)
        valid=np.ones(v,bool) if view_valid is None else np.asarray(view_valid,bool)
        if len(views)!=v or p.shape!=(v,3,4) or valid.shape!=(v,) or px.shape not in [(2,),(v,2)]:raise ValueError('Invalid camera or pixel shapes')
        if not np.isfinite(px).all() or (px<=0).any():raise ValueError('Invalid pixels')
        logits=predictions['objectness_logits'];validated={};reasons={}
        for gi,group in enumerate(groups):
            try:
                pairs=[tuple(x) for x in group]
                if any(len(x)!=2 or not isinstance(x[1],(int,np.integer)) or isinstance(x[1],bool) for x in pairs):raise ValueError()
                if len(pairs)<3:reasons[gi]='fewer_than_three_views';continue
                if len({x[0] for x in pairs})!=len(pairs):reasons[gi]='duplicate_group_view';continue
                indices=[]
                for physical,q in pairs:
                    if views.count(physical)!=1 or q<0 or q>=6:reasons[gi]='missing_or_duplicate_view_or_query';break
                    vi=views.index(physical)
                    if not valid[vi] or not np.isfinite(p[vi]).all() or np.linalg.matrix_rank(p[vi])<3:reasons[gi]='invalid_calibration';break
                    if not torch.isfinite(logits[vi,q]) or logits[vi,q].sigmoid()<.5:reasons[gi]='nonconfident_query';break
                    indices.append((vi,int(q)))
                if gi not in reasons:validated[gi]=indices
            except (TypeError,ValueError):
                reasons[gi]='malformed_group'
        # Reject every overlapping group rather than making the first win.
        overlapping={i for i,a in validated.items() for j,b in validated.items() if i!=j and set(a)&set(b)}
        for gi in overlapping:reasons[gi]='overlapping_groups'
        xy=original.clone()
        for gi,indices in validated.items():
            if gi in reasons:continue
            indices=sorted(indices,key=lambda z:views[z[0]])
            ids=[i for i,q in indices]
            sub={key:value[ids] for key,value in predictions.items()}
            sub['objectness_logits']=torch.full_like(logits[ids],-20)
            for row,(vi,q) in enumerate(indices):sub['objectness_logits'][row,q]=logits[vi,q]
            refined=self.refiner(sub,p[ids],[views[i] for i in ids],px if px.ndim==1 else px[ids],
                                 local_enabled=local_enabled,fusion_enabled=fusion_enabled)
            for row,(vi,q) in enumerate(indices):
                xy[vi,q,KP]=refined['xy'][row,q,KP]
                result['associated'][vi,q]=True
                for key in ['local_eligible','fusion_eligible']:result[key][vi,q]=refined[key][row,q]
            result['accepted_groups'].append(gi)
        result.update(xy=xy,changed=(xy[:,:,KP]!=original[:,:,KP]).any(-1),
                      rejected_groups=[dict(index=i,reason=reasons[i]) for i in sorted(reasons)])
        return result

