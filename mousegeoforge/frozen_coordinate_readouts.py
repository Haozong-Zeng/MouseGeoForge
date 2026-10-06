"""Frozen F/global/relative coordinate residual diagnostic; no production install."""
import torch
from torch.nn import functional as F
from .spatial_anchor_experiment import spatial_anchor_context
from .spatial_heatmap_probe import spatial_expectation
from .visible_heatmap_supervision import anchor_scores
from .deterministic_local_sampling import deterministic_local_refine
from .instance_adaptive_gate import predicted_quality
from .instance_coordinate_residual import reference_from_shape
from .mask_box_refiner import features_from_logits


def coordinate_readouts(model,heads,images):
    if images.ndim!=5 or images.shape[0]!=1:raise ValueError('Diagnostic supports B1 with real view dimension')
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval base required')
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in heads.values()):raise ValueError('Frozen residual heads required')
    head=model.instance_pose_head;branch=head.spatial_anchor;captured={};handles=[]
    def capture(name,inputs=False):
        def hook(m,a,value):
            captured[name]=value.detach()
            if inputs:captured['tokens']=a[0].detach()
        return hook
    try:
        handles=[head.pyramid_refine[0].register_forward_hook(capture('pixels')),head.coarse_keypoint_xy.register_forward_hook(capture('raw',True)),
                 head.local_pixel_projection.register_forward_hook(capture('local_pixel')),head.local_query_projection.register_forward_hook(capture('local_query'))]
        with torch.no_grad():
            with torch.autocast('cuda',dtype=torch.bfloat16):cache,start=model.aggregator(images)
            base=model.forward_cached(images,cache,start,coupling='off');s=dict(captured)
            s.update(local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),scale=head.last_temperature_scale.detach(),size=head.local_heatmap_size,radius=head.local_window_radius)
            scores=anchor_scores(branch,s['tokens'],s['pixels'],base['instance_mask_logits'][0])
            anchor=spatial_expectation(s['raw'],scores,enabled=True).clamp(1e-5,1-1e-5)
            mixed=branch(s['raw'],s['tokens'],s['pixels'],base['instance_mask_logits'][0],enabled=True).sigmoid()
            with spatial_anchor_context(model,enabled=True):frozen=model.forward_cached(images,cache,start,coupling='off')
            torch.testing.assert_close(mixed,frozen['instance_keypoint_coarse_xy'][0],atol=0,rtol=0)
            replay=deterministic_local_refine(s,mixed)
            torch.testing.assert_close(replay,frozen['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
            for k in ('depth','instance_boxes_cxcywh','instance_mask_logits','instance_objectness_logits'):
                torch.testing.assert_close(base[k],frozen[k],atol=0,rtol=0)
            quality=predicted_quality(scores,base['instance_keypoint_visibility_logits'][0],base['instance_mask_logits'][0],base['instance_objectness_logits'][0])
            geometry=reference_from_shape(features_from_logits(base['instance_mask_logits'][0],images.shape[-2:])).to(mixed)
            outputs={'frozen':(frozen['instance_keypoint_xy'][0].cpu(),mixed.cpu())}
            for name,decoder in heads.items():
                coarse=decoder(mixed,anchor,s['tokens'],quality,geometry)
                outputs[name]=(deterministic_local_refine(s,coarse).cpu(),coarse.cpu())
            return outputs,{k:base[k].cpu() for k in ('instance_mask_logits','instance_objectness_logits')},float((replay-frozen['instance_keypoint_xy'][0]).abs().max())
    finally:
        for handle in handles:handle.remove()
        captured.clear()
