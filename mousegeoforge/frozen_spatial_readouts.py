"""Read-only spatial/local decomposition. CF coordinates are not trained models."""
import torch
from torch.nn import functional as F
from .spatial_anchor_experiment import spatial_anchor_context
from .spatial_heatmap_probe import spatial_expectation
from .visible_heatmap_supervision import anchor_scores
from .deterministic_local_sampling import deterministic_local_refine

ARMS=('raw_coarse','baseline_final','spatial_CF','spatial_local_CF','mixed_coarse','trained_final')


def frozen_readouts(model,images):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Requires frozen eval model')
    head=model.instance_pose_head;branch=head.spatial_anchor;captured={};handles=[]
    def capture(name,inputs=False):
        def hook(m,a,value):
            captured[name]=value.detach()
            if inputs:captured['tokens']=a[0].detach()
        return hook
    try:
        handles=[head.pyramid_refine[0].register_forward_hook(capture('pixels')),
                 head.coarse_keypoint_xy.register_forward_hook(capture('raw',True)),
                 head.local_pixel_projection.register_forward_hook(capture('local_pixel')),
                 head.local_query_projection.register_forward_hook(capture('local_query'))]
        with torch.no_grad():
            with torch.autocast('cuda',dtype=torch.bfloat16):cache,start=model.aggregator(images)
            base=model.forward_cached(images,cache,start,coupling='off');s=dict(captured)
            s.update(local_query=F.normalize(s['local_query'],dim=-1,eps=1e-6),scale=head.last_temperature_scale.detach(),
                     size=head.local_heatmap_size,radius=head.local_window_radius)
            scores=anchor_scores(branch,s['tokens'],s['pixels'],base['instance_mask_logits'][0])
            anchor=spatial_expectation(s['raw'],scores,enabled=True).clamp(1e-5,1-1e-5)
            mixed=branch(s['raw'],s['tokens'],s['pixels'],base['instance_mask_logits'][0],enabled=True).sigmoid()
            raw=s['raw'].sigmoid();base_replay=deterministic_local_refine(s,raw)
            final=deterministic_local_refine(s,mixed);pure_local=deterministic_local_refine(s,anchor)
            with spatial_anchor_context(model,enabled=True):trained=model.forward_cached(images,cache,start,coupling='off')
            torch.testing.assert_close(base_replay,base['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(mixed,trained['instance_keypoint_coarse_xy'][0],atol=0,rtol=0)
            torch.testing.assert_close(final,trained['instance_keypoint_xy'][0],atol=1e-6,rtol=1e-6)
            for k in ('depth','instance_boxes_cxcywh','instance_mask_logits','instance_objectness_logits'):
                torch.testing.assert_close(base[k],trained[k],atol=0,rtol=0)
            coordinates=[raw,base['instance_keypoint_xy'][0],anchor,pure_local,mixed,trained['instance_keypoint_xy'][0]]
            result={a:xy.cpu() for a,xy in zip(ARMS,coordinates)}
            return result, {k:base[k].cpu() for k in ('instance_mask_logits','instance_objectness_logits')},float((final-trained['instance_keypoint_xy'][0]).abs().max())
    finally:
        for handle in handles:handle.remove()
        captured.clear()
