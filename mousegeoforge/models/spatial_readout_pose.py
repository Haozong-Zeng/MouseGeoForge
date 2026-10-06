"""Opt-in inference composition for full-query spatial pose readout.

No labels, GT matching or GT instance counts enter forward. This is an
experimental wrapper, not enabled by existing model factories/checkpoints.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .surface_curve_pose import SurfaceCurveDepthPoseModel
from .core23_spatial_readout import Core23SpatialReadout
from .spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi


@contextmanager
def _cudnn_tf32_only(value):
    # cudnn.flags(allow_tf32=...) defaults enabled=False in this runtime.
    # Change only the requested flag, preserving all other backend settings.
    previous=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=value
        yield
    finally:
        torch.backends.cudnn.allow_tf32=previous


class SpatialReadoutDepthPoseModel(SurfaceCurveDepthPoseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_readout = Core23SpatialReadout('spatial')
        self.spatial_pose_enabled = False
        self.spatial_detector = SpatialDetectionReadout('roi')
        self.spatial_detector_enabled = False
        # Training caches stored fp16 memory, then resized to48x48float32.
        # Keep this explicit numeric path for reproducible live/cache parity.
        self.spatial_memory_fp16_roundtrip = True
        self.spatial_decoder_tf32 = True

    def forward(self, images, *, coupling=None):
        if not (self.spatial_pose_enabled or self.spatial_detector_enabled):
            return super().forward(images,coupling=coupling)
        # Match the original real cache: bf16 encoder with cuDNN TF32 off,
        # fp32 decoder with cuDNN TF32 on. Contexts restore caller flags.
        with _cudnn_tf32_only(False):
            return super().forward(images,coupling=coupling)

    def _decode(self, images, cached, patch_start, *, coupling=None):
        if not (self.spatial_pose_enabled or self.spatial_detector_enabled):
            return super()._decode(images, cached, patch_start, coupling=coupling)
        mode = self.coupling if coupling is None else coupling
        if mode != 'off':
            raise ValueError('Spatial readout was trained with coupling=off; joint bridge integration is unvalidated')
        memory, tokens, instance_tokens = [], [], []
        head = self.instance_pose_head
        a = head.instance_decoder.register_forward_pre_hook(lambda module,args: memory.append(args[1]))
        b = head.keypoint_decoder.register_forward_hook(lambda module,args,out: tokens.append(out))
        c = head.instance_decoder.register_forward_hook(lambda module,args,out: instance_tokens.append(out))
        try:
            with _cudnn_tf32_only(self.spatial_decoder_tf32):
                pred = super()._decode(images, cached, patch_start, coupling=coupling)
        finally:
            a.remove()
            b.remove()
            c.remove()
        if len(memory) != 1 or len(tokens) != 1 or len(instance_tokens)!=1:
            raise RuntimeError('Expected exactly one instance/keypoint decoder invocation')
        batch, views = images.shape[:2]
        h, w = images.shape[-2:]
        q, k = pred['instance_keypoint_coarse_xy'].shape[2:4]
        count = batch * views
        maps = memory[0].transpose(1,2).reshape(count,head.hidden_dim,h//head.patch_size,w//head.patch_size)
        if self.spatial_memory_fp16_roundtrip:
            maps = maps.half().float()
        maps = F.interpolate(maps.float(), (48,48), mode='bilinear', align_corners=False)
        if self.spatial_detector_enabled:
            original_boxes=pred['instance_boxes_cxcywh'];original_logits=pred['instance_objectness_logits']
            roi,global_memory=sample_predicted_roi(maps,original_boxes.reshape(count,q,4))
            # Detection cache stores sampled ROI/global context in fp16 too.
            adapted=self.spatial_detector(instance_tokens[0].reshape(count,q,head.hidden_dim),roi.half().float(),global_memory.half().float(),original_boxes.reshape(count,q,4),original_logits.reshape(count,q))
            pred={**pred,'instance_boxes_original_cxcywh':original_boxes,'instance_objectness_original_logits':original_logits,
                  'instance_boxes_cxcywh':adapted['instance_boxes_cxcywh'].reshape(batch,views,q,4),
                  'instance_objectness_logits':adapted['instance_objectness_logits'].reshape(batch,views,q)}
        if not self.spatial_pose_enabled:return pred
        pose = tokens[0].reshape(count*q,k,head.hidden_dim)
        shared = maps[:,None].expand(-1,q,-1,-1,-1).reshape(count*q,head.hidden_dim,48,48)
        xy = self.spatial_readout(pose,pred['instance_keypoint_coarse_xy'].reshape(count*q,k,2),shared,
                                  predicted_boxes=pred['instance_boxes_cxcywh'].reshape(count*q,4))
        # Keep local-refined reference explicit. Its local heatmaps remain
        # auxiliary legacy outputs; they do NOT describe the new final XY.
        return {**pred,
                'instance_keypoint_original_xy': pred['instance_keypoint_xy'],
                'instance_keypoint_xy': xy.reshape(batch,views,q,k,2)}
