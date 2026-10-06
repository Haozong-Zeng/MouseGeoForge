"""Frozen baseline17 RGB-only runtime shared by bounded experiments."""
from pathlib import Path
import torch
from torch.nn import functional as F
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from run_depth_pose_ablation import fingerprint
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/core23_spatial_cycle_20261004';KP=[0,14,15,16,17]


class FrozenSingleViewCandidateRuntime:
    def __init__(self):
        self.paths=dict(encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt',initial=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot/initial_heads.pt',
            base=OLD/'detection_features_v1/initial_detector.pt',detector=OLD/'detection_view_context_control_v1/mixed_seed17_latest.pt',pose=OLD/'full_context_pose_control_v1/full_seed17_latest.pt')
        self.hashes={k:fingerprint(p) for k,p in self.paths.items()}
        import json
        native=json.loads((OLD/'native_decoder_memory_v1/manifest.json').read_text())
        assert self.hashes['encoder']==native['encoder_sha256'] and self.hashes['initial']==native['initial_sha256']
        m=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
        raw=torch.load(self.paths['encoder'],map_location='cpu',weights_only=True,mmap=True);selected={k:v for k,v in raw.items() if k.startswith(('aggregator.','dense_head.'))}
        result=m.load_state_dict(selected,strict=False);assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del raw,selected
        self.model=SurfaceCurveDepthPoseModel.from_spatial_model(m).cuda().eval().requires_grad_(False);del m
        result=self.model.load_state_dict(torch.load(self.paths['initial'],map_location='cpu',weights_only=True),strict=False);assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys)
        self.base=InstanceDetectionReadout().cuda().eval().requires_grad_(False);self.base.load_state_dict(torch.load(self.paths['base'],map_location='cpu',weights_only=True),strict=True)
        self.detector=SpatialDetectionReadout().cuda().eval().requires_grad_(False);self.detector.load_state_dict(torch.load(self.paths['detector'],map_location='cpu',weights_only=False)['state'],strict=True)
        self.pose=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);self.pose.load_state_dict(torch.load(self.paths['pose'],map_location='cpu',weights_only=False)['state'],strict=True)

    @torch.no_grad()
    def readout(self,memory,tokens):
        torch.backends.cudnn.allow_tf32=True
        original=self.base(tokens);maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False)
        roi,glob=sample_predicted_roi(maps,original['instance_boxes_cxcywh'])
        d=self.detector(tokens,roi.half().float(),glob.half().float(),original['instance_boxes_cxcywh'],original['instance_objectness_logits'])
        p=self.pose(tokens,memory,d['instance_boxes_cxcywh'])
        xy,prob=self.pose.spatial_readout(p['tokens'].reshape(6,23,256),p['coarse'].reshape(6,23,2),maps.expand(6,-1,-1,-1),return_probability=True,predicted_boxes=d['instance_boxes_cxcywh'].reshape(6,4),detach_pose_tokens=False)
        assert float((xy-p['xy'][0]).abs().max())<1e-7
        return dict(xy=p['xy'][0],boxes=d['instance_boxes_cxcywh'][0],objectness_logits=d['instance_objectness_logits'][0],tokens=p['tokens'][0,:,KP],probability=prob[:,KP],maps=maps)

    @torch.no_grad()
    def from_images(self,images):
        if images.ndim!=5 or images.shape[:3]!=(1,1,3):raise ValueError('Independent V1 RGB required')
        images=images.cuda();memory=[];instance=[];head=self.model.instance_pose_head
        hooks=[head.instance_decoder.register_forward_pre_hook(lambda module,args:memory.append(args[1].detach())),head.instance_decoder.register_forward_hook(lambda module,args,out:instance.append(out.detach()))]
        try:
            torch.backends.cudnn.allow_tf32=False
            with torch.autocast('cuda',dtype=torch.bfloat16):features,start=self.model.aggregator(images)
            features=[x if j in (4,11,17,23) else None for j,x in enumerate(features)]
            torch.backends.cudnn.allow_tf32=True
            self.model.forward_cached(images,features,start,coupling='off')
        finally:
            for hook in hooks:hook.remove()
        assert len(memory)==len(instance)==1
        h,w=images.shape[-2:];spatial=memory[0].transpose(1,2).reshape(1,256,h//head.patch_size,w//head.patch_size)
        return self.readout(spatial,instance[0])
