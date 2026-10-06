"""Experimental RGB-gradient route; no production factory or annotation inputs."""
import torch
from vggt_omega.models.heads.instance_pose_head import _sine_position_encoding
from mousegeoforge.models.low_rank_linear import LowRankLinear
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime


class PixelAdaptationRuntime(FrozenSingleViewCandidateRuntime):
    def __init__(self,rank=8):
        super().__init__()
        attention=self.model.aggregator.patch_embed.blocks[-1].attn
        self.adapter=LowRankLinear(attention.qkv,rank=rank,alpha=rank)
        attention.qkv=self.adapter

    def native_forward(self,images,predicted_boxes,*,allow_input_grad=False):
        if images.ndim!=5 or images.shape[:3]!=(1,1,3):raise ValueError('Actual independentV1 RGB required')
        images=images.cuda();head=self.model.instance_pose_head
        torch.backends.cudnn.allow_tf32=False
        with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=self.model.aggregator(images)
        torch.backends.cudnn.allow_tf32=True
        # Reproduce only the graph that supplies native memory/instance tokens.
        # The unrelated depth/mask branches are not needed for this pose pilot.
        with torch.autocast('cuda',enabled=False):
            adapted=self.model.shared_adapter(cached,start);h,w=images.shape[-2:];ph,pw=h//head.patch_size,w//head.patch_size
            projected=head._project_cached_tokens(adapted,start,1,1,ph,pw)
            memory=torch.stack(projected).mean(0).flatten(2).transpose(1,2)
            memory=memory+_sine_position_encoding(ph,pw,head.hidden_dim,dtype=memory.dtype,device=memory.device)
            instance=head.instance_decoder(head.instance_queries.weight[None],memory)
            native=memory.transpose(1,2).reshape(1,256,ph,pw)
            output=self.pose(instance,native,predicted_boxes.detach().cuda(),allow_input_grad=allow_input_grad)
        return output

    def named_all_parameters(self):
        for prefix,module in [('model',self.model),('base',self.base),('detector',self.detector),('pose',self.pose)]:
            for name,param in module.named_parameters():yield prefix+'.'+name,param

