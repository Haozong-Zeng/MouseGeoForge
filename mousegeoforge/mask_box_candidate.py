"""Explicit, default-OFF candidate loading and inference-only box postprocessing.

CPU/FP32 connected-component prototype. Never changes base flags, modes or weights.
Capture contexts are intentionally non-reentrant and one-forward/one-consume.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import torch
from mousegeoforge.mask_box_refiner import features_from_logits
from mousegeoforge.mask_reference_box import MaskReferenceBoxRefiner


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_mapping_sha256(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class MaskBoxCandidate:
    def __init__(self,refiner=None,*,base_state_sha256=None,encoder_state_sha256=None):
        self._refiner=refiner.cpu().float().eval().requires_grad_(False) if refiner is not None else None
        self._base_hash=base_state_sha256;self._encoder_hash=encoder_state_sha256
        self._active=False;self._tokens=None;self._calls=0

    @classmethod
    def load(cls,directory,*,base_checkpoint,encoder_checkpoint,repository_root):
        directory=Path(directory).resolve();root=Path(repository_root).resolve()
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('schema_version')!=1 or manifest.get('default_enabled') is not False:raise ValueError('Unsupported candidate manifest')
        for path,key in ((base_checkpoint,'base_checkpoint_sha256'),(encoder_checkpoint,'encoder_checkpoint_sha256')):
            if file_sha256(path)!=manifest[key]:raise ValueError(f'Candidate {key} mismatch')
        for relative,expected in manifest['source_sha256'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or file_sha256(path)!=expected:raise ValueError('Candidate source mismatch')
        weights=(directory/manifest['weights']).resolve()
        if weights.parent!=directory or file_sha256(weights)!=manifest['weights_sha256']:raise ValueError('Candidate weights mismatch')
        state=torch.load(weights,map_location='cpu',weights_only=True)
        if tensor_mapping_sha256(state)!=manifest['refiner_state_sha256']:raise ValueError('Candidate state mismatch')
        refiner=MaskReferenceBoxRefiner();refiner.load_state_dict(state,strict=True)
        return cls(refiner,base_state_sha256=manifest['base_state_sha256'],encoder_state_sha256=manifest['encoder_state_sha256'])

    @contextmanager
    def capture(self,model=None,*,enabled=False):
        if not enabled:
            yield self
            return
        if self._active:raise RuntimeError('Candidate capture is not reentrant')
        if self._refiner is None or not self._base_hash or not self._encoder_hash:raise RuntimeError('Load a paired candidate before enabling')
        if model is None or any(m.training for m in model.modules()):raise ValueError('Base model must already be in eval mode')
        state=model.state_dict()
        if tensor_mapping_sha256({k:v for k,v in state.items() if not k.startswith('aggregator.')})!=self._base_hash:raise ValueError('In-memory base model mismatch')
        if tensor_mapping_sha256({k:v for k,v in state.items() if k.startswith('aggregator.')})!=self._encoder_hash:raise ValueError('In-memory encoder mismatch')
        self._active=True;self._tokens=None;self._calls=0
        def capture_tokens(module,args,value):
            self._calls+=1;self._tokens=value.detach().cpu()
        hook=model.instance_pose_head.instance_decoder.register_forward_hook(capture_tokens)
        try:yield self
        finally:
            hook.remove();self._active=False;self._tokens=None;self._calls=0

    @torch.no_grad()
    def __call__(self,predictions,image_hw=None,*,enabled=False):
        if not enabled:return predictions
        if not self._active or self._calls!=1 or self._tokens is None:raise RuntimeError('Exactly one captured forward is required per enabled call')
        token=self._tokens;self._tokens=None;self._calls=0
        if image_hw is None or len(image_hw)!=2 or any(not isinstance(x,int) or x<=0 for x in image_hw):raise ValueError('Expected positive integer image HW')
        boxes=predictions['instance_boxes_cxcywh'];logits=predictions['instance_mask_logits']
        if boxes.ndim!=4 or boxes.shape[-1]!=4 or boxes.dtype!=torch.float32:raise ValueError('Expected FP32 B,V,Q,4 boxes')
        b,v,q,_=boxes.shape
        if logits.ndim!=5 or logits.shape[:3]!=(b,v,q) or token.shape!=(b*v,q,self._refiner.token_dim):raise ValueError('Mask/token/query shape mismatch')
        shape=features_from_logits(logits,image_hw)
        with torch.autocast('cpu',enabled=False):
            refined=self._refiner(boxes.detach().cpu(),token.reshape(b,v,q,self._refiner.token_dim).float(),shape,enabled=True,use_prior=True)
        return dict(predictions,instance_boxes_cxcywh=refined.to(device=boxes.device,dtype=boxes.dtype))
