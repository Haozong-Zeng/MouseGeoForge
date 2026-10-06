"""RNG-neutral loading guard for the immutable, replay-verified v1 runtime.

Forward and capture are inherited without changes. Load once during single-thread
setup; this protects the caller's CPU Torch RNG from temporary layer initialization.
"""
import torch
from mousegeoforge.mask_box_candidate import MaskBoxCandidate as AuditedCandidate


class MaskBoxCandidate(AuditedCandidate):
    @classmethod
    def load(cls,*args,**kwargs):
        with torch.random.fork_rng(devices=[]):
            return super().load(*args,**kwargs)
