"""Experimental per-layer query-position vectors for cross-attention Q only.

These are learned embeddings, not box coordinates or persistent animal IDs.
No GT, image role, instance count, masks, hard spatial constraints or NMS.
"""
from contextlib import contextmanager
import torch
from torch import nn


class LayerQueryPosition(nn.Module):
    def __init__(self,layers,queries,dim):
        super().__init__()
        if min(layers,queries,dim)<=0:raise ValueError('Positive dimensions required')
        self.offsets=nn.Parameter(torch.zeros(layers,queries,dim))

    def forward(self,query,layer):
        if query.ndim!=3 or query.shape[1:]!=self.offsets.shape[1:]:
            raise ValueError('Expected batch-first query geometry')
        return query+self.offsets[layer].to(query)[None]


def install_query_position(model):
    head=model.instance_pose_head if hasattr(model,'instance_pose_head') else model
    if hasattr(head,'query_position'):raise ValueError('Query positions already installed')
    ref=next(head.parameters())
    head.query_position=LayerQueryPosition(len(head.instance_decoder.layers),head.num_queries,head.hidden_dim).to(ref)
    return model


@contextmanager
def query_position_context(model=None,*,enabled=False):
    if not enabled:
        yield
        return
    head=model.instance_pose_head if hasattr(model,'instance_pose_head') else model
    if not hasattr(head,'query_position'):raise ValueError('Install query positions first')
    if getattr(head,'_query_position_active',False):raise RuntimeError('Non-reentrant query-position context')
    handles=[];head._query_position_active=True
    try:
        for layer_index,layer in enumerate(head.instance_decoder.layers):
            def inject(module,args,kwargs,index=layer_index):
                if args:
                    return (head.query_position(args[0],index),*args[1:]),kwargs
                if 'query' not in kwargs:raise ValueError('Missing cross-attention query')
                return args,dict(kwargs,query=head.query_position(kwargs['query'],index))
            handles.append(layer.multihead_attn.register_forward_pre_hook(inject,with_kwargs=True))
        yield
    finally:
        for handle in handles:handle.remove()
        head._query_position_active=False
