"""Zero-start feature adapters over four frozen encoder levels; no backbone updates."""
import torch
from torch import nn
from mousegeoforge.semseg_training import MouseSemseg
class MouseSemsegFeatureAdapter(MouseSemseg):
 def __init__(self,encoder,*args,**kwargs):
  super().__init__(encoder,*args,**kwargs)
  channels=encoder.embed_dim;hidden=4 if channels==32 else 32
  self.adapters=nn.ModuleList([nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),nn.GELU(),nn.Conv2d(hidden,channels,1,bias=False)) for _ in range(4)])
  for adapter in self.adapters:nn.init.zeros_(adapter[2].weight)
 def features(self,image,arm):
  layers=super().features(image,arm)
  return [feature+adapter(feature) for feature,adapter in zip(layers,self.adapters)]
def configure(model,arm,step,phase_steps):
 model.eval()
 for name,p in model.named_parameters():p.requires_grad_(name.startswith('head.') or (model.adapter_mode=='adapter' and name.startswith('adapters.')))
 return []
def optimizer(model,arm,head_lr):
 groups=[]
 for prefix,peak in [('head.',3e-6),('adapters.',1e-5)]:
  parameters=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
  if parameters:groups.append(dict(params=parameters,name=prefix[:-1],peak_lr=peak,lr=peak))
 return torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
