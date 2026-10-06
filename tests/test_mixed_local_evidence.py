import sys,unittest,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss

class MixedEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17);self.backbone=nn.Sequential(nn.Conv2d(3,32,3,stride=4,padding=1),nn.BatchNorm2d(32),nn.ReLU());self.rgb=torch.rand(1,3,32,32);self.target=torch.rand(2,5,2);self.known=torch.ones(2,5,dtype=torch.bool)
    def test_no_gt_interface(self):self.assertEqual(list(inspect.signature(MixedLocalEvidence.forward).parameters),['self','normalized_rgb','image_hw'])
    def test_frozen_backbone_grad(self):
        m=MixedLocalEvidence(self.backbone);torch.nn.init.normal_(m.readout.layers[-1].weight,std=.01);point_evidence_loss(m(self.rgb,(32,32)),self.target,self.known,(32,32)).backward();self.assertTrue(all(p.grad is None for p in m.backbone.parameters()));self.assertTrue(any(p.grad is not None for p in m.readout.parameters()))
    def test_adapt_gradient_and_buffers(self):
        m=MixedLocalEvidence(self.backbone,adapt_backbone=True);m.train();before={k:v.clone() for k,v in m.backbone.named_buffers()};torch.nn.init.normal_(m.readout.layers[-1].weight,std=.01);point_evidence_loss(m(self.rgb,(32,32)),self.target,self.known,(32,32)).backward();self.assertGreater(float(m.backbone[0].weight.grad.norm()),0);self.assertFalse(m.backbone[1].training);self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.backbone.named_buffers()))
    def test_native_features(self):
        m=MixedLocalEvidence(self.backbone);self.assertTrue(torch.equal(m(self.rgb,(32,32))['features'],m.backbone(self.rgb)[0]))
    def test_eval_chain_returns_model(self):
        m=MixedLocalEvidence(self.backbone);self.assertIs(m.eval(),m);self.assertIs(m.train(),m)
if __name__=='__main__':unittest.main()
