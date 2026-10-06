import sys,unittest,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mousegeoforge.models.local_point_evidence_probe import LocalPointEvidenceProbe,point_evidence_loss,gaussian_mixture_targets,fixed_top_peaks

class EvidenceProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17);self.m=LocalPointEvidenceProbe();self.f=torch.randn(32,12,16);self.hw=(44,60);self.target=torch.rand(2,5,2);self.valid=torch.ones(2,5,dtype=torch.bool)
    def test_no_gt_forward_interface(self):self.assertEqual(list(inspect.signature(self.m.forward).parameters),['features','image_hw'])
    def test_padding_exclusion(self):
        out=self.m(self.f,self.hw);p=fixed_top_peaks(out,self.hw);self.assertTrue((p['probability_map'][:,-1,:]==0).all());self.assertTrue((p['probability_map'][:,:,-1]==0).all());self.assertTrue(p['available'].all());self.assertTrue((p['xy']<1).all())
    def test_instance_permutation(self):
        out=self.m(self.f,self.hw);a=point_evidence_loss(out,self.target,self.valid,self.hw);b=point_evidence_loss(out,self.target.flip(0),self.valid.flip(0),self.hw);self.assertEqual(float(a),float(b))
    def test_missing_nan_ignored(self):
        out=self.m(self.f,self.hw);self.valid[0]=False;a=point_evidence_loss(out,self.target,self.valid,self.hw);self.target[0]=float('nan');b=point_evidence_loss(out,self.target,self.valid,self.hw);self.assertEqual(float(a),float(b))
    def test_unknown_class_no_gradient(self):
        self.valid[:,2]=False;out=self.m(self.f,self.hw);point_evidence_loss(out,self.target,self.valid,self.hw).backward();self.assertTrue((self.m.layers[-1].weight.grad[2]==0).all())
    def test_all_unknown_zero(self):
        out=self.m(self.f,self.hw);loss=point_evidence_loss(out,torch.full_like(self.target,float('nan')),torch.zeros_like(self.valid),self.hw);loss.backward();self.assertEqual(float(loss),0);self.assertTrue(all(p.grad is None or (p.grad==0).all() for p in self.m.parameters()))
    def test_finite_gradient(self):
        out=self.m(self.f,self.hw);loss=point_evidence_loss(out,self.target,self.valid,self.hw);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(self.m.layers[-1].weight.grad.norm()),0);self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters()))
    def test_mixture_normalized(self):
        out=self.m(self.f,self.hw);d,k=gaussian_mixture_targets(out,self.target,self.valid,self.hw);self.assertTrue(torch.allclose(d.sum((-1,-2)),torch.ones(5)));self.assertTrue(k.all());self.assertTrue((d[:,-1]==0).all())
    def test_duplicate_point_invariant(self):
        out=self.m(self.f,self.hw);t=self.target[:1];v=self.valid[:1];a=gaussian_mixture_targets(out,t,v,self.hw)[0];b=gaussian_mixture_targets(out,t.expand(2,-1,-1),v.expand(2,-1),self.hw)[0];self.assertTrue(torch.equal(a,b))
    def test_nms_separation(self):
        p=fixed_top_peaks(self.m(self.f,self.hw),self.hw);xy=p['xy']*torch.tensor([60,44]);distance=torch.cdist(xy,xy);distance.diagonal(dim1=-2,dim2=-1).fill_(100);self.assertTrue((distance>8).all())
    def test_known_nan_rejected(self):
        self.target[0,0]=float('nan')
        with self.assertRaises(ValueError):point_evidence_loss(self.m(self.f,self.hw),self.target,self.valid,self.hw)
if __name__=='__main__':unittest.main()
