import unittest
import torch
from mousegeoforge.models.independent_view_pose_probe import POSE_KEYS,independent_view_pose_probe

class IndependentViewProbeTests(unittest.TestCase):
    def forward(self,images):
        value=images.mean((2,3,4))[...,None]+images[:,0].mean((1,2,3))[:,None,None]
        return {**{k:value.clone() for k in POSE_KEYS},'depth':images.mean(2),'auxiliary_unvalidated':value.clone()}

    def test_single_unchanged_no_extra_forward(self):
        x=torch.randn(2,1,3,4,4);p=self.forward(x)
        def forbidden(_):raise AssertionError('Single already computed')
        r=independent_view_pose_probe(x,p,forbidden)
        for k in p:self.assertIs(r[k],p[k])

    def test_physical_permutation_consistency_and_geometry_preservation(self):
        x=torch.randn(2,3,3,4,4);p=self.forward(x);snapshot={k:v.clone() for k,v in p.items()}
        a=independent_view_pose_probe(x,p,self.forward)
        b=independent_view_pose_probe(x.flip(1),self.forward(x.flip(1)),self.forward)
        for k in POSE_KEYS:torch.testing.assert_close(a[k],b[k].flip(1),atol=0,rtol=0)
        for k in p:torch.testing.assert_close(p[k],snapshot[k],atol=0,rtol=0)
        self.assertIs(a['depth'],p['depth']);self.assertIs(a['auxiliary_unvalidated'],p['auxiliary_unvalidated'])

    def test_no_gradient(self):
        x=torch.randn(2,3,3,4,4,requires_grad=True);r=independent_view_pose_probe(x,self.forward(x),self.forward)
        for k in POSE_KEYS:self.assertFalse(r[k].requires_grad)

    def test_invalid_shapes(self):
        x=torch.randn(2,3,3,4,4);p=self.forward(x)
        with self.assertRaises(ValueError):independent_view_pose_probe(x[0],p,self.forward)
        with self.assertRaises(ValueError):independent_view_pose_probe(x,{},self.forward)
        with self.assertRaises(ValueError):independent_view_pose_probe(x,p,lambda _:self.forward(x))

if __name__=='__main__':unittest.main()
