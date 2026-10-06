import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from train_synthetic_retention_20261005 import retention_objective,objective,KP
class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.x=torch.rand(6,23,2,requires_grad=True);self.g=torch.rand(6,5,2);self.v=torch.ones(6,5,dtype=torch.bool);self.f=torch.rand(6,23,2);self.a=torch.zeros(6,23,dtype=torch.bool);self.p=torch.tensor([768.,432.])
    def call(self,kind='single',weight=.25):return retention_objective(self.x,self.g,self.v,self.f,self.a,self.p,kind,weight)
    def test_real_exact_primary(self):
        ref=objective(self.x[:,KP],self.g,self.v,self.p,'pixel_huber');new=self.call('real')
        self.assertTrue(torch.equal(ref,new));self.assertTrue(torch.equal(torch.autograd.grad(ref,self.x,retain_graph=True)[0],torch.autograd.grad(new,self.x)[0]))
    def test_aux_does_not_dilute_primary(self):
        self.a[:,1]=True;ref=objective(self.x[:,KP],self.g,self.v,self.p,'pixel_huber');new=self.call()
        oldgrad=torch.autograd.grad(ref,self.x,retain_graph=True)[0];newgrad=torch.autograd.grad(new,self.x)[0]
        self.assertTrue(torch.equal(oldgrad[:,KP],newgrad[:,KP]));self.assertGreater(float(newgrad[:,1].norm()),0)
    def test_forbidden_targets(self):
        self.a[:,7]=True
        with self.assertRaises(ValueError):self.call('dual')
        with self.assertRaises(ValueError):self.call('real')
        self.a[:]=False;self.a[:,KP[0]]=True
        with self.assertRaises(ValueError):self.call()
    def test_nan_unknown_aux_excluded(self):
        self.f[:]=float('nan');self.a[:,1]=True;self.f[:,1]=.2
        loss=self.call();loss.backward();self.assertTrue(torch.isfinite(loss) and torch.isfinite(self.x.grad).all())
if __name__=='__main__':unittest.main()
