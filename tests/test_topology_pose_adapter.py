import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vggt-omega'))
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.models.topology_pose_adapter import TopologyPoseAdapter,instance_semantic_mask


class TopologyAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.base=DecoderSpatialPoseReadout().eval()
        # Test gradients with nonzero readout weights, like the trained baseline.
        torch.nn.init.normal_(self.base.spatial_readout.query[-1].weight,std=.02)
        self.model=TopologyPoseAdapter(self.base).eval()
        self.args=(torch.randn(1,6,256),torch.randn(1,256,4,5),torch.rand(1,6,4)*.5+.1)

    def test_mask_rules(self):
        m=instance_semantic_mask(6,23)
        self.assertFalse(m[0,1]);self.assertFalse(m[0,23]);self.assertTrue(m[0,24]);self.assertFalse(m.diag().any())
        self.assertFalse(instance_semantic_mask(1,23).any())

    def test_initial_and_off_exact(self):
        with torch.no_grad():
            a=self.base(*self.args);b=self.model(*self.args);c=self.model(*self.args,enabled=True)
        torch.testing.assert_close(a['xy'],b['xy'],atol=0,rtol=0)
        torch.testing.assert_close(a['xy'],c['xy'],atol=0,rtol=0)

    def test_equal_parameters(self):
        other=TopologyPoseAdapter(self.base,'full')
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),sum(p.numel() for p in other.parameters() if p.requires_grad))

    def test_gradient_and_frozen_off(self):
        before=self.model(*self.args)['xy'].clone()
        out=self.model(*self.args,enabled=True);out['xy'][...,0].sum().backward()
        self.assertGreater(float(self.model.residual.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in self.model.baseline.parameters()))
        with torch.no_grad():self.model.residual.weight.add_(.001*self.model.residual.weight.grad)
        torch.testing.assert_close(self.model(*self.args)['xy'],before,atol=0,rtol=0)

    def test_query_permutation_equivariance(self):
        torch.nn.init.normal_(self.model.residual.weight,std=.01)
        perm=torch.tensor([3,0,5,1,4,2]);tokens,memory,boxes=self.args
        with torch.no_grad():
            a=self.model(*self.args,enabled=True)['xy'];b=self.model(tokens[:,perm],memory,boxes[:,perm],enabled=True)['xy'][:,perm.argsort()]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=0)

    def test_untargeted_coordinates_remain_frozen(self):
        torch.nn.init.normal_(self.model.residual.weight,std=.01)
        keep=[k for k in range(23) if k not in [0,14,15,16,17]]
        with torch.no_grad():
            a=self.model(*self.args)['xy'];b=self.model(*self.args,enabled=True)['xy']
        torch.testing.assert_close(a[:,:,keep],b[:,:,keep],atol=0,rtol=0)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):TopologyPoseAdapter(self.base,'unknown')


if __name__=='__main__':unittest.main()
