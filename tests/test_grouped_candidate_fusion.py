import unittest
import numpy as np
import torch
from torch import nn
from mousegeoforge.models.grouped_candidate_fusion import GroupedCandidateFusion
from mousegeoforge.candidate_selector_features import KP


class Probe(nn.Module):
    def forward(self,p,projections,views,pixels,local_enabled=False,fusion_enabled=False):
        xy=p['xy'].clone();chosen=p['objectness_logits'].sigmoid()>=.5
        assert (chosen.sum(-1)==1).all()
        mask=chosen[...,None].expand(-1,-1,5)
        xy[:,:,KP]+=mask[...,None]*((.01 if local_enabled else 0)+(.02 if fusion_enabled else 0))
        return dict(xy=xy,local_eligible=mask&local_enabled,fusion_eligible=mask&fusion_enabled)


class GroupedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.p=dict(xy=torch.rand(4,6,23,2),objectness_logits=torch.tensor([[4,3,-5,-5,-5,-5]]*4).float())
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.projections=np.stack([k@np.column_stack((np.eye(3),[-x,0,0])) for x in range(4)])
        self.views=[10,20,30,40];self.groups=[[(10,0),(20,1),(30,0)]]
        self.model=GroupedCandidateFusion(Probe())

    def call(self,**kwargs):
        args=dict(groups=self.groups,projections=self.projections,physical_views=self.views,pixels=[640,480],enabled=True)
        args.update(kwargs);return self.model(self.p,**args)

    def test_off_and_no_groups_exact(self):
        self.assertIs(self.model(self.p)['xy'],self.p['xy'])
        torch.testing.assert_close(self.call(groups=[])['xy'],self.p['xy'],atol=0,rtol=0)

    def test_selected_only_no_mutation(self):
        before={k:x.clone() for k,x in self.p.items()};out=self.call(local_enabled=True)
        self.assertEqual(int(out['changed'].sum()),15)
        for k,x in self.p.items():torch.testing.assert_close(x,before[k],atol=0,rtol=0)
        other=[i for i in range(23) if i not in KP]
        torch.testing.assert_close(out['xy'][:,:,other],self.p['xy'][:,:,other],atol=0,rtol=0)
        self.assertFalse(out['changed'][3].any())

    def test_camera_order_and_query_permutation(self):
        expected=self.call()['xy'];perm=[3,1,0,2];qp=[1,0,2,3,4,5]
        p={k:x[perm][:,qp] for k,x in self.p.items()}
        groups=[[(v,qp.index(q)) for v,q in group] for group in self.groups]
        out=self.model(p,groups,self.projections[perm],[self.views[i] for i in perm],[640,480],enabled=True)
        restored=out['xy'][[perm.index(i) for i in range(4)]][:,[qp.index(i) for i in range(6)]]
        torch.testing.assert_close(restored,expected,atol=0,rtol=0)

    def test_short_and_duplicate_views(self):
        for group in [[(10,0)],[(10,0),(20,1)],[(10,0),(10,1),(30,0)]]:
            torch.testing.assert_close(self.call(groups=[group])['xy'],self.p['xy'],atol=0,rtol=0)
        out=self.call(physical_views=[10,20,30,30])
        torch.testing.assert_close(out['xy'],self.p['xy'],atol=0,rtol=0)

    def test_bad_calibration_and_low_confidence(self):
        for overrides in [dict(projections=None),dict(view_valid=[False]*4)]:
            torch.testing.assert_close(self.call(**overrides)['xy'],self.p['xy'],atol=0,rtol=0)
        bad=self.projections.copy();bad[1]=np.nan
        torch.testing.assert_close(self.call(projections=bad)['xy'],self.p['xy'],atol=0,rtol=0)
        self.p['objectness_logits'][1,1]=-5
        torch.testing.assert_close(self.call()['xy'],self.p['xy'],atol=0,rtol=0)

    def test_overlap_order_rejected(self):
        groups=self.groups+[[(10,0),(20,0),(40,0)]]
        for gs in [groups,list(reversed(groups))]:
            out=self.call(groups=gs);self.assertEqual(len(out['rejected_groups']),2)
            torch.testing.assert_close(out['xy'],self.p['xy'],atol=0,rtol=0)

    def test_disjoint_groups_and_frozen_output(self):
        groups=self.groups+[[(10,1),(20,0),(30,1),(40,1)]]
        a=self.call(groups=groups);b=self.call(groups=list(reversed(groups)))
        torch.testing.assert_close(a['xy'],b['xy'],atol=0,rtol=0)
        self.assertEqual(int(a['changed'].sum()),35);self.assertFalse(a['xy'].requires_grad)

if __name__=='__main__':unittest.main()
