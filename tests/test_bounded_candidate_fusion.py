import unittest
import numpy as np
import torch
from mousegeoforge.models.prior_candidate_selector import PriorCandidateSelector
from mousegeoforge.models.bounded_candidate_fusion import BoundedCandidateFusion
from mousegeoforge.candidate_selector_features import KP


class BoundedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17);v=4
        self.p=dict(xy=torch.rand(v,6,23,2),candidate_xy=torch.rand(v,6,5,6,2),candidate_mass=torch.ones(v,6,5,6),objectness_logits=torch.tensor([[4,-4,-4,-4,-4,-4]]*v).float(),tokens=torch.randn(v,6,5,256),candidate_memory=torch.randn(v,6,5,6,256),probability_entropy=torch.ones(v,6,5)*.5,probability_variance=torch.ones(v,6,5,2)*.01)
        self.p['candidate_xy'][:,:,:,0]=self.p['xy'][:,:,KP]
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.cameras=np.stack([k@np.column_stack((np.eye(3),[-x,0,0])) for x in range(v)])
        self.local=PriorCandidateSelector('unary');self.geo=PriorCandidateSelector('geometry');self.model=BoundedCandidateFusion(self.local,self.geo)

    def favor_nonbaseline(self,head):
        with torch.no_grad():
            head.score[0].weight.zero_();head.score[0].bias.zero_();head.score[0].weight[0,66]=-1;head.score[0].bias[0]=1
            head.score[-1].weight.zero_();head.score[-1].bias.zero_();head.score[-1].weight[0,0]=1

    def run_ids(self,ids,**kw):
        p={k:x[ids] for k,x in self.p.items()}
        return self.model(p,self.cameras[ids],ids,[640,480],**kw)

    def test_off_identity(self):self.assertIs(self.model(self.p)['xy'],self.p['xy'])

    def test_initial_on_exact(self):
        torch.testing.assert_close(self.run_ids([0,1,2,3],local_enabled=True,fusion_enabled=True)['xy'],self.p['xy'],atol=0,rtol=0)

    def test_v1_v2_and_missing_calibration_local_fallback(self):
        self.favor_nonbaseline(self.local)
        for ids in [[0],[0,1]]:
            a=self.run_ids(ids,local_enabled=True);b=self.run_ids(ids,local_enabled=True,fusion_enabled=True)
            self.assertTrue(a['changed'].any());self.assertFalse(b['fusion_eligible'].any());torch.testing.assert_close(a['xy'],b['xy'],atol=0,rtol=0)
        a=self.model(self.p,pixels=[640,480],local_enabled=True,fusion_enabled=True);b=self.run_ids([0,1,2,3],local_enabled=True)
        torch.testing.assert_close(a['xy'],b['xy'],atol=0,rtol=0)

    def test_geometry_baseline_choice_overrides_local(self):
        self.favor_nonbaseline(self.local)
        a=self.run_ids([0,1,2,3],local_enabled=True);b=self.run_ids([0,1,2,3],local_enabled=True,fusion_enabled=True)
        self.assertTrue(a['changed'].any());self.assertTrue(b['fusion_eligible'].any());torch.testing.assert_close(b['xy'],self.p['xy'],atol=0,rtol=0)

    def test_order_duplicate_and_other_points(self):
        self.favor_nonbaseline(self.local);self.favor_nonbaseline(self.geo)
        a=self.run_ids([0,1,2,3],local_enabled=True,fusion_enabled=True)
        b=self.run_ids([3,1,0,2],local_enabled=True,fusion_enabled=True)
        c=self.run_ids([0,1,2,3,1,2],local_enabled=True,fusion_enabled=True)
        torch.testing.assert_close(a['xy'],b['xy'][[2,1,3,0]],atol=0,rtol=0);torch.testing.assert_close(a['xy'],c['xy'][:4],atol=0,rtol=0)
        keep=[k for k in range(23) if k not in KP];torch.testing.assert_close(a['xy'][:,:,keep],self.p['xy'][:,:,keep],atol=0,rtol=0)

    def test_invalid_views_and_multiple_queries_abstain(self):
        self.favor_nonbaseline(self.local);self.favor_nonbaseline(self.geo)
        a=self.run_ids([0,1,2,3],local_enabled=True,fusion_enabled=True,view_valid=[False]*4)
        torch.testing.assert_close(a['xy'],self.p['xy'],atol=0,rtol=0)
        self.p['objectness_logits'][:,1]=4
        b=self.run_ids([0,1,2,3],local_enabled=True,fusion_enabled=True)
        torch.testing.assert_close(b['xy'],self.p['xy'],atol=0,rtol=0)

    def test_frozen_inference(self):
        self.assertFalse(any(p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.run_ids([0],local_enabled=True)['xy'].requires_grad)


if __name__=='__main__':unittest.main()
