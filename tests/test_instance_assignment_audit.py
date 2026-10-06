import unittest
import torch
import torch.nn.functional as F
from mousegeoforge.instance_assignment_audit import matching_costs,assignment,assignment_gap,objectness_terms
from training.instance_pose_losses import hungarian_match_instances


class AssignmentAuditTests(unittest.TestCase):
    def fixture(self):
        gen=torch.Generator().manual_seed(101)
        p={'instance_objectness_logits':torch.randn(1,2,6,generator=gen),
           'instance_boxes_cxcywh':torch.rand(1,2,6,4,generator=gen),
           'instance_keypoint_xy':torch.rand(1,2,6,23,2,generator=gen),
           'instance_mask_logits':torch.randn(1,2,6,3,4,generator=gen)}
        t={'instance_valid':torch.ones(1,2,2,dtype=torch.bool),
           'boxes_cxcywh':torch.rand(1,2,2,4,generator=gen),
           'box_valid':torch.tensor([[[True,False],[True,True]]]),
           'keypoint_xy':torch.rand(1,2,2,23,2,generator=gen),
           'keypoint_valid':torch.rand(1,2,2,23,generator=gen)>.3,
           'masks':(torch.rand(1,2,2,6,8,generator=gen)>.5).float(),
           'mask_valid':torch.tensor([[[True,False],[True,True]]])}
        return p,t

    def test_original_matches(self):
        p,t=self.fixture()
        for v,(q,m) in enumerate(hungarian_match_instances(p,t)[0]):
            ids,c=matching_costs(p,t,0,v)
            self.assertEqual(assignment(c['original'],ids),{int(b):int(a) for a,b in zip(q,m)})
            self.assertGreaterEqual(assignment_gap(c['original']),0)

    def test_mask_cost_exact_formula(self):
        p,t=self.fixture();ids,c=matching_costs(p,t,0,0)
        y=F.interpolate(t['masks'][0,0,0][None,None],size=(3,4),mode='nearest')[0,0]
        for q in range(6):
            x=p['instance_mask_logits'][0,0,q]
            torch.testing.assert_close(c['mask_bce'][0,q],F.binary_cross_entropy_with_logits(x,y))
            dice=1-(2*(x.sigmoid()*y).sum()+1)/(x.sigmoid().sum()+y.sum()+1)
            torch.testing.assert_close(c['mask_dice'][0,q],dice)
        self.assertEqual(c['mask_bce'][1].abs().sum(),0)

    def test_missing_masks_identity(self):
        p,t=self.fixture();del t['masks'];ids,c=matching_costs(p,t,0,0)
        torch.testing.assert_close(c['original'],c['mask_augmented'],atol=0,rtol=0)

    def test_empty_targets(self):
        p,t=self.fixture();t['instance_valid'].zero_();ids,c=matching_costs(p,t,0,0)
        self.assertEqual(assignment(c['original'],ids),{})
        self.assertIsNone(assignment_gap(c['original']))

    def test_negative_gradients_and_partial_annotations(self):
        p,t=self.fixture();x=p['instance_objectness_logits'].clone().requires_grad_()
        matches=hungarian_match_instances(p,t);complete=torch.tensor([[True,False]])
        y,w,g=objectness_terms(x,matches,complete)
        loss=(F.binary_cross_entropy_with_logits(x,y,reduction='none')*w).sum()/w.sum().clamp_min(1)
        loss.backward();torch.testing.assert_close(x.grad,g)
        self.assertTrue((g[0,0][y[0,0]==0]>0).all())
        self.assertTrue((g[0,1][y[0,1]==0]==0).all())


if __name__=='__main__':unittest.main()
