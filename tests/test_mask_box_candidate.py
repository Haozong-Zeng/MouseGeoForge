import unittest
import torch
from torch import nn
from mousegeoforge.mask_box_candidate import MaskBoxCandidate,tensor_mapping_sha256
from mousegeoforge.mask_reference_box import MaskReferenceBoxRefiner


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__();self.aggregator=nn.Linear(2,2);self.instance_pose_head=nn.Module()
        self.instance_pose_head.instance_decoder=nn.Identity();self.weight=nn.Parameter(torch.ones(1))


class MaskBoxCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);self.model=FakeBase().eval();self.refiner=MaskReferenceBoxRefiner()
        state=self.model.state_dict();self.candidate=MaskBoxCandidate(self.refiner,
            base_state_sha256=tensor_mapping_sha256({k:v for k,v in state.items() if not k.startswith('aggregator.')}),
            encoder_state_sha256=tensor_mapping_sha256({k:v for k,v in state.items() if k.startswith('aggregator.')}))
    def predictions(self,b=1,v=1,q=6):
        return dict(instance_boxes_cxcywh=torch.ones(b,v,q,4)*.2,instance_mask_logits=torch.full((b,v,q,8,10),-2.),extra=torch.rand(3))
    def test_disabled_needs_no_model_or_fields(self):
        p={'unrelated':object()};c=MaskBoxCandidate()
        with c.capture():self.assertIs(c(p),p)
        self.assertFalse(c._active)
    def test_batch_views_empty_masks_and_other_objects_unchanged(self):
        p=self.predictions(2,4);before=tensor_mapping_sha256(self.model.state_dict());flags=[x.requires_grad for x in self.model.parameters()]
        with self.candidate.capture(self.model,enabled=True):
            self.model.instance_pose_head.instance_decoder(torch.rand(8,6,256))
            out=self.candidate(p,(32,40),enabled=True)
            torch.testing.assert_close(out['instance_boxes_cxcywh'],p['instance_boxes_cxcywh'],atol=0,rtol=0)
            self.assertIs(out['extra'],p['extra']);self.assertIs(out['instance_mask_logits'],p['instance_mask_logits'])
            with self.assertRaises(RuntimeError):self.candidate(p,(32,40),enabled=True)
        self.assertEqual(tensor_mapping_sha256(self.model.state_dict()),before);self.assertEqual(flags,[x.requires_grad for x in self.model.parameters()])
        self.assertEqual(len(self.model.instance_pose_head.instance_decoder._forward_hooks),0)
    def test_mismatch_mode_and_reentrancy(self):
        self.model.train()
        with self.assertRaises(ValueError):
            with self.candidate.capture(self.model,enabled=True):pass
        self.model.eval()
        with self.candidate.capture(self.model,enabled=True):
            with self.assertRaises(RuntimeError):
                with self.candidate.capture(self.model,enabled=True):pass
        with torch.no_grad():self.model.weight.add_(1)
        with self.assertRaises(ValueError):
            with self.candidate.capture(self.model,enabled=True):pass
    def test_exception_cleanup_and_shape_rejection(self):
        with self.assertRaises(ValueError):
            with self.candidate.capture(self.model,enabled=True):
                self.model.instance_pose_head.instance_decoder(torch.rand(1,5,256))
                self.candidate(self.predictions(),(32,40),enabled=True)
        self.assertFalse(self.candidate._active);self.assertEqual(len(self.model.instance_pose_head.instance_decoder._forward_hooks),0)
    def test_empty_queries(self):
        with self.candidate.capture(self.model,enabled=True):
            self.model.instance_pose_head.instance_decoder(torch.empty(1,0,256))
            out=self.candidate(self.predictions(q=0),(32,40),enabled=True)
            self.assertEqual(out['instance_boxes_cxcywh'].shape,(1,1,0,4))


if __name__=='__main__':unittest.main()
