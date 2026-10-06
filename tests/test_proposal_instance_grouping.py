import inspect,unittest
import torch
from mousegeoforge.models.local_point_evidence_probe import LocalPointEvidenceProbe
from mousegeoforge.models.proposal_instance_grouping import ProposalInstanceGrouping,proposal_grouping_loss,decode_proposal_grouped

class ProposalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)
    def evidence(self,flat=False):
        features=torch.randn(32,20,24,requires_grad=True);e=LocalPointEvidenceProbe()(features,(72,88));e['features']=features
        e['logits']=(torch.zeros(5,20,24) if flat else torch.randn(5,20,24)).masked_fill(~e['spatial_valid'],-torch.inf)
        c={k:v for k,v in e.items() if k!='features'};c['logits']=(torch.zeros(1,20,24) if flat else torch.randn(1,20,24)).masked_fill(~e['spatial_valid'],-torch.inf)
        return e,c
    def output(self,flat=False,interaction=False):
        e,c=self.evidence(flat);m=ProposalInstanceGrouping(interaction=interaction);return m(e,c,(72,88)),e,m
    def target(self):return torch.rand(2,5,2),torch.ones(2,5,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.tensor([[.3,.4],[.7,.6]]),torch.ones(2,dtype=torch.bool),(72,88)
    def test_interface(self):self.assertEqual(list(inspect.signature(ProposalInstanceGrouping.forward).parameters),['self','evidence','center_evidence','image_hw'])
    def test_parameters_match(self):
        a=ProposalInstanceGrouping();b=ProposalInstanceGrouping(interaction=True);b.load_state_dict(a.state_dict());self.assertEqual(sum(x.numel() for x in a.parameters()),sum(x.numel() for x in b.parameters()))
    def test_loss_gradient(self):
        for interaction in [False,True]:
            out,e,m=self.output(interaction=interaction);loss=proposal_grouping_loss(out,*self.target())['total'];self.assertTrue(torch.isfinite(loss));loss.backward();self.assertGreater(float(e['features'].grad.abs().sum()),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_unknown_nan_no_assignment_loss(self):
        out,_,_=self.output();args=list(self.target());args[0][:]=torch.nan;args[1][:]=False;loss=proposal_grouping_loss(out,*args);self.assertEqual(float(loss['assignment']),0);self.assertTrue(torch.isfinite(loss['total']))
    def test_instance_permutation(self):
        out,_,_=self.output();args=self.target();other=tuple(x.flip(0) for x in args[:-1])+(args[-1],);torch.testing.assert_close(proposal_grouping_loss(out,*args)['total'],proposal_grouping_loss(out,*other)['total'])
    def test_flat_empty_proposals_finite(self):
        out,e,_=self.output(flat=True);self.assertFalse(out['center_available'].any());loss=proposal_grouping_loss(out,*self.target());self.assertTrue(torch.isfinite(loss['total']));self.assertEqual(float(loss['missing_center_proposals']),2);loss['total'].backward();self.assertFalse(decode_proposal_grouped(out)['instance_active'].any())
    def test_unavailable_proposals_ignored(self):
        out,_,_=self.output();out['center_available'][:]=False;out['object_logits'][:]=100;self.assertFalse(decode_proposal_grouped(out)['instance_active'].any())
    def test_no_forced_count_and_max_capacity(self):
        out,_,_=self.output();out['object_logits']=torch.full_like(out['object_logits'],-10);self.assertEqual(int(decode_proposal_grouped(out)['instance_active'].sum()),0)
        out['object_logits'][:]=10;self.assertEqual(int(decode_proposal_grouped(out)['instance_active'].sum()),2)
    def test_one_positive_per_gt_and_two_close_mice(self):
        out,_,_=self.output();args=list(self.target());args[3]=torch.tensor([[.4,.4],[.54,.4]])
        out['centers']=torch.tensor([[.4,.4],[.54,.4],[.41,.4],[.55,.4],[0.,0.],[.9,.9],[.9,.1],[.1,.9]])
        out['center_available'][:]=True;r=proposal_grouping_loss(out,*args);self.assertEqual(r['matched_queries'].tolist(),[0,1]);self.assertEqual(float(r['missing_center_proposals']),0)
    def test_far_no_fake_gt_insertion(self):
        out,_,_=self.output();args=list(self.target());args[3][:]=.99;out['centers'][:]=0;r=proposal_grouping_loss(out,*args);self.assertEqual(len(r['matched_queries']),0);self.assertEqual(float(r['missing_center_proposals']),2)
    def test_unknown_instance_nan_and_incomplete(self):
        out,_,_=self.output();args=list(self.target());args[2][:]=False;args[4][:]=False;args[3][:]=torch.nan;args[1][:]=False;args[0][:]=torch.nan
        r=proposal_grouping_loss(out,*args,annotation_complete=False);self.assertEqual(float(r['total']),0)
    def test_padding_and_flat_points_null(self):
        e,c=self.evidence();e['logits'][:]=0;out=ProposalInstanceGrouping()(e,c,(72,88));self.assertFalse(out['candidate_available'].any());out['object_logits'][:]=10;d=decode_proposal_grouped(out);self.assertFalse(d['accepted'].any())
        xy=out['centers'][out['center_available']];self.assertTrue(((xy>=0)&(xy<1)).all())

if __name__=='__main__':unittest.main()
