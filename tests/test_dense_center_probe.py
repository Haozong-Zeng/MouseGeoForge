import unittest
import torch
from mousegeoforge.dense_center_probe import DenseCenterProbe,center_targets,center_loss,decode_centers,candidate_metrics


class CenterProbeTests(unittest.TestCase):
    def data(self):return torch.tensor([[[.2,.3,.15,.2],[.75,.7,.2,.2]]]),torch.ones(1,2,dtype=torch.bool)

    def test_target_decode_roundtrip(self):
        b,v=self.data();t=center_targets(b,v,(8,12))
        pred=dict(heat_logits=torch.where(t['mask'],torch.tensor(10.),torch.tensor(-10.)),offset=t['offset'],size=t['size'])
        d=decode_centers(pred)
        m=candidate_metrics(d['boxes'][0],d['scores'][0],b[0]);self.assertEqual(m['tp'],2);self.assertEqual(m['fp'],0)
        self.assertEqual(t['near_center_pairs'],0)

    def test_collision_explicit(self):
        b,v=self.data();b[0,1,:2]=b[0,0,:2]
        with self.assertRaises(ValueError):center_targets(b,v,(8,12))

    def test_near_cells_reported(self):
        b,v=self.data();b[0,0,:2]=torch.tensor([.21,.3]);b[0,1,:2]=torch.tensor([.30,.3])
        self.assertEqual(center_targets(b,v,(8,12))['near_center_pairs'],1)

    def test_backward_views(self):
        b,v=self.data()
        for views in (1,2,4):
            m=DenseCenterProbe(16,8);t=center_targets(b.expand(views,-1,-1),v.expand(views,-1),(8,12))
            loss,_=center_loss(m(torch.randn(views,16,8,12)),t);loss.backward()
            for p in m.parameters():self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)

    def test_empty_target(self):
        b,v=self.data();v[:]=False;t=center_targets(b,v,(8,12));m=DenseCenterProbe(16,8)
        loss,_=center_loss(m(torch.randn(1,16,8,12)),t);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(candidate_metrics(torch.zeros(6,4),torch.ones(6),b[0,:0])['fp'],6)

    def test_duplicate_and_score_gate(self):
        b,v=self.data();boxes=torch.stack((b[0,0],b[0,0],b[0,1]))
        m=candidate_metrics(boxes,torch.tensor([.9,.8,.1]),b[0])
        self.assertEqual(m['tp'],1);self.assertEqual(m['fp'],1);self.assertEqual(m['all_topk_covered'],2);self.assertTrue(m['duplicate'])


if __name__=='__main__':unittest.main()
