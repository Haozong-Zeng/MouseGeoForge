import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vggt-omega'))
import torch
from mousegeoforge.keypoint_breakdown import keypoint_rows,summarize_joint_rows


class KeypointBreakdownTests(unittest.TestCase):
    def sample(self):
        box=torch.tensor([[[[.5,.5,.5,.5]]]])
        xy=torch.full((1,1,1,23,2),.5)
        target=dict(boxes_cxcywh=box,keypoint_xy=xy,keypoint_valid=torch.ones(1,1,1,23,dtype=torch.bool),keypoint_visible=torch.ones(1,1,1,23,dtype=torch.bool))
        pred=dict(instance_boxes_cxcywh=box,instance_keypoint_xy=xy.clone(),instance_objectness_logits=torch.ones(1,1,1))
        return pred,dict(images=torch.zeros(1,1,3,100,100)),target

    def test_correct_and_missed_detection(self):
        p,b,t=self.sample();r=keypoint_rows(p,b,t,[([0],[0])])
        self.assertEqual(summarize_joint_rows(r)['detection_pck'],1.)
        p['instance_objectness_logits'].fill_(-1)
        s=summarize_joint_rows(keypoint_rows(p,b,t,[([0],[0])]))
        self.assertEqual(s['pck'],1.);self.assertEqual(s['detection_pck'],0.)

    def test_normalization_and_validity(self):
        p,b,t=self.sample();p['instance_keypoint_xy'][...,0]+=.1
        t['keypoint_valid'][...,7:14]=False
        r=keypoint_rows(p,b,t,[([0],[0])])
        self.assertEqual(len(r),16)
        self.assertAlmostEqual(r[0]['nme'],.1/(.5**2+.5**2)**.5,places=6)

    def test_visibility_unknown_and_empty(self):
        p,b,t=self.sample();t['keypoint_visibility_valid']=torch.zeros_like(t['keypoint_valid'])
        self.assertIsNone(keypoint_rows(p,b,t,[([0],[0])])[0]['visible'])
        self.assertEqual(summarize_joint_rows([])['n'],0)


if __name__=='__main__':unittest.main()
