import unittest
import torch
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi

class SpatialDetectionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)
    def test_roi_sampling_and_frozen_inputs(self):
        memory=torch.ones(2,256,7,9,requires_grad=True);boxes=torch.tensor([[[.5,.5,.2,.3]],[[.2,.3,.7,.8]]],requires_grad=True)
        roi,g=sample_predicted_roi(memory,boxes);self.assertEqual(roi.shape,(2,1,25,256));torch.testing.assert_close(roi,torch.ones_like(roi));self.assertFalse(roi.requires_grad)
    def test_exact_initial_and_parameters_matched(self):
        tokens=torch.randn(2,6,256,requires_grad=True);roi=torch.randn(2,6,25,256,requires_grad=True);g=torch.randn(2,256,requires_grad=True);boxes=.05+.9*torch.rand(2,6,4);logits=torch.randn(2,6)
        a=SpatialDetectionReadout('roi');b=SpatialDetectionReadout('global');b.load_state_dict(a.state_dict());self.assertEqual(sum(x.numel() for x in a.parameters()),sum(x.numel() for x in b.parameters()))
        for model in [a,b]:
            p=model(tokens,roi,g,boxes,logits);torch.testing.assert_close(p['instance_boxes_cxcywh'],boxes,atol=1e-6,rtol=0);torch.testing.assert_close(p['instance_objectness_logits'],logits,atol=0,rtol=0)
            (p['instance_boxes_cxcywh'].sum()+p['instance_objectness_logits'].sum()).backward()
        self.assertIsNone(tokens.grad);self.assertIsNone(roi.grad);self.assertIsNone(g.grad)
    def test_global_ignores_roi_roi_uses_it(self):
        t=torch.randn(1,2,256);r=torch.randn(1,2,25,256);g=torch.randn(1,256);b=torch.full((1,2,4),.3);l=torch.zeros(1,2)
        for mode in ['global','roi']:
            m=SpatialDetectionReadout(mode);torch.nn.init.normal_(m.residual[-1].weight,std=.1)
            p=m(t,r,g,b,l)['instance_boxes_cxcywh'];p2=m(t,r*0,g,b,l)['instance_boxes_cxcywh']
            if mode=='global':torch.testing.assert_close(p,p2,atol=0,rtol=0)
            else:self.assertGreater(float((p-p2).detach().abs().max()),1e-5)
    def test_spatial_sampling_translation(self):
        x=(torch.arange(40)+.5)/40;mem=x[None,None,None].expand(1,256,40,40)
        boxes=torch.tensor([[[.3,.5,.1,.1],[.7,.5,.1,.1]]]);roi,_=sample_predicted_roi(mem,boxes)
        torch.testing.assert_close(roi[0,:,12,0],torch.tensor([.3,.7]),atol=1e-6,rtol=0)

if __name__=='__main__':unittest.main()
