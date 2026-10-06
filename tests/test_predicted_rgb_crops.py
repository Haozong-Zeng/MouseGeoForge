import unittest
import torch
from mousegeoforge.predicted_rgb_crops import predicted_rgb_crops,MEAN


class CropTests(unittest.TestCase):
    def test_full_and_padding(self):
        x=torch.ones(3,20,40);r=predicted_rgb_crops(x,torch.tensor([[.5,.5,1.,1.]]),32)
        self.assertEqual(r['bounds'].tolist(),[[0,0,40,20]]);self.assertEqual(r['content_bounds'].tolist(),[[0,8,32,24]])
        torch.testing.assert_close(r['images'][0,:,0,0],torch.tensor(MEAN));torch.testing.assert_close(r['images'][0,:,10,10],torch.ones(3))
    def test_order_and_immutable(self):
        x=torch.rand(3,20,40);old=x.clone();b=torch.tensor([[.3,.5,.3,.8],[.7,.5,.3,.8]])
        a=predicted_rgb_crops(x,b);c=predicted_rgb_crops(x,b.flip(0));torch.testing.assert_close(a['images'],c['images'].flip(0));torch.testing.assert_close(x,old)
    def test_invalid_abstains(self):
        r=predicted_rgb_crops(torch.rand(3,20,40),torch.tensor([[2.,2.,.1,.1],[.5,.5,-1.,.1],[float('nan'),0,1,1]]))
        self.assertFalse(r['valid'].any())
    def test_clipped(self):
        r=predicted_rgb_crops(torch.rand(3,20,40),torch.tensor([[0.,.5,1.,1.]]));self.assertEqual(r['bounds'].tolist(),[[0,0,20,20]])
    def test_gray_content_fixed_padding(self):
        x=torch.zeros(3,20,40);x[0]=1;r=predicted_rgb_crops(x,torch.tensor([[.5,.5,1.,1.]]),32,grayscale=True)
        torch.testing.assert_close(r['images'][0,:,10,10],torch.full((3,),.299));torch.testing.assert_close(r['images'][0,:,0,0],torch.tensor(MEAN))


if __name__=='__main__':unittest.main()
