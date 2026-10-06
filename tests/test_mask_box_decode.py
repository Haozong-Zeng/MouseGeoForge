import unittest
import torch
from torch.nn import functional as F
from mousegeoforge.mask_box_decode import mask_boxes_from_logits


class MaskBoxDecodeTests(unittest.TestCase):
    def test_disabled_identity_without_mask(self):
        boxes=torch.rand(1,3,6,4,requires_grad=True)
        result,used=mask_boxes_from_logits(None,boxes,(8,12))
        self.assertIs(result,boxes);self.assertFalse(used.any())

    def test_exact_rectangle_and_empty_fallback(self):
        logits=torch.full((1,2,3,8,12),-1.);logits[0,0,0,2:6,3:9]=1
        boxes=torch.rand(1,2,3,4);before=boxes.clone()
        result,used=mask_boxes_from_logits(logits,boxes,(8,12),enabled=True)
        torch.testing.assert_close(result[0,0,0],torch.tensor([.5,.5,.5,.5]),atol=0,rtol=0)
        self.assertEqual(int(used.sum()),1);torch.testing.assert_close(result[~used],boxes[~used],atol=0,rtol=0)
        torch.testing.assert_close(before,boxes,atol=0,rtol=0)

    def test_exclusive_last_pixel_and_zero_threshold(self):
        logits=torch.zeros(1,4,7);logits[0,3,6]=1
        result,used=mask_boxes_from_logits(logits,torch.zeros(1,4),(4,7),enabled=True)
        torch.testing.assert_close(result[0],torch.tensor([6.5/7,3.5/4,1/7,1/4]))
        self.assertTrue(used.all())

    def test_disconnected_pixels_are_not_silently_removed(self):
        logits=torch.full((1,5,5),-1.);logits[0,0,0]=1;logits[0,4,4]=1
        result,_=mask_boxes_from_logits(logits,torch.zeros(1,4),(5,5),enabled=True)
        torch.testing.assert_close(result,torch.tensor([[.5,.5,1.,1.]]),atol=0,rtol=0)

    def test_resize_matches_existing_mask_evaluator(self):
        torch.manual_seed(12);logits=torch.randn(2,3,4);boxes=torch.rand(2,4)
        result,_=mask_boxes_from_logits(logits,boxes,(9,13),enabled=True)
        masks=F.interpolate(logits[:,None],size=(9,13),mode='bilinear',align_corners=False)[:,0]>0
        for i,mask in enumerate(masks):
            y,x=torch.where(mask);lo=torch.tensor([x.min()/13,y.min()/9]);hi=torch.tensor([(x.max()+1)/13,(y.max()+1)/9])
            torch.testing.assert_close(result[i],torch.cat(((lo+hi)/2,hi-lo)),atol=0,rtol=0)

    def test_query_permutation_and_single_view(self):
        torch.manual_seed(23);masks=torch.randn(1,1,6,4,4);boxes=torch.rand(1,1,6,4)
        a,used=mask_boxes_from_logits(masks,boxes,(8,8),enabled=True)
        b,other=mask_boxes_from_logits(masks.flip(2),boxes.flip(2),(8,8),enabled=True)
        torch.testing.assert_close(a.flip(2),b,atol=0,rtol=0);self.assertTrue(torch.equal(used.flip(2),other))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):mask_boxes_from_logits(None,torch.rand(2,4),(8,8),enabled=True)
        with self.assertRaises(ValueError):mask_boxes_from_logits(torch.zeros(2,4,4),torch.rand(3,4),(8,8),enabled=True)
        with self.assertRaises(ValueError):mask_boxes_from_logits(torch.full((2,4,4),float('nan')),torch.rand(2,4),(8,8),enabled=True)
        with self.assertRaises(ValueError):mask_boxes_from_logits(torch.zeros(2,4,4),torch.rand(2,4),(0,8),enabled=True)


if __name__=='__main__':unittest.main()
