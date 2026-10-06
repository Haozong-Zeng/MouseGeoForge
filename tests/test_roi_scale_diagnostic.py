import unittest
import numpy as np
from mousegeoforge.roi_scale_diagnostic import body_roi_affine,warp_rgb_masks,inverse_points,mask_query_iou
from mousegeoforge.affine_augmentation import transform_xy


class TestROI(unittest.TestCase):
    def test_inverse_and_projection(self):
        xy=np.stack([np.linspace(10,40,23),np.linspace(20,70,23)],-1);v=np.ones(23,bool)
        a,meta=body_roi_affine(xy,v,(432,768))
        self.assertLessEqual(meta['scale'],3)
        np.testing.assert_allclose(inverse_points(transform_xy(xy,a),a),xy,atol=1e-12)
        k=np.array([[800.,0,384],[0,800,216],[0,0,1.]])
        xyz=np.array([[.1,.2,2],[.4,-.2,3]]);p=xyz@k.T;p=p[:,:2]/p[:,2:]
        p2=xyz@(a@k).T;p2=p2[:,:2]/p2[:,2:]
        np.testing.assert_allclose(transform_xy(p,a),p2,atol=1e-12)

    def test_multimask_identity_and_query_permutation(self):
        rgb=np.zeros((80,100,3),np.float32);m=np.zeros((2,80,100),bool)
        m[0,10:20,20:30]=1;m[1,50:70,60:80]=1;rgb[:,:,0]=m[0];rgb[:,:,1]=m[1]
        out,mask,pad=warp_rgb_masks(rgb,m,np.eye(3))
        np.testing.assert_array_equal(out,rgb);np.testing.assert_array_equal(mask,m);self.assertEqual(pad,0)
        q,iou=mask_query_iou(mask[::-1],m[0]);self.assertEqual(q,1);self.assertEqual(iou[q],1)
        self.assertFalse(np.shares_memory(out,rgb));self.assertFalse(np.shares_memory(mask,m))

    def test_tail_exclusion_and_all_body_retained(self):
        xy=np.ones((23,2))*[300,200];xy[0]=[260,160];xy[22]=[350,250];v=np.ones(23,bool)
        a,_=body_roi_affine(xy,v,(432,768));xy[7:14]=1e6
        b,_=body_roi_affine(xy,v,(432,768));np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):body_roi_affine(xy,np.zeros(23,bool),(432,768))


if __name__=='__main__':unittest.main()
