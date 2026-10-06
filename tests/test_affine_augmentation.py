import unittest
import numpy as np
from mousegeoforge.affine_augmentation import centered_affine,transform_xy,warp_view


class TestAffineAugmentation(unittest.TestCase):
    def sample(self):
        rgb=np.zeros((12,16,3),np.uint8);rgb[3,4]=[255,0,0];rgb[8,12]=[0,255,0]
        masks=np.zeros((2,12,16),bool);masks[0,3,4]=True;masks[1,8,12]=True
        return dict(rgb=rgb,depth=np.full((12,16),123.5,np.float32),depth_valid=np.ones((12,16),bool),
                    masks=masks,xy=np.array([[[4.,3.],[0.,0.]],[[12.,8.],[15.,11.]]]),
                    point_valid=np.array([[True,False],[True,True]]),boxes_xyxy=np.array([[3.,2.,6.,5.],[11.,7.,16.,12.]]),
                    intrinsic=np.array([[100.,0.,8.],[0.,100.,6.],[0.,0.,1.]]),extrinsic=np.eye(4)[:3])

    def test_identity_and_no_mutation(self):
        s=self.sample();before={k:v.copy() for k,v in s.items()};r=warp_view(**s,affine=centered_affine(12,16))
        for k in s:
            np.testing.assert_array_equal(s[k],before[k]);np.testing.assert_array_equal(r[k],s[k])
            self.assertFalse(np.shares_memory(s[k],r[k]))

    def test_translation_multimouse_alignment_clipping_depth_units(self):
        s=self.sample();r=warp_view(**s,affine=centered_affine(12,16,dx=3,dy=-2))
        np.testing.assert_array_equal(r['rgb'][1,7],[255,0,0]);self.assertTrue(r['masks'][0,1,7])
        np.testing.assert_array_equal(r['rgb'][6,15],[0,255,0]);self.assertTrue(r['masks'][1,6,15])
        np.testing.assert_array_equal(r['point_valid'],[[True,False],[True,False]])
        self.assertFalse(r['depth_valid'][:,:3].any());self.assertFalse(r['depth_valid'][-2:].any())
        self.assertTrue((r['depth'][r['depth_valid']]==123.5).all());self.assertTrue((r['depth'][~r['depth_valid']]==0).all())
        np.testing.assert_array_equal(r['extrinsic'],s['extrinsic']);np.testing.assert_array_equal(r['boxes_xyxy'][1],[14,5,16,10])

    def test_projection_commutes_scale_and_translation(self):
        s=self.sample();a=centered_affine(12,16,scale=.9,dx=2,dy=-1);r=warp_view(**s,affine=a)
        xyz=np.array([[.1,.2,5.],[-.2,.1,4.],[.2,-.1,6.]])
        def project(k):
            uv=xyz@k.T;return uv[:,:2]/uv[:,2:]
        np.testing.assert_allclose(project(r['intrinsic']),transform_xy(project(s['intrinsic']),a),atol=1e-12)
        np.testing.assert_array_equal(r['extrinsic'],s['extrinsic'])
        self.assertTrue((r['depth'][r['depth_valid']]==123.5).all())

    def test_invalid_depth_and_invalid_transform(self):
        s=self.sample();s['depth'][0,0]=np.nan;s['depth'][1,1]=-1;s['depth_valid'][2,2]=False
        r=warp_view(**s,affine=centered_affine(12,16))
        for p in [(0,0),(1,1),(2,2)]:self.assertFalse(r['depth_valid'][p]);self.assertEqual(r['depth'][p],0)
        with self.assertRaises(ValueError):centered_affine(12,16,scale=-1)
        with self.assertRaises(ValueError):warp_view(**s,affine=np.array([[1,1,0],[0,1,0],[0,0,1.]]))


if __name__=='__main__':unittest.main()
