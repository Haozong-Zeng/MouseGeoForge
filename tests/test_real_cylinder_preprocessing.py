import sys,unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_real_cylinder_bottom import resize_image,resize_field,rotation_error


class RealCylinderPreprocessingTest(unittest.TestCase):
    def test_portrait_resize_and_padding(self):
        im=np.full((1200,1080,3),255,np.uint8)
        output,T,geometry=resize_image(im)
        self.assertEqual(output.shape,(768,704,3));self.assertEqual(geometry,(691,768,6,7))
        self.assertFalse(output[:,:6].any());self.assertFalse(output[:,-7:].any())
        self.assertTrue((output[:,6:-7]==255).all())

    def test_intrinsic_projection_commutes_with_image_transform(self):
        _,T,_=resize_image(np.zeros((1200,1080,3),np.uint8))
        K=np.array([[1790,0,540],[0,1780,600],[0,0,1.]])
        x=np.array([10.,-20.,250.]);p=K@x;p/=p[2]
        q=T@K@x;q/=q[2]
        np.testing.assert_allclose(q,T@p,atol=1e-10)

    def test_masks_stay_binary(self):
        m=np.zeros((1200,1080),np.uint8);m[300:600,300:600]=255
        _,_,g=resize_image(np.zeros((1200,1080,3),np.uint8))
        self.assertEqual(set(np.unique(resize_field(m,g))),{0,255})

    def test_camera_rotation_error(self):
        self.assertAlmostEqual(rotation_error(np.eye(3),np.eye(3)),0.)
        R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        self.assertAlmostEqual(rotation_error(R,np.eye(3)),90.)


if __name__=='__main__':unittest.main()
