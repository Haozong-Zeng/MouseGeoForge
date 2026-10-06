import unittest
import numpy as np
from mousegeoforge.multiview_camera_geometry import projection_matrix,project_points,fundamental_from_projections,symmetric_epipolar_distance,weighted_dlt

class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[700.,0,320],[0,710,240],[0,0,1]]);self.e1=np.eye(4);self.e2=np.eye(4);self.e2[0,3]=-1.;self.p1=projection_matrix(self.k,self.e1);self.p2=projection_matrix(self.k,self.e2);self.xyz=np.array([[.1,.2,4.],[.5,-.3,6.]]);self.a,_=project_points(self.p1,self.xyz);self.b,_=project_points(self.p2,self.xyz)
    def test_projection_direction(self):
        np.testing.assert_allclose(self.a[0],[337.5,275.5]);self.assertLess(self.b[0,0],self.a[0,0])
    def test_epipolar_exact(self):
        f=fundamental_from_projections(self.p1,self.p2);self.assertLess(np.max(symmetric_epipolar_distance(f,self.a,self.b)),1e-8)
    def test_reverse_symmetric(self):
        noisy=self.b+np.array([3,5]);a=symmetric_epipolar_distance(fundamental_from_projections(self.p1,self.p2),self.a,noisy);b=symmetric_epipolar_distance(fundamental_from_projections(self.p2,self.p1),noisy,self.a);np.testing.assert_allclose(a,b,atol=1e-8)
    def test_resize_and_padding(self):
        affine=np.array([[.5,0,6],[0,.5,0],[0,0,1]]);p=projection_matrix(affine@self.k,self.e1);uv,_=project_points(p,self.xyz);np.testing.assert_allclose(uv,self.a*.5+[6,0])
    def test_world_gauge(self):
        transform=np.eye(4);transform[:3,:3]*=2;transform[:3,3]=[3,-1,5];points=self.xyz*2+transform[:3,3];a,_=project_points(self.p1@np.linalg.inv(transform),points);np.testing.assert_allclose(a,self.a)
    def test_dlt_exact_and_camera_scale(self):
        x=weighted_dlt([self.p1*7,self.p2*.03],[self.a[0],self.b[0]],[.2,1]);np.testing.assert_allclose(x,self.xyz[0],atol=1e-10)
    def test_degenerate_and_single(self):
        self.assertIsNone(weighted_dlt([self.p1],[self.a[0]]));self.assertIsNone(weighted_dlt([self.p1,self.p1],[self.a[0],self.a[0]]))
        with self.assertRaises(ValueError):fundamental_from_projections(self.p1,self.p1)
    def test_zero_weight_outlier(self):
        x=weighted_dlt([self.p1,self.p2,self.p1],[self.a[0],self.b[0],[1000,1000]],[1,1,0]);np.testing.assert_allclose(x,self.xyz[0],atol=1e-10)
    def test_nonfinite_projection_safe(self):
        uv,_=project_points(self.p1,np.array([[0.,0,0.]]));self.assertTrue(np.isnan(uv).all())

if __name__=='__main__':unittest.main()
