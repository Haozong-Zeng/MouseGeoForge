import unittest
import numpy as np
from mousegeoforge.multiview_camera_geometry import project_points
from mousegeoforge.prediction_only_triangulation import leave_one_out_projection


class PredictionTriangulationTests(unittest.TestCase):
    def setUp(self):
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.p=np.stack([k@np.column_stack((np.eye(3),[-x,0,0])) for x in [-2,-1,0,1,2]])
        self.xyz=np.array([.2,.3,8.]);self.xy=np.stack([project_points(p,self.xyz)[0] for p in self.p])
    def run_case(self,xy=None,**kw):
        return leave_one_out_projection(self.p,self.xy if xy is None else xy,np.ones(5),np.ones(5,bool),np.arange(5),0,**kw)
    def test_exact(self):
        r=self.run_case();np.testing.assert_allclose(r['uv'],self.xy[0],atol=1e-8)
        self.assertNotIn(0,r['support_indices'])
    def test_target_cannot_leak(self):
        xy=self.xy.copy();xy[0]=[np.nan,np.nan]
        np.testing.assert_allclose(self.run_case(xy)['uv'],self.run_case()['uv'],atol=0)
    def test_one_outlier(self):
        xy=self.xy.copy();xy[4]+=[180,90]
        r=self.run_case(xy);np.testing.assert_allclose(r['uv'],self.xy[0],atol=1e-8)
        self.assertNotIn(4,r['support_indices'])
    def test_single_and_two_support_robust_abstain(self):
        for valid in [[1,0,0,0,0],[1,1,1,0,0]]:
            self.assertIsNone(leave_one_out_projection(self.p,self.xy,np.ones(5),valid,np.arange(5),0))
    def test_physical_duplicates_not_extra_evidence(self):
        self.assertIsNone(leave_one_out_projection(self.p,self.xy,np.ones(5),np.ones(5,bool),[0,1,1,2,2],0))
    def test_permutation(self):
        order=[3,0,4,2,1]
        r=leave_one_out_projection(self.p[order],self.xy[order],np.ones(5),np.ones(5,bool),np.arange(5)[order],1)
        np.testing.assert_allclose(r['uv'],self.run_case()['uv'],atol=1e-8)
    def test_coincident_abstain(self):
        self.assertIsNone(leave_one_out_projection(np.repeat(self.p[:1],5,axis=0),np.repeat(self.xy[:1],5,axis=0),np.ones(5),np.ones(5,bool),np.arange(5),0))


if __name__=='__main__':unittest.main()
