import unittest
import numpy as np
from mousegeoforge.epipolar_candidate_readout import probability_candidates,rerank_candidates
from mousegeoforge.multiview_camera_geometry import project_points


class EpipolarCandidateTests(unittest.TestCase):
    def setUp(self):
        k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.p=np.stack([k@np.column_stack((np.eye(3),[-x,0,0])) for x in [-1,0,1]])
        xy=np.stack([project_points(p,np.array([.2,.3,8.]))[0] for p in self.p])
        self.xy=np.stack([xy,xy+[0,90]],axis=1);self.weight=np.array([[.4,.6],[.9,.1],[.9,.1]])
        self.base=np.array([100.,100.])
    def run_case(self,**kw):
        return rerank_candidates(self.p,self.xy,self.weight,[1,1,1],[0,1,2],0,self.base,**kw)
    def test_default_off_exact_object(self):self.assertIs(self.run_case()['xy'],self.base)
    def test_geometry_selects_supported_second_mass(self):
        r=self.run_case(enabled=True);self.assertEqual(r['choice'],0)
        self.assertEqual(self.run_case(enabled=True,geometry_weight=0)['choice'],1)
    def test_missing_and_duplicate_fallback(self):
        for valid,views in [([1,1,0],[0,1,2]),([1,1,1],[0,1,1]),([1,0,0],[0,1,2])]:
            r=rerank_candidates(self.p,self.xy,self.weight,valid,views,0,self.base,enabled=True)
            self.assertIs(r['xy'],self.base);self.assertFalse(r['active'])
    def test_duplicate_invariance(self):
        ids=[0,1,2,1];r=rerank_candidates(self.p[ids],self.xy[ids],self.weight[ids],[1]*4,ids,0,self.base,enabled=True)
        np.testing.assert_array_equal(r['xy'],self.run_case(enabled=True)['xy'])
    def test_permutation(self):
        ids=[2,0,1];r=rerank_candidates(self.p[ids],self.xy[ids],self.weight[ids],[1]*3,ids,1,self.base,enabled=True)
        np.testing.assert_array_equal(r['xy'],self.run_case(enabled=True)['xy'])
    def test_invalid_support_safe(self):
        xy=self.xy.copy();xy[2]=np.nan
        r=rerank_candidates(self.p,xy,self.weight,[1,1,0],[0,1,2],0,self.base,enabled=True)
        self.assertIs(r['xy'],self.base)
    def test_candidates_local_mean(self):
        p=np.zeros((8,8));p[2,3]=.75;p[2,4]=.25
        xy,m=probability_candidates(p,[80,80],count=1)
        np.testing.assert_allclose(xy[0],[37.5,25]);self.assertEqual(m[0],1)
    def test_single_view(self):
        r=rerank_candidates(self.p[:1],self.xy[:1],self.weight[:1],[1],[0],0,self.base,enabled=True)
        self.assertIs(r['xy'],self.base)


if __name__=='__main__':unittest.main()
