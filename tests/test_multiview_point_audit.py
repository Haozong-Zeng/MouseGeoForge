import unittest
import numpy as np
from mousegeoforge.multiview_point_audit import project_point,triangulate_observations


class TestMultiviewAudit(unittest.TestCase):
    def setUp(self):
        self.K=np.repeat(np.array([[[700.,0,320],[0,700,240],[0,0,1]]]),3,axis=0)
        self.E=np.repeat(np.eye(4)[None],3,axis=0);self.E[:,0,3]=[-100,0,100];self.xyz=np.array([5.,8.,500.])
        self.uv=np.asarray([project_point(self.xyz,k,e)[0] for k,e in zip(self.K,self.E)])
    def test_exact_geometry(self):
        r=triangulate_observations(self.uv,self.K,self.E);self.assertTrue(r['valid']);np.testing.assert_allclose(r['xyz_mm'],self.xyz,atol=1e-7)
    def test_heldview_not_needed(self):
        r=triangulate_observations(self.uv[[0,2]],self.K[[0,2]],self.E[[0,2]]);p,_=project_point(r['xyz_mm'],self.K[1],self.E[1]);np.testing.assert_allclose(p,self.uv[1],atol=1e-7)
    def test_bad_geometry_rejected(self):
        r=triangulate_observations(np.repeat(self.uv[:1],2,axis=0),self.K[:2],np.repeat(self.E[:1],2,axis=0));self.assertFalse(r['valid'])
    def test_missing_rejected(self):
        uv=self.uv.copy();uv[0,0]=np.nan;self.assertFalse(triangulate_observations(uv,self.K,self.E)['valid'])


if __name__=='__main__':unittest.main()
