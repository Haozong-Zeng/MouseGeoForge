import unittest
import numpy as np
from mousegeoforge.association_evidence import pairwise_affinity_evidence,KP


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
        self.pmat=np.stack([self.k@np.column_stack([np.eye(3),[-x,0,0]]) for x in [0,1,2]])
        self.p=dict(xy=np.zeros((3,2,23,2)),boxes=np.ones((3,2,4)),objectness_logits=np.ones((3,2))*3,tokens=np.zeros((3,2,5,4)),candidate_memory=np.zeros((3,2,5,6,4)))
        for v in range(3):
            for q in range(2):
                for j,kp in enumerate(KP):
                    xyz=np.array([.2*j,q+.1*j,5.,1.]);uv=self.pmat[v]@xyz
                    self.p['xy'][v,q,kp]=uv[:2]/uv[2]/[640,480]
                self.p['tokens'][v,q,:,q]=1;self.p['candidate_memory'][v,q,:,:,q]=1
    def call(self,p=None,ids=None):
        ids=list(range(3)) if ids is None else ids;p=self.p if p is None else p
        return pairwise_affinity_evidence({k:x[ids] for k,x in p.items()},self.pmat[ids],ids,[640,480])
    def test_separation(self):
        r=self.call();self.assertAlmostEqual(r['evidence']['epipolar_mean_px'][0,2],0)
        self.assertGreater(r['evidence']['epipolar_mean_px'][0,3],100)
        self.assertEqual(r['evidence']['semantic_cosine'][0,2],1);self.assertEqual(r['evidence']['local_cosine'][0,3],0)
    def test_same_camera_invalid(self):
        r=self.call();self.assertTrue(all(np.isnan(x[0,1]) for x in r['evidence'].values()))
    def test_view_order_duplicates(self):
        a=self.call();b=self.call(ids=[2,0,1,2])
        for k in a['evidence']:np.testing.assert_allclose(a['evidence'][k],b['evidence'][k],equal_nan=True)
    def test_query_permutation(self):
        p={k:x.copy() for k,x in self.p.items()}
        for x in p.values():x[1]=x[1,::-1]
        a=self.call();b=self.call(p);perm=[0,1,3,2,4,5]
        for k in a['evidence']:np.testing.assert_allclose(a['evidence'][k],b['evidence'][k][np.ix_(perm,perm)],equal_nan=True)
    def test_empty_and_single(self):
        p={k:x.copy() for k,x in self.p.items()};p['objectness_logits'][:]=-20
        self.assertEqual(self.call(p)['nodes'],[])
        self.assertTrue(np.isnan(self.call(ids=[0])['evidence']['epipolar_mean_px']).all())
    def test_invalid_camera_does_not_fabricate_geometry(self):
        cameras=self.pmat.copy();cameras[1]=np.nan
        r=pairwise_affinity_evidence(self.p,cameras,[0,1,2],[640,480])
        self.assertTrue(np.isnan(r['evidence']['epipolar_mean_px'][0,2]));self.assertEqual(r['evidence']['semantic_cosine'][0,2],1)
    def test_pixel_and_duplicate_validation(self):
        with self.assertRaises(ValueError):pairwise_affinity_evidence(self.p,self.pmat,[0,1,2],[0,480])
        with self.assertRaises(ValueError):pairwise_affinity_evidence(self.p,self.pmat,[0,0,2],[640,480])


if __name__=='__main__':unittest.main()
