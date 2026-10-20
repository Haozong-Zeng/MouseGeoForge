"""CPU geometry regression tests; no datasets, GPU, or model are needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import bottom_pose_frontier_20261019 as f
import bottom_pose_silhouette_data_20261019 as data

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.Ha=np.array([[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]])
        self.Hb=np.array([[-.5,-.25],[0.,-.25],[0.,.25],[-.5,.25]])
        self.I=np.eye(3);self.zero=np.zeros(3);self.C=np.array([1.,0.,0.])
        self.obs=[(self.Hb,self.I,[(self.Ha,self.I,self.I,self.zero)])]
    def test_true_tangencies(self):
        np.testing.assert_allclose(f.pair_residuals(self.Ha,self.I,self.I,self.zero,self.Hb,self.I,self.I,self.C),0.,atol=1e-12)
    def test_invalid_initial_pose_is_rejected(self):
        R=Rotation.from_euler('z',60,degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError,'Invalid initial'):
            f.fit(self.obs,R,np.array([0.,0.,1.]),np.array([0.,1.,0.]),'full')
    def test_zero_baseline_is_invalid(self):
        self.assertIsNone(f.pair_residuals(self.Ha,self.I,self.I,self.zero,self.Hb,self.I,self.I,self.zero))
    def test_invalid_trial_penalty_and_rejection(self):
        from types import SimpleNamespace
        def fake_ls(fun,x0,**kwargs):
            trial=x0.copy();trial[3:]=(np.array([0.,0.,1.])-self.C)*50.
            self.assertGreater(np.linalg.norm(fun(trial)),10.)
            return SimpleNamespace(x=trial,success=True,status=1,message='invalid trial fixture',nfev=1,njev=1,optimality=0.,jac=np.eye(6))
        with patch.object(f,'least_squares',fake_ls):
            R,C,_,after,details=f.fit(self.obs,self.I,self.C,np.array([0.,1.,0.]),'full',True)
        self.assertTrue(details['rejected_invalid_final']);self.assertFalse(details['success'])
        self.assertEqual(details['invalid_final_pairs'],0);np.testing.assert_allclose(C,self.C)
    def test_fixed_mode_has_five_parameters(self):
        *_,details=f.fit(self.obs,self.I,self.C,np.array([0.,1.,0.]),'azimuth_fixed',True)
        self.assertEqual(details['parameters'],5);self.assertEqual(details['invalid_final_pairs'],0)
    def test_epipolar_formula(self):
        rng=np.random.default_rng(17);Ra=Rotation.random(random_state=rng).as_matrix();Rb=Rotation.random(random_state=rng).as_matrix();Ca=rng.normal(size=3);Cb=rng.normal(size=3)
        K=np.array([[500.,0.,320.],[0.,480.,240.],[0.,0.,1.]])
        F=f.fundamental(K,Ra,Ca,K,Rb,Cb)
        for X in rng.normal(size=(20,3)):
            a=K@Ra@(X-Ca);b=K@Rb@(X-Cb)
            self.assertAlmostEqual(float(b@F@a),0.,places=9)
    def test_canonical_projection_and_rig_gauge_invariance(self):
        rng=np.random.default_rng(29);R=Rotation.random(random_state=rng).as_matrix();C=rng.normal(size=3);Rref=Rotation.random(random_state=rng).as_matrix();Cref=rng.normal(size=3);X=rng.normal(size=3)
        Rc,Cc=data.canonical_pose(R,C,Rref,Cref)
        np.testing.assert_allclose(R@(X-C),Rc@(Rref@(X-Cref)-Cc),atol=1e-12)
        Q=Rotation.random(random_state=rng).as_matrix();shift=np.array([5.,-25.,-5.])
        R2,C2=data.canonical_pose(R@Q.T,Q@C+shift,Rref@Q.T,Q@Cref+shift)
        np.testing.assert_allclose(Rc,R2,atol=1e-12);np.testing.assert_allclose(Cc,C2,atol=1e-12)
    def test_random_subset_prior_uses_only_selected_frames(self):
        frames=[dict(frame_id=['session',i],sides=[dict(id='B')]) for i in range(10)]
        pairs=[dict(group='session',frame=i,key=('known_rig_diagnostic','rig',1,0),R=Rotation.from_euler('z',i,degrees=True).as_matrix()) for i in range(10)]
        cameras={'A':dict(R=self.I,C=self.C),'B':dict(R=self.I)}
        with patch.object(data,'real_inputs',return_value=(frames,pairs,cameras,{})):
            a=data.real_package(4,'random',17);b=data.real_package(4,'random',17);c=data.real_package(4,'random',29)
        self.assertEqual(a['metadata']['frame_ids'],b['metadata']['frame_ids'])
        self.assertNotEqual(a['metadata']['frame_ids'],c['metadata']['frame_ids'])
        self.assertEqual(a['metadata']['frame_ids'],[list(x) for x in a['metadata']['prior_frame_ids']])
    def test_output_never_overwrites(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'evidence.json';data.write_new(p,{'x':1})
            with self.assertRaises(FileExistsError):data.write_new(p,{'x':2})

if __name__=='__main__':unittest.main()
