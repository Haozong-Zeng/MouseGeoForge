import importlib.util
from pathlib import Path
import numpy as np
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'tools/audit_admitted_real_calibration_20261025_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CalibrationAuditTests(unittest.TestCase):
    def test_camera_z_not_euclidean_distance(self):
        xy=np.array([[0.,0.],[.5,0.]])
        valid,near,far=m.ray_box(xy,np.eye(3),np.eye(4),[[-2,-2,10],[2,2,20]])
        assert valid.tolist()==[True,False]
        assert near[0]==10 and far[0]==20

    def test_parallel_inside_and_outside(self):
        valid,_,_=m.ray_box(np.array([[0.,0.]]),np.eye(3),np.eye(4),[[1,-2,10],[2,2,20]])
        assert not valid[0]
        valid,_,_=m.ray_box(np.array([[0.,0.]]),np.eye(3),np.eye(4),[[-1,-2,10],[2,2,20]])
        assert valid[0]

    def test_behind_camera_rejected(self):
        valid,_,_=m.ray_box(np.array([[0.,0.]]),np.eye(3),np.eye(4),[[-2,-2,-20],[2,2,-10]])
        assert not valid[0]

    def test_world_translation_of_camera_and_volume(self):
        xy=np.array([[0.,0.],[.1,.1]])
        a=m.ray_box(xy,np.eye(3),np.eye(4),[[-2,-2,10],[2,2,20]])
        e=np.eye(4);e[:3,3]=[-50,2,-100]
        b=m.ray_box(xy,np.eye(3),e,np.array([[-2,-2,10],[2,2,20]])+[50,-2,100])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_halfpixel_native_ray_invariance(self):
        k=np.array([[1800.,0,550],[0,1790,600],[0,0,1.]])
        assert m.halfpixel_resize(k,np.array([[0,0],[550,600],[1079,1199]]),384/1200)<1e-12

    def test_reflection_calibration_rejected(self):
        e=np.eye(4);e[0,0]=-1
        c=dict(K=np.eye(3).tolist(),world_to_camera_mm=e.tolist(),distortion=[0]*5,lens_correction_applied=True,input_assumed_distorted=True)
        with self.assertRaises(AssertionError):m.camera_checks(c)

if __name__=="__main__":unittest.main()
