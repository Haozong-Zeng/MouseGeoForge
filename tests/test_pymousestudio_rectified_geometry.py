"""Focused numerical tests for the corrected real-data pilot."""
from pathlib import Path
import sys
import unittest
import tempfile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mousegeoforge.raw_validation import rectify_points
from reconstruct_pymousestudio_pilot import triangulate
from add_dannce_real_multiview_pseudo_depth import project,write_pfm,read_pfm


def test_rectified_points_project_back_to_distorted_pixels():
    K=np.array([[1790.,0,550.],[0,1791.,555.],[0,0,1.]])
    D=np.array([-.49,.26,-.00006,-.00078,.078])
    xyz=np.array([[-50.,-40.,200.],[30.,50.,210.],[0,0,180.]])
    raw,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)
    rect=rectify_points(raw[:,0],K,D)
    ideal=(xyz@K.T);ideal=ideal[:,:2]/ideal[:,2:]
    np.testing.assert_allclose(rect,ideal,atol=1e-5)


def test_robust_triangulation_rejects_one_bad_camera():
    K=np.array([[1790.,0,550.],[0,1791.,555.],[0,0,1.]])
    Ks={};Es={};obs={};truth=np.array([10.,12.,20.])
    for name,t in zip('BCDE',[[-50,0,200],[50,0,200],[0,50,200],[0,-50,200]]):
        E=np.eye(4);E[:3,3]=t;Ks[name]=K;Es[name]=E
        obs[name]=project(truth[None],K,E)[0][0]
    obs['E']=obs['E']+np.array([200.,300.])
    xyz,error=triangulate(obs,Ks,Es)
    np.testing.assert_allclose(xyz,truth,atol=1e-6)
    assert error<1e-6


def test_pfm_mm_camera_z_preserves_rows_and_invalid_zero():
    depth=np.array([[0.,123.5,180.],[240.,200.,0.]],np.float32)
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'test.pfm';write_pfm(p,depth)
        np.testing.assert_array_equal(read_pfm(p),depth)


class RectifiedGeometryTests(unittest.TestCase):
    def test_point_rectification(self):test_rectified_points_project_back_to_distorted_pixels()
    def test_robust_triangulation(self):test_robust_triangulation_rejects_one_bad_camera()
    def test_metric_pfm(self):test_pfm_mm_camera_z_preserves_rows_and_invalid_zero()


if __name__=='__main__':unittest.main()
