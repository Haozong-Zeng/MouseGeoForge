import pickle
import unittest
import cv2
import numpy as np
from mousegeoforge.raw_validation import primitive_column_metadata,rectify_points,group_for_session


class RawValidationTests(unittest.TestCase):
    def test_primitive_columns(self):
        x=[('scorer','nose','x'),('scorer','nose','y'),('scorer','nose','likelihood')]
        self.assertEqual(primitive_column_metadata(pickle.dumps(x,protocol=0)),x)

    def test_reject_executable_pickle_metadata(self):
        with self.assertRaises(ValueError):primitive_column_metadata(b'cos\nsystem\n.')

    def test_lens_roundtrip(self):
        K=np.array([[1000.,0,540],[0,1000,600],[0,0,1]])
        D=np.array([-.49,.26,.001,-.001,.08])
        xyz=np.array([[-.3,-.3,1],[.3,.4,1],[0,0,1.]])
        uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)
        ideal,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,np.zeros(5))
        np.testing.assert_allclose(rectify_points(uv.reshape(-1,2),K,D),ideal.reshape(-1,2),atol=1e-6)

    def test_missing_points_stay_missing(self):
        uv=np.array([[np.nan,np.nan],[10.,20.]])
        out=rectify_points(uv,np.eye(3),np.zeros(5))
        self.assertTrue(np.isnan(out[0]).all());np.testing.assert_allclose(out[1],uv[1])

    def test_conservative_source_group(self):
        self.assertEqual(group_for_session('AZR2L2'),group_for_session('AZL3'))
        self.assertEqual(group_for_session('UR2'),group_for_session('UR3'))
