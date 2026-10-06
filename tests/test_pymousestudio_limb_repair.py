import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from repair_pymousestudio_limb_hull import landmark_roi, connected_additions, reject_new_islands


class LimbRepairTest(unittest.TestCase):
    def test_no_landmark_no_roi(self):
        self.assertFalse(landmark_roi((40,50),[]).any())

    def test_roi_local_and_finite(self):
        roi=landmark_roi((40,50),[np.array([10.,10.]),np.array([np.nan,2.])],radius=3)
        self.assertTrue(roi[10,10])
        self.assertFalse(roi[10,14])

    def test_only_main_body_connected_voxels(self):
        old=np.zeros((12,12,12),bool);old[1:3,1:3,1:3]=True;old[10,10,10]=True
        proposed=np.zeros_like(old);proposed[3:6,2,2]=True;proposed[9,10,10]=True
        added=connected_additions(old.ravel(),proposed.ravel(),old.shape).reshape(old.shape)
        self.assertEqual(int(added.sum()),3)
        self.assertFalse(added[9,10,10])

    def test_no_body_no_new_geometry(self):
        old=np.zeros((3,3,3),bool)
        self.assertFalse(connected_additions(old.ravel(),~old.ravel(),old.shape).any())

    def test_2d_bridge_accepted_but_new_island_rejected(self):
        old=np.zeros((12,12),bool);old[2,2]=True;old[2,6]=True
        added=np.zeros_like(old);added[2,3:6]=True;added[10,10]=True
        result=reject_new_islands(old,added)
        self.assertEqual(int(result.sum()),3)
        self.assertFalse(result[10,10])


if __name__=='__main__':
    unittest.main()
