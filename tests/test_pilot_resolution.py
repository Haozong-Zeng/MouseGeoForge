import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_depth_pose_ablation import pilot_image_hw


class TestPilotResolution(unittest.TestCase):
    def test_resolution_ladder(self):
        for side,height in ((256,144),(512,288),(768,432)):
            self.assertEqual(pilot_image_hw(side,True),[height,side])
            self.assertEqual(pilot_image_hw(side,False),[side,side])
            self.assertEqual(height%16,0)

    def test_reject_unregistered_resolution(self):
        with self.assertRaises(ValueError): pilot_image_hw(500,True)
