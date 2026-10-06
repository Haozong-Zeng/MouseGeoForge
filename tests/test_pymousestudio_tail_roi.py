from pathlib import Path
import sys,unittest
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from repair_pymousestudio_tail_hull import tail_roi


class TailROITests(unittest.TestCase):
    def test_thin_posterior_tail_not_body_or_forepaw(self):
        mask=np.zeros((400,400),np.uint8)
        cv2.ellipse(mask,(190,115),(62,75),0,0,360,255,-1)
        cv2.line(mask,(190,180),(235,365),255,17)
        cv2.circle(mask,(115,80),12,255,-1)
        roi,meta=tail_roi(mask,np.array([190.,183.]),np.array([196.,205.]))
        self.assertTrue(meta['valid'])
        self.assertEqual(roi[115,190],0)
        self.assertEqual(roi[80,115],0)
        self.assertEqual(roi[350,231],255)
        self.assertTrue(np.all(roi[mask==0]==0))

    def test_missing_silhouette_has_no_invented_tail(self):
        roi,meta=tail_roi(np.zeros((400,400),np.uint8),np.array([190.,183.]))
        self.assertFalse(meta['valid']);self.assertFalse(roi.any())

    def test_broad_body_without_tail_does_not_create_line(self):
        mask=np.zeros((400,400),np.uint8);cv2.ellipse(mask,(190,115),(62,75),0,0,360,255,-1)
        roi,meta=tail_roi(mask,np.array([190.,183.]),np.array([196.,205.]))
        self.assertFalse(meta['valid']);self.assertFalse(roi.any())


if __name__=='__main__':unittest.main()
