from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from retry_cylinder_epipolar_localization import summarize_default_temperature
from mousegeoforge.localization_audit import refinement_statistics


class DefaultTemperatureSummaryTests(unittest.TestCase):
    def test_no_uncomputed_temperature_fields_required(self):
        r=refinement_statistics(np.array([.5,.5]),np.array([.5,.5]),np.array([.55,.55]),np.zeros((8,8)),(768,768),(.4,.4),scales=(1,))
        summary=summarize_default_temperature([r,r])
        self.assertEqual(summary['points'],2)
        self.assertEqual(summary['scale_1_nme'],r['scale_1_nme'])
        self.assertNotIn('scale_4_nme',summary)


if __name__=='__main__':unittest.main()
