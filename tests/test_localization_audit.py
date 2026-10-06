import unittest
import numpy as np
from mousegeoforge.localization_audit import refinement_statistics


class LocalizationAuditTests(unittest.TestCase):
    def test_uniform_grid_no_shift(self):
        r=refinement_statistics([.5,.5],[.5,.5],[.55,.5],np.zeros((8,8)),[432,768],[.2,.2])
        self.assertAlmostEqual(r['entropy_normalized'],1.)
        self.assertAlmostEqual(r['max_probability'],1/64)
        self.assertAlmostEqual(r['coarse_px'],38.4)
        self.assertTrue(r['inside_window']);self.assertEqual(r['window_oracle_min_px'],0.)

    def test_outside_window_floor(self):
        r=refinement_statistics([.5,.5],[.5,.5],[.8,.5],np.zeros((8,8)),[768,768],[.2,.2])
        self.assertFalse(r['inside_window']);self.assertAlmostEqual(r['window_oracle_min_px'],153.6)

    def test_reconstruction_guard(self):
        with self.assertRaises(ValueError):
            refinement_statistics([.5,.5],[.6,.5],[.5,.5],np.zeros((8,8)),[768,768],[.2,.2])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            refinement_statistics([np.nan,.5],[.5,.5],[.5,.5],np.zeros((8,8)),[768,768],[.2,.2])


if __name__=='__main__':unittest.main()
