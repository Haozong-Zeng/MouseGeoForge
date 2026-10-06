import unittest
import torch
from mousegeoforge.models.regional_peak_candidates import regional_top_peaks


def output(a, valid=None):
    a = torch.tensor(a, dtype=torch.float32)
    if a.ndim == 2:
        a = a[None]
    h, w = a.shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(h)+.5)*4, (torch.arange(w)+.5)*4, indexing='ij')
    return dict(logits=a, grid=torch.stack([xx/(w*4), yy/(h*4)], -1),
                spatial_valid=torch.ones(h, w, dtype=torch.bool) if valid is None else valid)


class RegionalPeakTests(unittest.TestCase):
    def test_single_peak_no_background_filling(self):
        a=-torch.arange(25).reshape(5,5).float()
        p=regional_top_peaks(output(a.tolist()),(20,20))
        self.assertEqual(int(p['available'].sum()),1)
        self.assertTrue(torch.allclose(p['xy'][0,0],torch.tensor([.1,.1])))

    def test_flat_no_localized_evidence(self):
        p=regional_top_peaks(output([[1,1],[1,1]]),(8,8))
        self.assertFalse(p['available'].any())

    def test_large_flat_background_not_a_peak(self):
        a=torch.zeros(9,9);a[4,4]=5
        p=regional_top_peaks(output(a.tolist()),(36,36),radius_px=0)
        self.assertEqual(int(p['available'].sum()),1)

    def test_plateau_once_deterministic(self):
        p=regional_top_peaks(output([[0,0,0,0],[0,2,2,0],[0,2,2,0],[0,0,0,0]]),(16,16),radius_px=0)
        self.assertEqual(int(p['available'].sum()),1)
        self.assertTrue(torch.allclose(p['xy'][0,0],torch.tensor([.375,.375])))

    def test_two_peaks_radius(self):
        a=output([[5,3,1,3,4]])
        self.assertEqual(int(regional_top_peaks(a,(4,20),radius_px=8)['available'].sum()),2)
        self.assertEqual(int(regional_top_peaks(a,(4,20),radius_px=16)['available'].sum()),1)

    def test_padding_excluded(self):
        a=output([[1,0,100]],torch.tensor([[True,True,False]]))
        p=regional_top_peaks(a,(4,12))
        self.assertAlmostEqual(float(p['probability_map'][0,0,2]),0.)
        self.assertAlmostEqual(float(p['xy'][0,0,0]),1/6)

    def test_all_invalid_finite(self):
        p=regional_top_peaks(output([[1,2]],torch.tensor([[False,False]])),(4,8))
        self.assertFalse(p['available'].any())
        self.assertTrue(torch.isfinite(p['probability_map']).all())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):regional_top_peaks(output([[1,2]]),(4,8),count=0)
        with self.assertRaises(ValueError):regional_top_peaks(output([[1,float('nan')]]),(4,8))

    def test_channels_independent(self):
        p=regional_top_peaks(output([[[2,1,0]],[[0,1,2]]]),(4,12))
        self.assertTrue(p['xy'][0,0,0]<p['xy'][1,0,0])


if __name__=='__main__':unittest.main()
