"""Dataset-free guards for post-hoc calibration; no model fitting or test leakage."""
import math
import unittest

import torch

from mousegeoforge.uncertainty_calibration import (
    apply_temperature, fit_temperature, uncertainty_metrics,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.residual = torch.tensor([[3., 4.], [0., 5.]], dtype=torch.float64)
        self.sigma = torch.tensor([1., 2.], dtype=torch.float64)

    def test_analytic_mle(self):
        self.assertAlmostEqual(fit_temperature(self.residual, self.sigma), math.sqrt(7.8125))

    def test_scale_equivariance(self):
        t = fit_temperature(self.residual, self.sigma)
        other = fit_temperature(self.residual, self.sigma * 7)
        torch.testing.assert_close(apply_temperature(self.sigma, t),
                                   apply_temperature(self.sigma * 7, other))

    def test_positive_temperature_preserves_ranking(self):
        torch.testing.assert_close(self.sigma.argsort(), apply_temperature(self.sigma, 3.).argsort())

    def test_no_mutation(self):
        residual, sigma = self.residual.clone(), self.sigma.clone()
        fit_temperature(residual, sigma)
        uncertainty_metrics(residual, apply_temperature(sigma, 2.))
        torch.testing.assert_close(residual, self.residual)
        torch.testing.assert_close(sigma, self.sigma)

    def test_calibration_optimum(self):
        t = fit_temperature(self.residual, self.sigma)
        best = uncertainty_metrics(self.residual, self.sigma * t)['nll']
        for scale in (.5, .9, 1.1, 2.):
            self.assertLess(best, uncertainty_metrics(self.residual, self.sigma * t * scale)['nll'])

    def test_zero_error_is_finite(self):
        t = fit_temperature(torch.zeros_like(self.residual), self.sigma)
        self.assertGreater(t, 0.)
        self.assertTrue(math.isfinite(t))

    def test_invalid_inputs(self):
        bad_pairs = [(torch.zeros(3), torch.ones(1)),
                     (torch.zeros(2, 3), torch.ones(2)),
                     (torch.zeros(0, 2), torch.ones(0)),
                     (torch.zeros(2, 2), torch.zeros(2)),
                     (torch.full((2, 2), float('nan')), torch.ones(2)),
                     (torch.zeros(2, 2), torch.full((2,), float('inf')))]
        for residual, sigma in bad_pairs:
            for function in (fit_temperature, uncertainty_metrics):
                with self.subTest(function=function.__name__, shape=residual.shape):
                    with self.assertRaises(ValueError):
                        function(residual, sigma)

    def test_invalid_temperatures_and_scales(self):
        for t in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                apply_temperature(self.sigma, t)
        with self.assertRaises(ValueError):
            apply_temperature(-self.sigma, 1.)

    def test_constant_scale_has_no_selective_ranking(self):
        result = uncertainty_metrics(self.residual, torch.ones(2))
        self.assertFalse(result['ranking_informative'])
        self.assertIsNone(result['risk_coverage']['0.25'])
        self.assertEqual(result['risk_coverage']['1.0'], 5.)

    def test_coverage_uses_2d_radial_not_1d_interval(self):
        result = uncertainty_metrics(torch.tensor([[0., 0.], [2., 0.], [3., 0.]]), torch.ones(3))
        self.assertAlmostEqual(result['coverage_68'], 1 / 3)
        self.assertAlmostEqual(result['coverage_95'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
