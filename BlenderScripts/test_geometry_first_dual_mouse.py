#!/usr/bin/env python3

import unittest

import numpy as np

from geometry_first_dual_mouse import (
    Capsule,
    TailV2Config,
    capsule_clearance,
    generate_grounded_tail,
    separate_capsule_sets_xy,
    tail_bend_metrics,
)


class GeometryFirstTests(unittest.TestCase):
    def test_segment_capsule_clearance(self):
        first = Capsule((0, 0, 0), (10, 0, 0), 2, "a")
        second = Capsule((0, 5, 0), (10, 5, 0), 2, "b")
        clearance, _normal = capsule_clearance(first, second)
        self.assertAlmostEqual(clearance, 1.0, places=6)

    def test_capsule_separation(self):
        first = [Capsule((0, 0, 0), (10, 0, 0), 3, "a")]
        second = [Capsule((0, 2, 0), (10, 2, 0), 3, "b")]
        first_offset, second_offset, audit = separate_capsule_sets_xy(first, second, clearance=1.0)
        shifted_first = first[0].shifted_xy(first_offset)
        shifted_second = second[0].shifted_xy(second_offset)
        signed, _normal = capsule_clearance(shifted_first, shifted_second, dimensions=2)
        self.assertGreaterEqual(signed, 0.999)
        self.assertGreater(audit["maximum_initial_penetration_mm"], 0.0)

    def test_grounded_tail_is_exact_length_and_low_bend(self):
        lengths = np.full(13, 8.0)
        config = TailV2Config()
        chain = generate_grounded_tail(
            (0.0, 0.0, 12.0),
            (1.0, 0.0, 0.0),
            lengths,
            floor_z=0.0,
            tangent_offset_degrees=40.0,
            bend_degrees=8.0,
            pattern="s",
            config=config,
        )
        measured = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        self.assertTrue(np.allclose(measured, lengths, atol=1.0e-8))
        self.assertAlmostEqual(chain[-1, 2], config.tail_radius + config.ground_clearance, places=6)
        metrics = tail_bend_metrics(chain)
        self.assertLessEqual(metrics["maximum_bend_degrees"], config.maximum_bend_degrees + 1.0e-6)
        self.assertLessEqual(metrics["total_absolute_bend_degrees"], config.maximum_total_bend_degrees)


if __name__ == "__main__":
    unittest.main()
