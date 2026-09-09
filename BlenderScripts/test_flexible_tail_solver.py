"""Fast geometry tests for :mod:`flexible_tail_solver`."""

from __future__ import annotations

import unittest

import numpy as np

from flexible_tail_solver import (
    ArenaBounds,
    EllipsoidProxy,
    TailSolverConfig,
    sample_curvature_latents,
    solve_tail,
    solve_tail_sequence,
)


class FlexibleTailSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lengths = np.full(12, 4.0)
        self.arena = ArenaBounds((-70.0, -45.0, 0.0), (70.0, 45.0, 40.0))
        self.body = EllipsoidProxy((0.0, 0.0, 8.0), (18.0, 11.0, 8.0), "torso", 1.0)
        self.config = TailSolverConfig(body_collision_start_point=3)

    def test_deterministic_and_length_preserving(self) -> None:
        kwargs = dict(
            base=(-17.0, 0.0, 7.0),
            root_tangent=(-1.0, 0.0, -0.05),
            segment_lengths=self.lengths,
            seed=17,
            floor_z=0.0,
            arena=self.arena,
            collision_proxies=(self.body,),
            config=self.config,
        )
        first, audit = solve_tail(**kwargs)
        second, _ = solve_tail(**kwargs)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(np.diff(first, axis=0), axis=1), self.lengths, atol=1e-8)
        np.testing.assert_allclose(first[0], kwargs["base"])
        self.assertTrue(audit.accepted, audit.to_dict())

    def test_correction_resolves_floor_body_and_walls(self) -> None:
        points, audit = solve_tail(
            base=(-17.0, 0.0, 4.5),
            root_tangent=(-0.5, 0.8, -0.8),
            segment_lengths=self.lengths,
            seed=4,
            floor_z=0.0,
            arena=ArenaBounds((-26.0, -24.0, 0.0), (26.0, 24.0, 30.0)),
            collision_proxies=(self.body,),
            config=self.config,
        )
        self.assertTrue(audit.accepted, audit.to_dict())
        self.assertGreater(len(audit.corrected_points), 0)
        self.assertGreaterEqual(float(points[1:, 2].min()), self.config.tail_radius + self.config.floor_clearance - 0.051)
        self.assertLessEqual(float(np.abs(points[1:, 0]).max()), 26.0 - self.config.tail_radius + 0.051)
        self.assertLessEqual(float(np.abs(points[1:, 1]).max()), 24.0 - self.config.tail_radius + 0.051)

    def test_temporal_latents_are_smoother_than_independent_noise(self) -> None:
        latents = sample_curvature_latents(200, len(self.lengths), seed=9, temporal_ar=0.94)
        rng = np.random.default_rng(9)
        independent = rng.normal(size=latents.shape)
        smooth_delta = float(np.mean(np.linalg.norm(np.diff(latents, axis=0), axis=-1)))
        independent_delta = float(np.mean(np.linalg.norm(np.diff(independent, axis=0), axis=-1)))
        self.assertLess(smooth_delta, independent_delta * 0.5)

    def test_distal_tail_is_diverse_while_root_stays_conditioned(self) -> None:
        tips = []
        first_segments = []
        for seed in range(20):
            points, audit = solve_tail(
                (-17.0, 0.0, 7.0),
                (-1.0, 0.0, 0.0),
                self.lengths,
                seed=seed,
                floor_z=0.0,
                arena=self.arena,
                collision_proxies=(self.body,),
                config=self.config,
            )
            self.assertTrue(audit.accepted, audit.to_dict())
            tips.append(points[-1])
            first_segments.append(points[1])
        tip_spread = float(np.mean(np.std(np.asarray(tips), axis=0)))
        root_spread = float(np.mean(np.std(np.asarray(first_segments), axis=0)))
        self.assertGreater(tip_spread, root_spread * 5.0)

    def test_sequence_api(self) -> None:
        frames = 24
        bases = np.tile((-17.0, 0.0, 7.0), (frames, 1))
        tangents = np.tile((-1.0, 0.0, 0.0), (frames, 1))
        chains, audits = solve_tail_sequence(
            bases,
            tangents,
            self.lengths,
            seed=23,
            floor_z=0.0,
            arena=self.arena,
            collision_proxies_by_frame=[(self.body,)] * frames,
            config=self.config,
        )
        self.assertEqual(chains.shape, (frames, 13, 3))
        self.assertTrue(all(audit.accepted for audit in audits))
        tip_motion = np.linalg.norm(np.diff(chains[:, -1], axis=0), axis=1)
        self.assertLess(float(np.percentile(tip_motion, 95)), 20.0)


if __name__ == "__main__":
    unittest.main()

