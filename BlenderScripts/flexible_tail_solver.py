"""Geometry-only flexible tail sampling and penetration correction.

This module intentionally has no ``bpy`` dependency.  Blender adapters can feed
the solved points to a Spline IK curve, while tests and dataset tooling can audit
the same geometry in ordinary Python.

The source pose controls only ``tail_base`` and the initial tangent.  Distal tail
shape is sampled from smooth correlated curvature and then projected against the
floor, cage bounds, and ellipsoidal collision proxies without changing the body
or limb controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np


EPS = 1.0e-9


def _unit(vector: np.ndarray, fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def _orthogonal_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = _unit(tangent)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    lateral = np.cross(up, tangent)
    if np.linalg.norm(lateral) <= 1.0e-6:
        lateral = np.cross(np.array((1.0, 0.0, 0.0)), tangent)
    lateral = _unit(lateral, (0.0, 1.0, 0.0))
    vertical = _unit(np.cross(tangent, lateral), (0.0, 0.0, 1.0))
    return lateral, vertical


def _clamp_bend(previous: np.ndarray, candidate: np.ndarray, maximum_radians: float) -> np.ndarray:
    previous = _unit(previous)
    candidate = _unit(candidate, previous)
    cosine = float(np.clip(np.dot(previous, candidate), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= maximum_radians:
        return candidate
    orthogonal = candidate - previous * cosine
    if np.linalg.norm(orthogonal) <= EPS:
        orthogonal = _orthogonal_frame(previous)[0]
    orthogonal = _unit(orthogonal)
    return previous * np.cos(maximum_radians) + orthogonal * np.sin(maximum_radians)


def _project_unit_direction_to_bounds(
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Project a unit vector into component bounds while retaining unit length.

    Alternating box/sphere projection is sufficient here because the problem is
    only three-dimensional and the bounds always contain a feasible direction
    for a tail base inside the cage.
    """
    lower = np.maximum(np.asarray(lower, dtype=np.float64), -1.0)
    upper = np.minimum(np.asarray(upper, dtype=np.float64), 1.0)
    candidate = _unit(direction)
    for _ in range(30):
        candidate = np.minimum(np.maximum(candidate, lower), upper)
        norm = float(np.linalg.norm(candidate))
        if norm <= EPS:
            # Select the axis offering the greatest feasible magnitude.
            axis = int(np.argmax(np.maximum(np.abs(lower), np.abs(upper))))
            candidate[axis] = upper[axis] if abs(upper[axis]) >= abs(lower[axis]) else lower[axis]
            norm = float(np.linalg.norm(candidate))
        candidate /= max(norm, EPS)
        if np.all(candidate >= lower - 1.0e-10) and np.all(candidate <= upper + 1.0e-10):
            return candidate
    candidate = np.minimum(np.maximum(candidate, lower), upper)
    return _unit(candidate, direction)


@dataclass(frozen=True)
class EllipsoidProxy:
    center: tuple[float, float, float]
    radii: tuple[float, float, float]
    name: str = "body"
    margin: float = 0.0
    rotation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ArenaBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


@dataclass(frozen=True)
class TailSolverConfig:
    tail_radius: float = 0.8
    floor_clearance: float = 0.15
    proximal_lock_segments: int = 2
    body_collision_start_point: int = 2
    yaw_sigma_degrees: float = 10.0
    pitch_sigma_degrees: float = 5.0
    distal_noise_gain: float = 1.6
    curvature_ar: float = 0.78
    temporal_ar: float = 0.97
    maximum_bend_degrees: float = 28.0
    correction_iterations: int = 40
    collision_tolerance: float = 0.05


@dataclass
class TailSolveAudit:
    accepted: bool = False
    iterations: int = 0
    floor_pushes: int = 0
    wall_pushes: int = 0
    body_pushes: int = 0
    length_error_max: float = 0.0
    maximum_residual_penetration: float = 0.0
    corrected_points: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["corrected_points"] = sorted(self.corrected_points)
        return payload


def sample_curvature_latents(
    frame_count: int,
    segment_count: int,
    seed: int,
    temporal_ar: float = 0.94,
) -> np.ndarray:
    """Return smooth yaw/pitch latents with shape ``[F, segments, 2]``."""
    if frame_count < 1 or segment_count < 1:
        raise ValueError("frame_count and segment_count must be positive")
    if not 0.0 <= temporal_ar < 1.0:
        raise ValueError("temporal_ar must be in [0, 1)")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(frame_count, segment_count, 2))
    latents = np.empty_like(innovations)
    latents[0] = innovations[0]
    scale = np.sqrt(max(0.0, 1.0 - temporal_ar * temporal_ar))
    for frame in range(1, frame_count):
        latents[frame] = temporal_ar * latents[frame - 1] + scale * innovations[frame]
    return latents


def _initial_chain(
    base: np.ndarray,
    root_tangent: np.ndarray,
    segment_lengths: np.ndarray,
    latents: np.ndarray,
    config: TailSolverConfig,
) -> np.ndarray:
    point_count = len(segment_lengths) + 1
    points = np.empty((point_count, 3), dtype=np.float64)
    points[0] = base
    direction = _unit(root_tangent)
    yaw_state = 0.0
    pitch_state = 0.0
    yaw_sigma = np.deg2rad(config.yaw_sigma_degrees)
    pitch_sigma = np.deg2rad(config.pitch_sigma_degrees)
    maximum_bend = np.deg2rad(config.maximum_bend_degrees)

    for index, length in enumerate(segment_lengths):
        fraction = index / max(1, len(segment_lengths) - 1)
        if index < config.proximal_lock_segments:
            gain = 0.12 * (index + 1) / max(1, config.proximal_lock_segments)
        else:
            gain = 0.35 + config.distal_noise_gain * fraction
        yaw_state = config.curvature_ar * yaw_state + yaw_sigma * gain * float(latents[index, 0])
        pitch_state = config.curvature_ar * pitch_state + pitch_sigma * gain * float(latents[index, 1])
        lateral, vertical = _orthogonal_frame(direction)
        candidate = direction + np.tan(yaw_state) * lateral + np.tan(pitch_state) * vertical
        direction = _clamp_bend(direction, candidate, maximum_bend)
        points[index + 1] = points[index] + _unit(direction) * float(length)
    return points


def _ellipsoid_push(point: np.ndarray, proxy: EllipsoidProxy, radius: float) -> tuple[np.ndarray, float]:
    center = np.asarray(proxy.center, dtype=np.float64)
    radii = np.asarray(proxy.radii, dtype=np.float64) + float(proxy.margin) + radius
    radii = np.maximum(radii, EPS)
    rotation = np.asarray(proxy.rotation, dtype=np.float64).reshape(3, 3)
    relative = rotation.T @ (point - center)
    scaled = relative / radii
    distance = float(np.linalg.norm(scaled))
    if distance >= 1.0:
        return point, 0.0
    if distance <= EPS:
        relative = np.array((-1.0, 0.0, 0.0), dtype=np.float64)
        scaled = relative / radii
        distance = float(np.linalg.norm(scaled))
    surface = center + rotation @ (relative / distance)
    penetration = float(np.linalg.norm(surface - point))
    return surface, penetration


def _project_obstacles(
    points: np.ndarray,
    floor_z: float,
    arena: ArenaBounds | None,
    proxies: Iterable[EllipsoidProxy],
    config: TailSolverConfig,
    audit: TailSolveAudit,
) -> float:
    maximum = 0.0
    minimum_z = floor_z + config.tail_radius + config.floor_clearance
    arena_min = np.asarray(arena.minimum, dtype=np.float64) if arena else None
    arena_max = np.asarray(arena.maximum, dtype=np.float64) if arena else None
    for index in range(1, len(points)):
        if points[index, 2] < minimum_z:
            depth = float(minimum_z - points[index, 2])
            points[index, 2] = minimum_z
            maximum = max(maximum, depth)
            audit.floor_pushes += 1
            audit.corrected_points.add(index)
        if arena is not None:
            low = arena_min + config.tail_radius
            high = arena_max - config.tail_radius
            clipped = np.minimum(np.maximum(points[index], low), high)
            depth = float(np.max(np.abs(clipped - points[index])))
            if depth > 0.0:
                points[index] = clipped
                maximum = max(maximum, depth)
                audit.wall_pushes += 1
                audit.corrected_points.add(index)
        if index >= config.body_collision_start_point:
            for proxy in proxies:
                points[index], depth = _ellipsoid_push(points[index], proxy, config.tail_radius)
                if depth > 0.0:
                    maximum = max(maximum, depth)
                    audit.body_pushes += 1
                    audit.corrected_points.add(index)
    return maximum


def _restore_chain(
    points: np.ndarray,
    base: np.ndarray,
    root_tangent: np.ndarray,
    segment_lengths: np.ndarray,
    config: TailSolverConfig,
    floor_z: float,
    arena: ArenaBounds | None,
) -> None:
    points[0] = base
    maximum_bend = np.deg2rad(config.maximum_bend_degrees)
    previous = _unit(root_tangent)
    for index, length in enumerate(segment_lengths):
        candidate = _unit(points[index + 1] - points[index], previous)
        if index < config.proximal_lock_segments:
            # A graded root prior prevents a sharp kink where the tail exits the body.
            root_weight = 0.80 if index == 0 else 0.45
            candidate = _unit(root_weight * _unit(root_tangent) + (1.0 - root_weight) * candidate)
        direction = _clamp_bend(previous, candidate, maximum_bend)
        # Preserve bone length while making the next endpoint floor/cage-feasible.
        # Coordinate-clamping the endpoint would shorten the bone.
        minimum_z = floor_z + config.tail_radius + config.floor_clearance
        lower = np.full(3, -1.0, dtype=np.float64)
        upper = np.full(3, 1.0, dtype=np.float64)
        lower[2] = max(lower[2], float((minimum_z - points[index, 2]) / length))
        if arena is not None:
            arena_low = np.asarray(arena.minimum, dtype=np.float64) + config.tail_radius
            arena_high = np.asarray(arena.maximum, dtype=np.float64) - config.tail_radius
            arena_low[2] = max(arena_low[2], minimum_z)
            lower = np.maximum(lower, (arena_low - points[index]) / length)
            upper = np.minimum(upper, (arena_high - points[index]) / length)
        direction = _project_unit_direction_to_bounds(direction, lower, upper)
        points[index + 1] = points[index] + direction * float(length)
        previous = direction


def _residual_penetration(
    points: np.ndarray,
    floor_z: float,
    arena: ArenaBounds | None,
    proxies: Iterable[EllipsoidProxy],
    config: TailSolverConfig,
) -> float:
    maximum = 0.0
    minimum_z = floor_z + config.tail_radius + config.floor_clearance
    arena_min = np.asarray(arena.minimum, dtype=np.float64) if arena else None
    arena_max = np.asarray(arena.maximum, dtype=np.float64) if arena else None
    for index in range(1, len(points)):
        maximum = max(maximum, max(0.0, float(minimum_z - points[index, 2])))
        if arena is not None:
            low = arena_min + config.tail_radius
            high = arena_max - config.tail_radius
            maximum = max(maximum, float(np.max(np.maximum(low - points[index], 0.0))))
            maximum = max(maximum, float(np.max(np.maximum(points[index] - high, 0.0))))
        if index >= config.body_collision_start_point:
            for proxy in proxies:
                center = np.asarray(proxy.center, dtype=np.float64)
                radii = np.asarray(proxy.radii, dtype=np.float64) + proxy.margin + config.tail_radius
                rotation = np.asarray(proxy.rotation, dtype=np.float64).reshape(3, 3)
                local = rotation.T @ (points[index] - center)
                distance = float(np.linalg.norm(local / np.maximum(radii, EPS)))
                if distance < 1.0:
                    maximum = max(maximum, (1.0 - distance) * float(np.min(radii)))
    return maximum


def solve_tail(
    base: Sequence[float],
    root_tangent: Sequence[float],
    segment_lengths: Sequence[float],
    *,
    seed: int = 0,
    curvature_latents: np.ndarray | None = None,
    floor_z: float = 0.0,
    arena: ArenaBounds | None = None,
    collision_proxies: Iterable[EllipsoidProxy] = (),
    config: TailSolverConfig = TailSolverConfig(),
) -> tuple[np.ndarray, TailSolveAudit]:
    """Sample and correct one tail chain while keeping the base fixed."""
    base_array = np.asarray(base, dtype=np.float64)
    tangent_array = _unit(np.asarray(root_tangent, dtype=np.float64))
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if base_array.shape != (3,) or tangent_array.shape != (3,):
        raise ValueError("base and root_tangent must be 3-vectors")
    if lengths.ndim != 1 or len(lengths) < 1 or np.any(lengths <= 0.0):
        raise ValueError("segment_lengths must be a non-empty positive vector")
    if curvature_latents is None:
        curvature_latents = np.random.default_rng(seed).normal(size=(len(lengths), 2))
    curvature_latents = np.asarray(curvature_latents, dtype=np.float64)
    if curvature_latents.shape != (len(lengths), 2):
        raise ValueError(f"curvature_latents must have shape {(len(lengths), 2)}")

    proxies = tuple(collision_proxies)
    points = _initial_chain(base_array, tangent_array, lengths, curvature_latents, config)
    audit = TailSolveAudit()
    for iteration in range(config.correction_iterations):
        _project_obstacles(points, floor_z, arena, proxies, config, audit)
        _restore_chain(points, base_array, tangent_array, lengths, config, floor_z, arena)
        residual = _residual_penetration(points, floor_z, arena, proxies, config)
        audit.iterations = iteration + 1
        if residual <= config.collision_tolerance:
            break

    residual = _residual_penetration(points, floor_z, arena, proxies, config)
    measured = np.linalg.norm(np.diff(points, axis=0), axis=1)
    audit.length_error_max = float(np.max(np.abs(measured - lengths)))
    audit.maximum_residual_penetration = residual
    audit.accepted = residual <= config.collision_tolerance and audit.length_error_max <= 1.0e-6
    return points, audit


def solve_tail_sequence(
    bases: np.ndarray,
    root_tangents: np.ndarray,
    segment_lengths: Sequence[float],
    *,
    seed: int = 0,
    floor_z: float = 0.0,
    arena: ArenaBounds | None = None,
    collision_proxies_by_frame: Sequence[Iterable[EllipsoidProxy]] | None = None,
    config: TailSolverConfig = TailSolverConfig(),
) -> tuple[np.ndarray, list[TailSolveAudit]]:
    """Solve a temporally correlated tail sequence."""
    bases = np.asarray(bases, dtype=np.float64)
    root_tangents = np.asarray(root_tangents, dtype=np.float64)
    if bases.ndim != 2 or bases.shape[1] != 3 or root_tangents.shape != bases.shape:
        raise ValueError("bases and root_tangents must both have shape [frames, 3]")
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    latents = sample_curvature_latents(len(bases), len(lengths), seed, config.temporal_ar)
    if collision_proxies_by_frame is None:
        collision_proxies_by_frame = [()] * len(bases)
    if len(collision_proxies_by_frame) != len(bases):
        raise ValueError("collision_proxies_by_frame must match the number of frames")

    chains = []
    audits = []
    for frame in range(len(bases)):
        chain, audit = solve_tail(
            bases[frame],
            root_tangents[frame],
            lengths,
            curvature_latents=latents[frame],
            floor_z=floor_z,
            arena=arena,
            collision_proxies=collision_proxies_by_frame[frame],
            config=config,
        )
        chains.append(chain)
        audits.append(audit)
    return np.stack(chains), audits
