"""Pure geometry helpers for the GeometryV2 dual-mouse refinement.

The source 2D track is intentionally a soft prior.  These utilities prioritize
non-penetrating body capsules, grounded low-curvature tails, and temporal
continuity without depending on Blender.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


EPS = 1.0e-9


def unit(vector: Sequence[float], fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


@dataclass(frozen=True)
class Capsule:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    radius: float
    name: str

    def shifted_xy(self, offset: Sequence[float]) -> "Capsule":
        shift = np.asarray((offset[0], offset[1], 0.0), dtype=np.float64)
        return Capsule(
            tuple(np.asarray(self.a, dtype=np.float64) + shift),
            tuple(np.asarray(self.b, dtype=np.float64) + shift),
            self.radius,
            self.name,
        )


def closest_points_segments(
    p1: Sequence[float],
    q1: Sequence[float],
    p2: Sequence[float],
    q2: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return closest points and distance for two finite segments."""
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a <= EPS and e <= EPS:
        return p1, p2, float(np.linalg.norm(p1 - p2))
    if a <= EPS:
        s = 0.0
        t = float(np.clip(f / max(e, EPS), 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0)) if denominator > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    first = p1 + d1 * s
    second = p2 + d2 * t
    return first, second, float(np.linalg.norm(first - second))


def capsule_clearance(first: Capsule, second: Capsule, *, dimensions: int = 3) -> tuple[float, np.ndarray]:
    a0 = np.asarray(first.a, dtype=np.float64)[:dimensions]
    a1 = np.asarray(first.b, dtype=np.float64)[:dimensions]
    b0 = np.asarray(second.a, dtype=np.float64)[:dimensions]
    b1 = np.asarray(second.b, dtype=np.float64)[:dimensions]
    pa, pb, distance = closest_points_segments(a0, a1, b0, b1)
    direction = pb - pa
    if np.linalg.norm(direction) <= EPS:
        direction = (b0 + b1) * 0.5 - (a0 + a1) * 0.5
    fallback = (1.0, 0.0) if dimensions == 2 else (1.0, 0.0, 0.0)
    return distance - first.radius - second.radius, unit(direction, fallback)


def separate_capsule_sets_xy(
    first: Sequence[Capsule],
    second: Sequence[Capsule],
    *,
    clearance: float = 1.0,
    maximum_iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Translate two capsule sets in XY until all pairs have clearance."""
    first_offset = np.zeros(2, dtype=np.float64)
    second_offset = np.zeros(2, dtype=np.float64)
    maximum_initial_penetration = 0.0
    maximum_final_penetration = 0.0
    iterations = 0
    for iteration in range(maximum_iterations):
        worst = None
        for first_capsule in first:
            shifted_first = first_capsule.shifted_xy(first_offset)
            for second_capsule in second:
                shifted_second = second_capsule.shifted_xy(second_offset)
                signed, normal = capsule_clearance(shifted_first, shifted_second, dimensions=2)
                penetration = clearance - signed
                if iteration == 0:
                    maximum_initial_penetration = max(maximum_initial_penetration, penetration)
                if worst is None or penetration > worst[0]:
                    worst = (penetration, normal, first_capsule.name, second_capsule.name)
        if worst is None or worst[0] <= 1.0e-4:
            iterations = iteration
            break
        penetration, normal, _first_name, _second_name = worst
        step = (penetration + 1.0e-3) * 0.505
        first_offset -= normal[:2] * step
        second_offset += normal[:2] * step
        iterations = iteration + 1
    for first_capsule in first:
        shifted_first = first_capsule.shifted_xy(first_offset)
        for second_capsule in second:
            shifted_second = second_capsule.shifted_xy(second_offset)
            signed, _normal = capsule_clearance(shifted_first, shifted_second, dimensions=2)
            maximum_final_penetration = max(maximum_final_penetration, clearance - signed)
    return first_offset, second_offset, {
        "iterations": iterations,
        "maximum_initial_penetration_mm": max(0.0, maximum_initial_penetration),
        "maximum_final_penetration_mm": max(0.0, maximum_final_penetration),
    }


@dataclass(frozen=True)
class TailV2Config:
    tail_radius: float = 0.8
    ground_clearance: float = 0.4
    wall_clearance: float = 1.5
    body_clearance: float = 0.8
    self_clearance: float = 0.4
    descent_length: float = 38.0
    proximal_lock_segments: int = 2
    maximum_bend_degrees: float = 20.0
    maximum_total_bend_degrees: float = 180.0
    own_body_ignore_points: int = 3
    temporal_weight: float = 12.0


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clamp_direction(previous: np.ndarray, candidate: np.ndarray, maximum_radians: float) -> np.ndarray:
    previous = unit(previous)
    candidate = unit(candidate, previous)
    cosine = float(np.clip(np.dot(previous, candidate), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= maximum_radians:
        return candidate
    orthogonal = candidate - previous * cosine
    if np.linalg.norm(orthogonal) <= EPS:
        orthogonal = np.cross(previous, np.asarray((0.0, 0.0, 1.0)))
        if np.linalg.norm(orthogonal) <= EPS:
            orthogonal = np.asarray((1.0, 0.0, 0.0))
    orthogonal = unit(orthogonal)
    return unit(previous * math.cos(maximum_radians) + orthogonal * math.sin(maximum_radians))


def generate_grounded_tail(
    base: Sequence[float],
    root_tangent: Sequence[float],
    segment_lengths: Sequence[float],
    *,
    floor_z: float,
    tangent_offset_degrees: float,
    bend_degrees: float,
    pattern: str,
    config: TailV2Config,
) -> np.ndarray:
    """Generate an exact-length tail that descends smoothly to the floor."""
    base = np.asarray(base, dtype=np.float64)
    tangent_xy = unit(np.asarray(root_tangent, dtype=np.float64)[:2], (1.0, 0.0))
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    points = np.empty((len(lengths) + 1, 3), dtype=np.float64)
    points[0] = base
    root_angle = math.atan2(tangent_xy[1], tangent_xy[0])
    previous_angle = root_angle
    previous_direction = unit(root_tangent)
    curve_angle = 0.0
    cumulative = 0.0
    ground_center_z = floor_z + config.tail_radius + config.ground_clearance
    maximum_bend = math.radians(config.maximum_bend_degrees)
    offset = math.radians(tangent_offset_degrees)
    nominal_bend = math.radians(bend_degrees)
    for index, length in enumerate(lengths):
        fraction = index / max(1, len(lengths) - 1)
        if pattern == "s":
            sign = 1.0 if fraction < 0.52 else -0.72
        elif pattern == "j":
            sign = 0.18 + 0.82 * fraction
        else:
            sign = 1.0
        curve_angle += nominal_bend * sign * (0.38 + 0.62 * fraction)
        offset_gain = _smoothstep((index + 1) / max(1, config.proximal_lock_segments + 2))
        desired_angle = root_angle + offset * offset_gain + curve_angle
        delta = float(np.clip(_wrap_angle(desired_angle - previous_angle), -maximum_bend, maximum_bend))
        angle = previous_angle + delta
        cumulative += float(length)
        descent = _smoothstep(cumulative / max(config.descent_length, EPS))
        target_z = base[2] * (1.0 - descent) + ground_center_z * descent
        dz = float(np.clip(target_z - points[index, 2], -0.82 * length, 0.82 * length))
        planar_length = math.sqrt(max(float(length * length - dz * dz), EPS))
        candidate = np.asarray(
            (math.cos(angle) * planar_length, math.sin(angle) * planar_length, dz),
            dtype=np.float64,
        )
        direction = _clamp_direction(previous_direction, candidate, maximum_bend)
        # Once the chain has reached the ground, do not allow numerical pitch
        # to push it below the tail-radius support plane.
        next_point = points[index] + direction * float(length)
        if next_point[2] < ground_center_z:
            dz = ground_center_z - points[index, 2]
            planar_length = math.sqrt(max(float(length * length - dz * dz), EPS))
            flat = unit(direction[:2], (math.cos(angle), math.sin(angle)))
            next_point = points[index] + np.asarray(
                (flat[0] * planar_length, flat[1] * planar_length, dz),
                dtype=np.float64,
            )
            direction = unit(next_point - points[index])
        points[index + 1] = next_point
        previous_angle = math.atan2(direction[1], direction[0])
        previous_direction = direction
    return points


def tail_bend_metrics(points: np.ndarray) -> dict:
    segments = np.diff(points, axis=0)
    directions = np.asarray([unit(segment) for segment in segments])
    bends = []
    for first, second in zip(directions[:-1], directions[1:], strict=True):
        bends.append(math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))))
    return {
        "maximum_bend_degrees": max(bends, default=0.0),
        "total_absolute_bend_degrees": sum(bends),
        "maximum_segment_length_error_mm": float(
            np.max(np.abs(np.linalg.norm(segments, axis=1) - np.linalg.norm(segments, axis=1)))
        ),
    }


def _point_capsule_clearance(point: np.ndarray, capsule: Capsule, radius: float) -> float:
    a = np.asarray(capsule.a, dtype=np.float64)
    b = np.asarray(capsule.b, dtype=np.float64)
    ab = b - a
    amount = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), EPS), 0.0, 1.0))
    closest = a + amount * ab
    return float(np.linalg.norm(point - closest)) - capsule.radius - radius


def score_tail_candidate(
    points: np.ndarray,
    *,
    arena_minimum: Sequence[float],
    arena_maximum: Sequence[float],
    own_capsules: Sequence[Capsule],
    other_capsules: Sequence[Capsule],
    other_tail: np.ndarray | None,
    tangent_offset_degrees: float,
    bend_degrees: float,
    config: TailV2Config,
) -> tuple[float, dict]:
    arena_minimum = np.asarray(arena_minimum, dtype=np.float64)
    arena_maximum = np.asarray(arena_maximum, dtype=np.float64)
    radius = config.tail_radius
    hard_penalty = 0.0
    minimum_wall_clearance = float("inf")
    minimum_body_clearance = float("inf")
    minimum_self_clearance = float("inf")
    for index, point in enumerate(points[1:], start=1):
        lower_clearance = point[:2] - (arena_minimum[:2] + radius + config.wall_clearance)
        upper_clearance = (arena_maximum[:2] - radius - config.wall_clearance) - point[:2]
        wall_clearance = float(min(np.min(lower_clearance), np.min(upper_clearance)))
        minimum_wall_clearance = min(minimum_wall_clearance, wall_clearance)
        hard_penalty += 2.0e5 * max(0.0, -wall_clearance) ** 2
        samples = [point]
        if index > 0:
            samples.append((points[index - 1] + point) * 0.5)
        for sample in samples:
            if index >= config.own_body_ignore_points:
                for capsule in own_capsules:
                    clearance = _point_capsule_clearance(
                        sample, capsule, radius + config.body_clearance
                    )
                    minimum_body_clearance = min(minimum_body_clearance, clearance)
                    hard_penalty += 2.0e5 * max(0.0, -clearance) ** 2
            for capsule in other_capsules:
                clearance = _point_capsule_clearance(
                    sample, capsule, radius + config.body_clearance
                )
                minimum_body_clearance = min(minimum_body_clearance, clearance)
                hard_penalty += 2.0e5 * max(0.0, -clearance) ** 2
    for first_index in range(len(points) - 1):
        for second_index in range(first_index + 3, len(points) - 1):
            _a, _b, distance = closest_points_segments(
                points[first_index], points[first_index + 1],
                points[second_index], points[second_index + 1],
            )
            clearance = distance - 2.0 * radius - config.self_clearance
            minimum_self_clearance = min(minimum_self_clearance, clearance)
            hard_penalty += 1.0e5 * max(0.0, -clearance) ** 2
    minimum_other_tail_clearance = float("inf")
    if other_tail is not None:
        for first_index in range(len(points) - 1):
            for second_index in range(len(other_tail) - 1):
                _a, _b, distance = closest_points_segments(
                    points[first_index], points[first_index + 1],
                    other_tail[second_index], other_tail[second_index + 1],
                )
                clearance = distance - 2.0 * radius - config.self_clearance
                minimum_other_tail_clearance = min(minimum_other_tail_clearance, clearance)
                hard_penalty += 1.5e5 * max(0.0, -clearance) ** 2
    metrics = tail_bend_metrics(points)
    hard_penalty += 1.0e4 * max(
        0.0, metrics["maximum_bend_degrees"] - config.maximum_bend_degrees - 1.0e-5
    ) ** 2
    hard_penalty += 1.0e3 * max(
        0.0, metrics["total_absolute_bend_degrees"] - config.maximum_total_bend_degrees
    ) ** 2
    natural = (
        0.04 * metrics["total_absolute_bend_degrees"] ** 2
        + 0.08 * tangent_offset_degrees ** 2
        + 0.12 * bend_degrees ** 2
    )
    audit = {
        **metrics,
        "minimum_wall_clearance_mm": minimum_wall_clearance,
        "minimum_body_clearance_mm": minimum_body_clearance,
        "minimum_self_clearance_mm": minimum_self_clearance,
        "minimum_other_tail_clearance_mm": minimum_other_tail_clearance,
        "hard_penalty": hard_penalty,
    }
    return hard_penalty + natural, audit


def _tail_local(points: np.ndarray, base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    forward = unit(np.asarray(tangent)[:2], (1.0, 0.0))
    lateral = np.asarray((-forward[1], forward[0]))
    relative = points - base
    return np.column_stack((relative[:, :2] @ forward, relative[:, :2] @ lateral, relative[:, 2]))


def solve_grounded_tail_sequence(
    bases: np.ndarray,
    root_tangents: np.ndarray,
    segment_lengths: Sequence[float],
    *,
    floor_z: float,
    arena_minimum: Sequence[float],
    arena_maximum: Sequence[float],
    own_capsules_by_frame: Sequence[Sequence[Capsule]],
    other_capsules_by_frame: Sequence[Sequence[Capsule]],
    other_tails_by_frame: np.ndarray | None = None,
    config: TailV2Config = TailV2Config(),
) -> tuple[np.ndarray, list[dict]]:
    """Choose a low-energy planar tail candidate with temporal continuity."""
    offsets = (-80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0)
    bends = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)
    patterns = ("c", "s", "j")
    chains = []
    audits = []
    previous_local = None
    previous_parameters = None
    for frame in range(len(bases)):
        best = None
        for offset in offsets:
            for bend in bends:
                for pattern in patterns:
                    chain = generate_grounded_tail(
                        bases[frame], root_tangents[frame], segment_lengths,
                        floor_z=floor_z,
                        tangent_offset_degrees=offset,
                        bend_degrees=bend,
                        pattern=pattern,
                        config=config,
                    )
                    other_tail = None if other_tails_by_frame is None else other_tails_by_frame[frame]
                    score, audit = score_tail_candidate(
                        chain,
                        arena_minimum=arena_minimum,
                        arena_maximum=arena_maximum,
                        own_capsules=own_capsules_by_frame[frame],
                        other_capsules=other_capsules_by_frame[frame],
                        other_tail=other_tail,
                        tangent_offset_degrees=offset,
                        bend_degrees=bend,
                        config=config,
                    )
                    local = _tail_local(chain, bases[frame], root_tangents[frame])
                    if previous_local is not None:
                        score += config.temporal_weight * float(np.mean(np.sum((local - previous_local) ** 2, axis=1)))
                        previous_offset, previous_bend, previous_pattern = previous_parameters
                        score += 0.8 * (offset - previous_offset) ** 2
                        score += 1.5 * (bend - previous_bend) ** 2
                        score += 120.0 if pattern != previous_pattern else 0.0
                    if best is None or score < best[0]:
                        best = (score, chain, audit, local, (offset, bend, pattern))
        score, chain, audit, previous_local, previous_parameters = best
        audit = {
            **audit,
            "frame_index": frame,
            "score": score,
            "tangent_offset_degrees": previous_parameters[0],
            "bend_parameter_degrees": previous_parameters[1],
            "pattern": previous_parameters[2],
            "tip_height_above_floor_mm": float(chain[-1, 2] - floor_z),
        }
        chains.append(chain)
        audits.append(audit)
    return np.stack(chains), audits
