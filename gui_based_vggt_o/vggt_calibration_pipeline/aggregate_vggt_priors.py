#!/usr/bin/env python3
"""Aggregate per-frame VGGT calibrations into a robust rig prior.

The multi-camera rig is static across a video, so every frame's VGGT
calibration is an independent noisy measurement of the same rig geometry.
This script:

1. loads all frame_XXXXXX/calibration.toml files under a dataset directory,
2. brings every frame into a common gauge before averaging. Default
   --align-mode procrustes similarity-aligns each frame's camera configuration
   to the running consensus (removes a per-frame global rotation/translation/
   scale), so the aggregate is gauge-invariant and survives a backbone that does
   not pin a camera. --align-mode scale only normalizes per-frame scale (the
   original behavior, which relies on the first camera being pinned, as VGGT
   pins camera A to identity),
3. rejects only catastrophic geometry blow-ups (default --exclude-mode absolute:
   a camera flipped >20 deg or displaced >30% of baseline from the across-frame
   consensus). The robust aggregate in step 4 already absorbs ordinary spread, so
   the older strict per-sigma trimming (--exclude-mode sigma) wasted good frames
   for no gain in prior quality; it is kept for reproducibility,
4. averages rotations (chordal mean), camera centers (median), and
   intrinsics (median) over inlier frames,
5. writes an aggregated calibration_prior.toml plus prior_stats.json with
   empirical per-camera and per-pair sigmas usable as calibrated prior
   strengths in refine_metric_calibration.py.

Outputs:
  OUTPUT_DIR/calibration_prior.toml
  OUTPUT_DIR/prior_stats.json
  OUTPUT_DIR/frame_deviations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


MAD_TO_SIGMA = 1.4826


@dataclass
class FrameCalibration:
    frame: str
    names: list[str]
    sizes: dict[str, tuple[int, int]]
    matrices: dict[str, np.ndarray]
    distortions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    fisheye: dict[str, bool]


def read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("Python 3.11+ tomllib is required.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_frame_calibration(path: Path) -> FrameCalibration:
    data = read_toml(path)
    names: list[str] = []
    sizes: dict[str, tuple[int, int]] = {}
    matrices: dict[str, np.ndarray] = {}
    distortions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {}
    fisheye: dict[str, bool] = {}
    entries = []
    for key, value in data.items():
        if not isinstance(value, dict) or not str(key).startswith("cam_"):
            continue
        try:
            index = int(str(key).split("_", 1)[1])
        except ValueError:
            continue
        entries.append((index, value))
    entries.sort(key=lambda item: item[0])
    for _, value in entries:
        name = str(value.get("name"))
        size_value = value.get("size") or [0, 0]
        rotation, _ = cv2.Rodrigues(np.asarray(value["rotation"], dtype=np.float64).reshape(3))
        tvec = np.asarray(value["translation"], dtype=np.float64).reshape(3)
        names.append(name)
        sizes[name] = (int(size_value[0]), int(size_value[1]))
        matrices[name] = np.asarray(value["matrix"], dtype=np.float64).reshape(3, 3)
        distortions[name] = np.asarray(value.get("distortions") or [0, 0, 0, 0, 0], dtype=np.float64).reshape(-1)
        rotations[name] = rotation
        centers[name] = -rotation.T @ tvec
        fisheye[name] = bool(value.get("fisheye", False))
    if len(names) < 2:
        raise ValueError(f"Fewer than two cameras in {path}")
    return FrameCalibration(
        frame=path.parent.name,
        names=names,
        sizes=sizes,
        matrices=matrices,
        distortions=distortions,
        rotations=rotations,
        centers=centers,
        fisheye=fisheye,
    )


def mean_baseline(centers: dict[str, np.ndarray], names: Sequence[str]) -> float:
    values = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            values.append(float(np.linalg.norm(centers[name_a] - centers[name_b])))
    return float(np.mean(values)) if values else 0.0


def chordal_mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    mean_matrix = np.mean(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _, vt = np.linalg.svd(mean_matrix)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    return rotation


def geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    value = np.clip((np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity (R, t, s) with s*R*src + t ~= dst, for (N,3) sets.

    Used to align each frame's camera configuration to the consensus before
    averaging, so the aggregation is invariant to a per-frame global rotation/
    translation/scale (gauge). Robust to a backbone that does not pin a camera.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    sigma = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(sigma)
    w = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        w[2, 2] = -1.0
    rotation = u @ w @ vt
    var = float((sc ** 2).sum() / len(src))
    scale = float(np.trace(np.diag(d) @ w) / var) if var > 1e-12 else 1.0
    translation = mu_d - scale * rotation @ mu_s
    return rotation, translation, scale


def robust_sigma(values: np.ndarray, floor: float = 1e-9) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return floor
    mad = float(np.median(np.abs(arr - np.median(arr))))
    return max(MAD_TO_SIGMA * mad, floor)


def toml_array(values: Any) -> str:
    return json.dumps(np.asarray(values, dtype=np.float64).tolist(), allow_nan=False)


def write_prior_calibration(
    path: Path,
    template: FrameCalibration,
    matrices: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
) -> None:
    lines = [
        "# Generated by vggt_calibration_pipeline/aggregate_vggt_priors.py",
        "# Robust aggregate of per-frame VGGT calibrations (static rig prior).",
        "",
    ]
    for index, name in enumerate(template.names):
        rvec, _ = cv2.Rodrigues(rotations[name])
        tvec = -rotations[name] @ centers[name]
        lines.extend(
            [
                f"[cam_{index}]",
                f"name = {json.dumps(name)}",
                f"size = {json.dumps([int(template.sizes[name][0]), int(template.sizes[name][1])])}",
                f"matrix = {toml_array(matrices[name])}",
                f"distortions = {toml_array(template.distortions[name])}",
                f"rotation = {toml_array(rvec.reshape(3))}",
                f"translation = {toml_array(tvec.reshape(3))}",
            ]
        )
        if template.fisheye[name]:
            lines.append("fisheye = true")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset_dir.expanduser().resolve()
    paths = sorted(dataset.glob(args.frame_glob + "/calibration.toml"))
    if args.max_frames and len(paths) > args.max_frames:
        stride = len(paths) / args.max_frames
        paths = [paths[int(i * stride)] for i in range(args.max_frames)]
    if len(paths) < 3:
        raise ValueError(f"Need at least 3 frame calibrations, found {len(paths)} in {dataset}")

    frames: list[FrameCalibration] = []
    for path in paths:
        try:
            frames.append(load_frame_calibration(path))
        except (ValueError, KeyError) as error:
            print(f"Skipping {path}: {error}")
    names = frames[0].names
    for frame in frames:
        if frame.names != names:
            raise ValueError(f"Camera names differ in {frame.frame}: {frame.names} vs {names}")

    # Normalize per-frame scale so that each frame's mean pairwise baseline
    # matches the median across frames (VGGT scale is arbitrary per inference).
    baselines = np.asarray([mean_baseline(frame.centers, names) for frame in frames], dtype=np.float64)
    reference_baseline = float(np.median(baselines))
    scaled_centers: list[dict[str, np.ndarray]] = []
    for frame, baseline in zip(frames, baselines):
        factor = reference_baseline / baseline if baseline > 1e-12 else 1.0
        scaled_centers.append({name: frame.centers[name] * factor for name in names})

    # Aligned poses used for consensus/deviation/trim. In "scale" mode this is
    # the scale-normalized centers + raw rotations (relies on the backbone pinning
    # a camera, e.g. VGGT pins camera A to identity). In "procrustes" mode every
    # frame is similarity-aligned to the running consensus, so the aggregation is
    # gauge-invariant and survives a backbone that does not pin a camera.
    align_mode = getattr(args, "align_mode", "procrustes")
    orig_centers = [{name: frame.centers[name] for name in names} for frame in frames]
    orig_rotations = [{name: frame.rotations[name] for name in names} for frame in frames]
    aligned_centers = [dict(c) for c in scaled_centers]
    aligned_rotations = [dict(r) for r in orig_rotations]

    def _consensus(acent, arot, mask):
        rmean, cmean = {}, {}
        for name in names:
            rmean[name] = chordal_mean_rotation([arot[i][name] for i in range(len(frames)) if mask[i]])
            cmean[name] = np.median(np.asarray([acent[i][name] for i in range(len(frames)) if mask[i]]), axis=0)
        return rmean, cmean

    def _align_to(cmean):
        target = np.asarray([cmean[name] for name in names])
        acent, arot = [], []
        for i in range(len(frames)):
            src = np.asarray([orig_centers[i][name] for name in names])
            rot_g, t_g, s_g = umeyama_similarity(src, target)
            acent.append({name: s_g * (rot_g @ orig_centers[i][name]) + t_g for name in names})
            arot.append({name: orig_rotations[i][name] @ rot_g.T for name in names})
        return acent, arot

    inlier_mask = np.ones(len(frames), dtype=bool)
    rotation_means: dict[str, np.ndarray] = {}
    center_means: dict[str, np.ndarray] = {}
    rotation_dev = np.zeros((len(frames), len(names)), dtype=np.float64)
    center_dev = np.zeros((len(frames), len(names)), dtype=np.float64)

    for _ in range(max(args.trim_iterations, 1)):
        rotation_means, center_means = _consensus(aligned_centers, aligned_rotations, inlier_mask)
        if align_mode == "procrustes":
            aligned_centers, aligned_rotations = _align_to(center_means)
            rotation_means, center_means = _consensus(aligned_centers, aligned_rotations, inlier_mask)
        for cam_idx, name in enumerate(names):
            for i in range(len(frames)):
                rotation_dev[i, cam_idx] = geodesic_deg(rotation_means[name], aligned_rotations[i][name])
                center_dev[i, cam_idx] = float(np.linalg.norm(aligned_centers[i][name] - center_means[name]))
        mode = getattr(args, "exclude_mode", "absolute")
        max_rot = getattr(args, "max_rotation_deg", 20.0)
        max_cen_frac = getattr(args, "max_center_frac", 0.30)
        new_mask = np.ones(len(frames), dtype=bool)
        for cam_idx, name in enumerate(names):
            rot_sigma = robust_sigma(rotation_dev[inlier_mask, cam_idx], floor=0.05)
            cen_sigma = robust_sigma(center_dev[inlier_mask, cam_idx], floor=1e-6 * reference_baseline)
            # Loosened default: keep everything except catastrophic geometry blow-ups
            # (a camera flipped/displaced far from the across-frame consensus). The
            # robust aggregate already absorbs ordinary spread, so the strict per-sigma
            # gate only wasted good frames; it is available via --exclude-mode sigma.
            rot_thr = np.inf
            cen_thr = np.inf
            if mode in ("sigma", "both"):
                rot_thr = max(args.trim_z * rot_sigma, args.min_rotation_keep_deg)
                cen_thr = args.trim_z * cen_sigma + 1e-12
            if mode in ("absolute", "both"):
                rot_thr = min(rot_thr, max_rot)
                cen_thr = min(cen_thr, max_cen_frac * reference_baseline)
            new_mask &= rotation_dev[:, cam_idx] <= rot_thr
            new_mask &= center_dev[:, cam_idx] <= cen_thr
            
        if getattr(args, "keep_best_percent", 100.0) < 100.0:
            scores = np.max(rotation_dev, axis=1)
            k = max(3, int(len(frames) * args.keep_best_percent / 100.0))
            if k < len(scores):
                threshold = np.sort(scores)[k - 1]
                new_mask &= scores <= threshold

        min_inlier_req = max(3, int(args.min_inlier_fraction * len(frames)))
        if getattr(args, "keep_best_percent", 100.0) < (args.min_inlier_fraction * 100.0):
            min_inlier_req = max(3, int(len(frames) * args.keep_best_percent / 100.0))

        if new_mask.sum() < min_inlier_req:
            break
        if np.array_equal(new_mask, inlier_mask):
            break
        inlier_mask = new_mask

    # Final consensus over inliers. In procrustes mode the consensus lives in an
    # arbitrary aligned frame, so re-gauge it to the conventional output frame
    # (first camera at identity/origin) for drop-in compatibility.
    rotation_means, center_means = _consensus(aligned_centers, aligned_rotations, inlier_mask)
    if align_mode == "procrustes":
        anchor = names[0]
        gauge = rotation_means[anchor].copy()
        anchor_center = center_means[anchor].copy()
        rotation_means = {name: rotation_means[name] @ gauge.T for name in names}
        center_means = {name: gauge @ (center_means[name] - anchor_center) for name in names}

    inlier_indices = [i for i in range(len(frames)) if inlier_mask[i]]

    # Final aggregates and empirical sigmas over inlier frames.
    matrices_out: dict[str, np.ndarray] = {}
    camera_stats: dict[str, dict[str, Any]] = {}
    for cam_idx, name in enumerate(names):
        fx = np.asarray([frames[i].matrices[name][0, 0] for i in inlier_indices])
        fy = np.asarray([frames[i].matrices[name][1, 1] for i in inlier_indices])
        cx = np.asarray([frames[i].matrices[name][0, 2] for i in inlier_indices])
        cy = np.asarray([frames[i].matrices[name][1, 2] for i in inlier_indices])
        matrix = frames[0].matrices[name].copy()
        matrix[0, 0] = float(np.median(fx))
        matrix[1, 1] = float(np.median(fy))
        matrix[0, 2] = float(np.median(cx))
        matrix[1, 2] = float(np.median(cy))
        matrices_out[name] = matrix
        rotation_sigma_deg = robust_sigma(rotation_dev[inlier_mask, cam_idx], floor=0.02)
        center_sigma = robust_sigma(center_dev[inlier_mask, cam_idx], floor=1e-6 * reference_baseline)
        camera_stats[name] = {
            "rotation_sigma_deg": rotation_sigma_deg,
            "center_sigma": center_sigma,
            "center_sigma_baseline_fraction": center_sigma / reference_baseline,
            "fx_median": float(np.median(fx)),
            "fx_sigma": robust_sigma(fx),
            "fx_sigma_percent": 100.0 * robust_sigma(fx) / float(np.median(fx)),
            "fy_median": float(np.median(fy)),
            "fy_sigma": robust_sigma(fy),
            "fy_sigma_percent": 100.0 * robust_sigma(fy) / float(np.median(fy)),
            "principal_point_sigma_px": float(
                math.hypot(robust_sigma(cx, floor=1e-6), robust_sigma(cy, floor=1e-6))
            ),
        }

    # Pairwise relative geometry stats (gauge-invariant per frame).
    pair_stats: dict[str, dict[str, Any]] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            relative_mean = chordal_mean_rotation(
                [frames[k].rotations[name_a] @ frames[k].rotations[name_b].T for k in inlier_indices]
            )
            relative_dev = np.asarray(
                [
                    geodesic_deg(relative_mean, frames[k].rotations[name_a] @ frames[k].rotations[name_b].T)
                    for k in inlier_indices
                ]
            )
            baseline_vectors = np.asarray(
                [
                    frames[k].rotations[name_a] @ (scaled_centers[k][name_b] - scaled_centers[k][name_a])
                    for k in inlier_indices
                ]
            )
            baseline_median = np.median(baseline_vectors, axis=0)
            baseline_dev = np.linalg.norm(baseline_vectors - baseline_median, axis=1)
            pair_stats[f"{name_a}-{name_b}"] = {
                "camera_a": name_a,
                "camera_b": name_b,
                "relative_rotation_sigma_deg": robust_sigma(relative_dev, floor=0.02),
                "baseline_vector_sigma": robust_sigma(baseline_dev, floor=1e-6 * reference_baseline),
                "baseline_length_median": float(np.linalg.norm(baseline_median)),
            }

    output_dir = args.output_dir or dataset / "vggt_prior"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prior_calibration(
        output_dir / "calibration_prior.toml",
        frames[0],
        matrices_out,
        rotation_means,
        center_means,
    )

    deviation_rows = []
    for i, frame in enumerate(frames):
        row: dict[str, Any] = {"frame": frame.frame, "inlier": bool(inlier_mask[i])}
        for cam_idx, name in enumerate(names):
            row[f"rotation_dev_deg_{name}"] = rotation_dev[i, cam_idx]
            row[f"center_dev_{name}"] = center_dev[i, cam_idx]
        deviation_rows.append(row)
    with (output_dir / "frame_deviations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deviation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(deviation_rows)

    stats = {
        "dataset": str(dataset),
        "num_frames": len(frames),
        "num_inlier_frames": int(inlier_mask.sum()),
        "trim_z": args.trim_z,
        "reference_mean_baseline": reference_baseline,
        "camera_names": names,
        "cameras": camera_stats,
        "pairs": pair_stats,
        "outputs": {
            "calibration_prior": str(output_dir / "calibration_prior.toml"),
            "frame_deviations": str(output_dir / "frame_deviations.csv"),
        },
    }
    (output_dir / "prior_stats.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory containing frame_XXXXXX/calibration.toml")
    parser.add_argument("--frame-glob", default="frame_*", help="Glob for frame directories (default: frame_*)")
    parser.add_argument("--output-dir", type=Path, help="Default: DATASET_DIR/vggt_prior")
    parser.add_argument("--max-frames", type=int, default=0, help="Subsample at most this many frames (0 = all)")
    parser.add_argument("--align-mode", choices=["procrustes", "scale"], default="procrustes",
                        help="procrustes (default): similarity-align each frame to the consensus before "
                             "averaging (gauge-invariant, robust to a backbone that does not pin a camera); "
                             "scale: scale-normalize only (original behavior, relies on the first camera being pinned)")
    parser.add_argument("--keep-best-percent", type=float, default=100.0,
                        help="Keep only this top percentage of most consistent frames (0-100)")
    parser.add_argument("--exclude-mode", choices=["absolute", "sigma", "both"], default="absolute",
                        help="absolute (default, loosened): drop only catastrophic geometry blow-ups; "
                             "sigma: original strict per-sigma trimming; both: apply whichever is tighter")
    parser.add_argument("--max-rotation-deg", type=float, default=20.0,
                        help="absolute/both: exclude if a camera rotates more than this from the consensus")
    parser.add_argument("--max-center-frac", type=float, default=0.30,
                        help="absolute/both: exclude if a camera center is farther than this fraction of baseline from consensus")
    parser.add_argument("--trim-z", type=float, default=4.0, help="sigma/both: outlier rejection threshold in robust sigmas")
    parser.add_argument("--trim-iterations", type=int, default=3)
    parser.add_argument("--min-rotation-keep-deg", type=float, default=0.5, help="sigma/both: never trim rotation deviations below this")
    parser.add_argument("--min-inlier-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = aggregate(args)
    print(json.dumps(
        {
            "num_frames": stats["num_frames"],
            "num_inlier_frames": stats["num_inlier_frames"],
            "cameras": {
                name: {
                    "rotation_sigma_deg": round(value["rotation_sigma_deg"], 4),
                    "center_sigma_baseline_fraction": round(value["center_sigma_baseline_fraction"], 5),
                    "fx_median": round(value["fx_median"], 2),
                    "fx_sigma_percent": round(value["fx_sigma_percent"], 3),
                }
                for name, value in stats["cameras"].items()
            },
            "outputs": stats["outputs"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
