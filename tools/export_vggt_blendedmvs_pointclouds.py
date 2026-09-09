#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from check_blendedmvs_depth_consistency import load_scene, read_pfm  # noqa: E402


def unproject_depth_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid_y, valid_x = np.nonzero(np.isfinite(depth) & (depth > 1e-8))
    if len(valid_x) > max_points:
        take = rng.choice(len(valid_x), size=max_points, replace=False)
        valid_x = valid_x[take]
        valid_y = valid_y[take]
    z = depth[valid_y, valid_x]
    x = (valid_x.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (valid_y.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack([x, y, z], axis=1)
    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    world_points = (r.T @ (camera_points - t).T).T.astype(np.float32)
    colors = rgb[valid_y, valid_x].astype(np.uint8)
    return world_points, colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32, copy=False)
    colors = colors.astype(np.uint8, copy=False)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    arr = np.empty(len(points), dtype=dtype)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["red"] = colors[:, 0]
    arr["green"] = colors[:, 1]
    arr["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def sample_arrays(points: np.ndarray, colors: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    take = rng.choice(len(points), size=max_points, replace=False)
    return points[take], colors[take]


def label_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_panel(
    points: np.ndarray,
    colors: np.ndarray,
    axes: tuple[int, int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    size: int,
    title: str,
) -> Image.Image:
    out = Image.new("RGB", (size, size + 34), (248, 248, 248))
    draw = ImageDraw.Draw(out)
    font = label_font(16)
    draw.rectangle([0, 0, size, 34], fill=(0, 0, 0))
    draw.text((8, 8), title[:70], fill=(255, 255, 255), font=font)

    xy = points[:, axes]
    lo = bbox_min[list(axes)]
    hi = bbox_max[list(axes)]
    span = np.maximum(hi - lo, 1e-8)
    pad = 0.04
    norm = (xy - lo) / span
    px = np.clip((pad + (1 - 2 * pad) * norm[:, 0]) * (size - 1), 0, size - 1).astype(np.int32)
    py = np.clip((pad + (1 - 2 * pad) * (1.0 - norm[:, 1])) * (size - 1), 0, size - 1).astype(np.int32) + 34
    arr = np.asarray(out).copy()
    order = np.argsort(points[:, 3 - axes[0] - axes[1]]) if set(axes) != {0, 1, 2} else np.arange(len(points))
    for idx in order:
        arr[py[idx], px[idx]] = colors[idx]
    return Image.fromarray(arr)


def render_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    title: str,
    rng: np.random.Generator,
    max_preview_points: int,
    panel_size: int,
) -> None:
    pts, cols = sample_arrays(points, colors, max_preview_points, rng)
    panels = [
        render_panel(pts, cols, (0, 1), bbox_min, bbox_max, panel_size, f"{title} XY"),
        render_panel(pts, cols, (0, 2), bbox_min, bbox_max, panel_size, f"{title} XZ"),
        render_panel(pts, cols, (1, 2), bbox_min, bbox_max, panel_size, f"{title} YZ"),
    ]
    out = Image.new("RGB", (panel_size * 3 + 24, panel_size + 34 + 16), (235, 235, 235))
    x = 6
    for panel in panels:
        out.paste(panel, (x, 8))
        x += panel_size + 6
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=92)


def fuse_pointcloud(
    scene_root: Path,
    view_ids: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    points_per_view: int,
    pfm_flip_y: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    frames, _ = load_scene(scene_root)
    all_points = []
    all_colors = []
    for idx, view_id_raw in enumerate(view_ids):
        view_id = int(view_id_raw)
        frame = frames[view_id]
        depth = read_pfm(frame.depth_path, flip_y=pfm_flip_y)
        rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
        points, colors = unproject_depth_points(
            depth,
            rgb,
            extrinsics[idx],
            intrinsics[idx],
            max_points=points_per_view,
            rng=rng,
        )
        all_points.append(points)
        all_colors.append(colors)
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)


def robust_bbox(point_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_points = np.concatenate(point_sets, axis=0)
    lo = np.percentile(all_points, 1.0, axis=0)
    hi = np.percentile(all_points, 99.0, axis=0)
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-6)
    half[:] = max(float(half.max()), 1e-6)
    return center - half, center + half


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    entries = []
    for row in rows:
        entries.append(
            "<tr>"
            f"<td>{html.escape(row['scene'])}</td>"
            f"<td><a href=\"{html.escape(Path(row['gt_ply']).name)}\">GT PLY</a></td>"
            f"<td><a href=\"{html.escape(Path(row['vggt_ply']).name)}\">VGGT PLY</a></td>"
            f"<td><img src=\"{html.escape(Path(row['gt_preview']).name)}\" style=\"max-width: 520px\"></td>"
            f"<td><img src=\"{html.escape(Path(row['vggt_preview']).name)}\" style=\"max-width: 520px\"></td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>VGGT-Omega vs BlendedMVS GT point clouds</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
th {{ background: #eee; }}
</style>
<h1>VGGT-Omega vs BlendedMVS GT Point Clouds</h1>
<p>Each scene is exported as two separate point-cloud files. Both use the same BlendedMVS RGB/depth samples; only the camera set changes. VGGT uses the Sim(3)-aligned VGGT-Omega predicted camera and predicted intrinsics from the earlier evaluation.</p>
<table>
<thead><tr><th>scene</th><th>GT control</th><th>VGGT-Omega</th><th>GT preview</th><th>VGGT preview</th></tr></thead>
<tbody>
{''.join(entries)}
</tbody></table>
"""
    path.write_text(doc)


def run(args: argparse.Namespace) -> None:
    pred_dir = Path(args.prediction_dir)
    scene_root = Path(args.scene_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    pred_files = sorted(pred_dir.glob("*_vggt_predictions.npz"))[: args.max_scenes]
    if not pred_files:
        raise FileNotFoundError(f"No *_vggt_predictions.npz files found under {pred_dir}")

    for pred_path in pred_files:
        scene = pred_path.name[: -len("_vggt_predictions.npz")]
        data = np.load(pred_path)
        view_ids = data["view_ids"]
        scene_dir = scene_root / scene
        if not scene_dir.exists():
            print(f"skip missing scene: {scene_dir}")
            continue
        gt_points, gt_colors = fuse_pointcloud(
            scene_dir,
            view_ids,
            data["gt_extrinsics"],
            data["gt_intrinsics"],
            args.points_per_view,
            args.pfm_flip_y,
            rng,
        )
        if args.vggt_intrinsics == "pred":
            vggt_intrinsics = data["pred_intrinsics_orig"]
            suffix = "vggt_omega_predK"
        else:
            vggt_intrinsics = data["gt_intrinsics"]
            suffix = "vggt_omega_gtK"
        vggt_points, vggt_colors = fuse_pointcloud(
            scene_dir,
            view_ids,
            data["pred_extrinsics_aligned"],
            vggt_intrinsics,
            args.points_per_view,
            args.pfm_flip_y,
            rng,
        )
        bbox_min, bbox_max = robust_bbox([gt_points, vggt_points])
        gt_ply = output_dir / f"{scene}_gt_control.ply"
        vggt_ply = output_dir / f"{scene}_{suffix}.ply"
        gt_preview = output_dir / f"{scene}_gt_control_preview.jpg"
        vggt_preview = output_dir / f"{scene}_{suffix}_preview.jpg"
        write_ply(gt_ply, gt_points, gt_colors)
        write_ply(vggt_ply, vggt_points, vggt_colors)
        render_preview(
            gt_preview,
            gt_points,
            gt_colors,
            bbox_min,
            bbox_max,
            f"{scene} GT",
            rng,
            args.max_preview_points,
            args.preview_panel_size,
        )
        render_preview(
            vggt_preview,
            vggt_points,
            vggt_colors,
            bbox_min,
            bbox_max,
            f"{scene} VGGT",
            rng,
            args.max_preview_points,
            args.preview_panel_size,
        )
        rows.append(
            {
                "scene": scene,
                "view_count": int(len(view_ids)),
                "gt_points": int(len(gt_points)),
                "vggt_points": int(len(vggt_points)),
                "gt_ply": str(gt_ply),
                "vggt_ply": str(vggt_ply),
                "gt_preview": str(gt_preview),
                "vggt_preview": str(vggt_preview),
            }
        )
        print(f"{scene}: wrote {gt_ply.name} and {vggt_ply.name}")

    write_html(output_dir / "index.html", rows)
    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_dir / 'index.html'}")
    print(f"wrote {output_dir / 'manifest.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="output/vggt_omega_blendedmvs_eval_7scene_8view")
    parser.add_argument("--scene-root", default="dataset/blendedmvs_sample")
    parser.add_argument("--output", default="output/vggt_omega_blendedmvs_pointclouds_7scene_8view")
    parser.add_argument("--max-scenes", type=int, default=7)
    parser.add_argument("--points-per-view", type=int, default=12000)
    parser.add_argument("--max-preview-points", type=int, default=90000)
    parser.add_argument("--preview-panel-size", type=int, default=420)
    parser.add_argument("--vggt-intrinsics", choices=["pred", "gt"], default="pred")
    parser.add_argument("--pfm-flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
