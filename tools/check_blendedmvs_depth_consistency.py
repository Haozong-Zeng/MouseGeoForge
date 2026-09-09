#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FrameData:
    scene: str
    view_id: int
    image_path: Path
    masked_image_path: Path
    depth_path: Path
    cam_path: Path
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_params: tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str


def read_pfm(path: Path, flip_y: bool) -> np.ndarray:
    with path.open("rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Not a PFM file: {path}")
        dims_line = f.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = f.readline().decode("ascii").strip()
        width, height = map(int, dims_line.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, endian + "f", width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    arr = np.reshape(data, shape)
    if flip_y:
        arr = np.flipud(arr)
    if channels == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float64)


def read_cam(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    ext_start = lines.index("extrinsic") + 1
    extrinsic = np.array([[float(v) for v in lines[ext_start + i].split()] for i in range(4)], dtype=np.float64)
    int_start = lines.index("intrinsic") + 1
    intrinsic = np.array([[float(v) for v in lines[int_start + i].split()] for i in range(3)], dtype=np.float64)
    depth_params: tuple[float, ...] = ()
    if len(lines) > int_start + 3:
        depth_params = tuple(float(v) for v in lines[int_start + 3].split())
    return extrinsic, intrinsic, depth_params


def read_pair_file(path: Path) -> dict[int, list[int]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    pair_map: dict[int, list[int]] = {}
    count = int(lines[0])
    idx = 1
    for _ in range(count):
        ref = int(lines[idx])
        idx += 1
        parts = lines[idx].split()
        idx += 1
        src_count = int(parts[0])
        srcs = [int(parts[1 + 2 * i]) for i in range(src_count)]
        pair_map[ref] = srcs
    return pair_map


def invert_w2c(extrinsic: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


def camera_matrix(frame: FrameData, candidate: Candidate) -> np.ndarray:
    if candidate.name == "as_is_w2c":
        return frame.extrinsic
    if candidate.name == "inverted_as_c2w":
        return invert_w2c(frame.extrinsic)
    if candidate.name == "transpose_r_keep_t":
        out = frame.extrinsic.copy()
        out[:3, :3] = out[:3, :3].T
        return out
    raise ValueError(f"Unknown candidate: {candidate.name}")


def make_candidates() -> list[Candidate]:
    return [
        Candidate("as_is_w2c", "Use BlendedMVS extrinsic as OpenCV-style world-to-camera."),
        Candidate("inverted_as_c2w", "Treat file extrinsic as camera-to-world, invert it."),
        Candidate("transpose_r_keep_t", "Transpose R while keeping T; diagnostic for row/column confusion."),
    ]


def load_scene(scene_root: Path) -> tuple[dict[int, FrameData], dict[int, list[int]]]:
    frames: dict[int, FrameData] = {}
    for cam_path in sorted((scene_root / "cams").glob("*_cam.txt")):
        view_id = int(cam_path.stem.split("_")[0])
        image_path = scene_root / "blended_images" / f"{view_id:08d}.jpg"
        masked_image_path = scene_root / "blended_images" / f"{view_id:08d}_masked.jpg"
        depth_path = scene_root / "rendered_depth_maps" / f"{view_id:08d}.pfm"
        if not image_path.exists() or not depth_path.exists():
            continue
        extrinsic, intrinsic, depth_params = read_cam(cam_path)
        frames[view_id] = FrameData(
            scene=scene_root.name,
            view_id=view_id,
            image_path=image_path,
            masked_image_path=masked_image_path,
            depth_path=depth_path,
            cam_path=cam_path,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_params=depth_params,
        )
    pair_path = scene_root / "cams" / "pair.txt"
    pair_map = read_pair_file(pair_path) if pair_path.exists() else {}
    return frames, pair_map


def discover_scenes(root: Path) -> list[Path]:
    if (root / "cams").exists():
        return [root]
    scenes = [path for path in sorted(root.iterdir()) if (path / "cams").exists()]
    if not scenes:
        raise ValueError(f"No BlendedMVS scenes found under {root}")
    return scenes


def project_points(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = (extrinsic[:3, :3] @ points_world.T + extrinsic[:3, 3:4]).T
    z = cam[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    positive = z > 1e-8
    uv[positive, 0] = intrinsic[0, 0] * cam[positive, 0] / z[positive] + intrinsic[0, 2]
    uv[positive, 1] = intrinsic[1, 1] * cam[positive, 1] / z[positive] + intrinsic[1, 2]
    return uv, z, positive


def rounded_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    x = np.full(len(uv), -1, dtype=np.int64)
    y = np.full(len(uv), -1, dtype=np.int64)
    x[finite] = np.rint(uv[finite, 0]).astype(np.int64)
    y[finite] = np.rint(uv[finite, 1]).astype(np.int64)
    return x, y, finite


def unproject_depth_sample(
    frame: FrameData,
    extrinsic: np.ndarray,
    depth: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_y, valid_x = np.nonzero(np.isfinite(depth) & (depth > 1e-8))
    if len(valid_x) > max_points:
        take = rng.choice(len(valid_x), size=max_points, replace=False)
        valid_x = valid_x[take]
        valid_y = valid_y[take]
    z = depth[valid_y, valid_x]
    k = frame.intrinsic
    x = (valid_x.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (valid_y.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    cam = np.stack([x, y, z], axis=1)
    r = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return (r.T @ (cam - t).T).T


def reprojection_metrics(
    src: FrameData,
    dst: FrameData,
    src_extrinsic: np.ndarray,
    dst_extrinsic: np.ndarray,
    src_depth: np.ndarray,
    dst_depth: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    prefix: str,
) -> dict[str, float]:
    points_world = unproject_depth_sample(src, src_extrinsic, src_depth, max_points=max_points, rng=rng)
    n = len(points_world)
    if n == 0:
        return {
            f"{prefix}sample_count": 0.0,
            f"{prefix}in_image_rate": float("nan"),
            f"{prefix}valid_target_rate": float("nan"),
            f"{prefix}median_relative_error": float("nan"),
            f"{prefix}agree_1pct_rate": float("nan"),
            f"{prefix}agree_5pct_rate": float("nan"),
            f"{prefix}agree_10pct_rate": float("nan"),
        }

    uv, proj_z, positive = project_points(points_world, dst_extrinsic, dst.intrinsic)
    h, w = dst_depth.shape
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    target_z = np.full(n, np.nan, dtype=np.float64)
    target_z[inside] = dst_depth[y[inside], x[inside]]
    valid = inside & np.isfinite(target_z) & (target_z > 1e-8)
    rel = np.abs(proj_z[valid] - target_z[valid]) / np.clip(target_z[valid], 1e-8, None)
    return {
        f"{prefix}sample_count": float(n),
        f"{prefix}positive_z_rate": float(positive.mean()),
        f"{prefix}in_image_rate": float(inside.mean()),
        f"{prefix}valid_target_rate": float(valid.mean()),
        f"{prefix}median_relative_error": finite_median(rel),
        f"{prefix}agree_1pct_rate": float((rel < 0.01).mean()) if len(rel) else float("nan"),
        f"{prefix}agree_5pct_rate": float((rel < 0.05).mean()) if len(rel) else float("nan"),
        f"{prefix}agree_10pct_rate": float((rel < 0.10).mean()) if len(rel) else float("nan"),
    }


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def nan_to_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def score(row: dict[str, float]) -> float:
    return (
        4.0 * nan_to_zero(row.get("cross_valid_target_rate", float("nan")))
        + 3.0 * nan_to_zero(row.get("cross_agree_5pct_rate", float("nan")))
        + 2.0 * nan_to_zero(row.get("cross_agree_10pct_rate", float("nan")))
        + 1.0 * nan_to_zero(row.get("cross_in_image_rate", float("nan")))
        - 2.0 * min(nan_to_zero(row.get("cross_median_relative_error", float("nan"))), 2.0)
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = sorted(
        key
        for key in {k for row in rows for k in row}
        if key not in {"scene", "candidate", "description", "kind", "src_view", "dst_view"}
    )
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate"]].append(row)
    out_rows = []
    for candidate, cand_rows in by_candidate.items():
        out: dict[str, Any] = {"candidate": candidate, "sample_count": len(cand_rows)}
        for metric in metric_names:
            values = []
            for row in cand_rows:
                try:
                    values.append(float(row.get(metric, float("nan"))))
                except Exception:
                    pass
            if metric.endswith("_rate"):
                out[metric] = finite_mean(values)
            else:
                out[metric] = finite_median(np.asarray(values, dtype=np.float64))
        out["diagnostic_score"] = score(out)
        out_rows.append(out)
    out_rows.sort(key=lambda row: row["diagnostic_score"], reverse=True)
    for rank, row in enumerate(out_rows, start=1):
        row["rank"] = rank
    return out_rows


def label_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_depth_reprojection(
    src: FrameData,
    dst: FrameData,
    src_extrinsic: np.ndarray,
    dst_extrinsic: np.ndarray,
    src_depth: np.ndarray,
    dst_depth: np.ndarray,
    title: str,
    max_points: int,
    max_width: int,
    rng: np.random.Generator,
) -> Image.Image:
    image = Image.open(dst.image_path).convert("RGB")
    arr = np.asarray(image).copy()
    h, w = arr.shape[:2]
    points_world = unproject_depth_sample(src, src_extrinsic, src_depth, max_points=max_points, rng=rng)
    uv, proj_z, positive = project_points(points_world, dst_extrinsic, dst.intrinsic)
    x, y, finite = rounded_uv(uv)
    inside = finite & positive & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    target_z = np.full(len(points_world), np.nan, dtype=np.float64)
    target_z[inside] = dst_depth[y[inside], x[inside]]
    valid = inside & np.isfinite(target_z) & (target_z > 1e-8)
    rel = np.full(len(points_world), np.inf, dtype=np.float64)
    rel[valid] = np.abs(proj_z[valid] - target_z[valid]) / np.clip(target_z[valid], 1e-8, None)
    good = valid & (rel < 0.05)
    bad = valid & ~good
    no_depth = inside & ~valid
    for mask, color in [
        (no_depth, np.array([255, 210, 0], dtype=np.uint8)),
        (bad, np.array([255, 40, 40], dtype=np.uint8)),
        (good, np.array([0, 255, 80], dtype=np.uint8)),
    ]:
        xi = x[mask]
        yi = y[mask]
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            xx = xi + dx
            yy = yi + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            arr[yy[ok], xx[ok]] = color
    out = Image.fromarray(arr)
    if out.width > max_width:
        scale = max_width / out.width
        out = out.resize((max_width, int(out.height * scale)), Image.BICUBIC)
    draw = ImageDraw.Draw(out)
    font = label_font(18)
    draw.rectangle([0, 0, out.width, 42], fill=(0, 0, 0))
    draw.text((8, 10), title[:150], fill=(255, 255, 255), font=font)
    return out


def make_contact_sheet(images: list[Image.Image], columns: int = 2, margin: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * cell_w + (columns + 1) * margin, rows * cell_h + (rows + 1) * margin), (245, 245, 245))
    for idx, img in enumerate(images):
        row, col = divmod(idx, columns)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        sheet.paste(img, (x, y))
    return sheet


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict[str, Any]], detail_csv: Path, images: list[Path]) -> None:
    headers = [
        "rank",
        "candidate",
        "diagnostic_score",
        "self_in_image_rate",
        "self_valid_target_rate",
        "self_median_relative_error",
        "self_agree_1pct_rate",
        "cross_in_image_rate",
        "cross_valid_target_rate",
        "cross_median_relative_error",
        "cross_agree_1pct_rate",
        "cross_agree_5pct_rate",
        "cross_agree_10pct_rate",
    ]
    rows_html = []
    for row in summary:
        cells = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.5g}" if np.isfinite(value) else "nan"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    image_html = "\n".join(
        f'<h3>{html.escape(img.name)}</h3><img src="{html.escape(img.name)}" style="max-width: 100%; border: 1px solid #ccc;">'
        for img in images
    )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>BlendedMVS depth consistency check</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
th {{ background: #eee; }}
code {{ background: #f3f3f3; padding: 1px 3px; }}
</style>
<h1>BlendedMVS Depth Consistency Check</h1>
<p><code>self_*</code> is same-frame depth round-trip; it should be near-perfect. <code>cross_*</code> is source-depth to target-depth reprojection using the BlendedMVS pair file.</p>
<p>Overlay colors: green = projected source depth agrees within 5%, red = valid target depth but mismatch, yellow = in image without target depth.</p>
<p>Detailed metrics: <code>{html.escape(detail_csv.name)}</code></p>
<table><thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead><tbody>
{''.join(rows_html)}
</tbody></table>
{image_html}
"""
    path.write_text(doc)


def choose_pairs(frames: dict[int, FrameData], pair_map: dict[int, list[int]], srcs_per_ref: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for ref in sorted(pair_map):
        if ref not in frames:
            continue
        for src in pair_map[ref][:srcs_per_ref]:
            if src in frames:
                pairs.append((ref, src))
                if len(pairs) >= max_pairs:
                    return pairs
    if pairs:
        return pairs
    ids = sorted(frames)
    return list(zip(ids[:-1], ids[1:]))[:max_pairs]


def run(args: argparse.Namespace) -> None:
    root = Path(args.scene_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_candidates()
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    image_outputs: list[Path] = []

    for scene_root in discover_scenes(root)[: args.max_scenes]:
        frames, pair_map = load_scene(scene_root)
        if not frames:
            continue
        print(f"{scene_root.name}: frames={len(frames)}")
        depth_cache = {
            view_id: read_pfm(frame.depth_path, flip_y=args.pfm_flip_y)
            for view_id, frame in frames.items()
        }
        pairs = choose_pairs(frames, pair_map, args.srcs_per_ref, args.max_pairs_per_scene)
        print(f"  pairs={len(pairs)}")
        for candidate in candidates:
            cameras = {view_id: camera_matrix(frame, candidate) for view_id, frame in frames.items()}
            for view_id, frame in frames.items():
                row = {
                    "scene": frame.scene,
                    "candidate": candidate.name,
                    "description": candidate.description,
                    "kind": "self_depth_roundtrip",
                    "src_view": view_id,
                    "dst_view": view_id,
                }
                row.update(
                    reprojection_metrics(
                        frame,
                        frame,
                        cameras[view_id],
                        cameras[view_id],
                        depth_cache[view_id],
                        depth_cache[view_id],
                        max_points=args.max_depth_points,
                        rng=rng,
                        prefix="self_",
                    )
                )
                all_rows.append(row)
            for ref, src in pairs:
                row = {
                    "scene": frames[ref].scene,
                    "candidate": candidate.name,
                    "description": candidate.description,
                    "kind": "cross_depth_reprojection",
                    "src_view": ref,
                    "dst_view": src,
                }
                row.update(
                    reprojection_metrics(
                        frames[ref],
                        frames[src],
                        cameras[ref],
                        cameras[src],
                        depth_cache[ref],
                        depth_cache[src],
                        max_points=args.max_depth_points,
                        rng=rng,
                        prefix="cross_",
                    )
                )
                all_rows.append(row)

        if args.visualize:
            ref, src = pairs[0]
            images = []
            for candidate in candidates:
                src_extrinsic = camera_matrix(frames[ref], candidate)
                dst_extrinsic = camera_matrix(frames[src], candidate)
                images.append(
                    draw_depth_reprojection(
                        frames[ref],
                        frames[src],
                        src_extrinsic,
                        dst_extrinsic,
                        depth_cache[ref],
                        depth_cache[src],
                        f"{scene_root.name} {ref}->{src} | {candidate.name}",
                        max_points=args.max_depth_points,
                        max_width=args.overlay_width,
                        rng=rng,
                    )
                )
            sheet = make_contact_sheet(images, columns=1)
            out_path = output_dir / f"{scene_root.name}_{ref:08d}_to_{src:08d}_depth_reprojection.jpg"
            sheet.save(out_path, quality=92)
            image_outputs.append(out_path)

    summary = aggregate(all_rows)
    detail_csv = output_dir / "per_pair_metrics.csv"
    summary_csv = output_dir / "summary_by_candidate.csv"
    write_csv(detail_csv, all_rows)
    write_csv(summary_csv, summary)
    write_html(output_dir / "index.html", summary, detail_csv, image_outputs)
    print(f"wrote {summary_csv}")
    print(f"wrote {detail_csv}")
    print(f"wrote {output_dir / 'index.html'}")
    for row in summary:
        print(
            f"#{row['rank']} {row['candidate']}: score={row['diagnostic_score']:.4f}, "
            f"cross_valid={row.get('cross_valid_target_rate', float('nan')):.4f}, "
            f"cross_rel={row.get('cross_median_relative_error', float('nan')):.5f}, "
            f"cross_5pct={row.get('cross_agree_5pct_rate', float('nan')):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", default="dataset/blendedmvs_sample")
    parser.add_argument("--output", default="output/blendedmvs_depth_consistency")
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--max-pairs-per-scene", type=int, default=20)
    parser.add_argument("--srcs-per-ref", type=int, default=1)
    parser.add_argument("--max-depth-points", type=int, default=8000)
    parser.add_argument("--overlay-width", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pfm-flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
