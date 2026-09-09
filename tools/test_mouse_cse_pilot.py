#!/usr/bin/env python3
"""Test a CSE-like mouse surface atlas on an exported Blender frame."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--vertex-keypoint-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument("--embedding-dim", type=int, default=16)
    return parser.parse_args()


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        width, height = (int(value) for value in dimensions.split())
        scale = float(handle.readline().decode("ascii").strip())
        channels = 3 if header == "PF" else 1
        dtype = "<f4" if scale < 0 else ">f4"
        values = np.fromfile(handle, dtype=dtype)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(values.reshape(shape)).astype(np.float32, copy=False)


def spectral_embedding(
    vertices: np.ndarray,
    edges: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(lengths, 1e-6)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([weights, weights])
    adjacency = coo_matrix(
        (values, (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    laplacian = csgraph.laplacian(adjacency, normed=True)
    eigenvalues, eigenvectors = eigsh(
        laplacian,
        k=dimension + 1,
        sigma=1e-6,
        which="LM",
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order][1 : dimension + 1]
    eigenvectors = eigenvectors[:, order][:, 1 : dimension + 1]
    for column in range(eigenvectors.shape[1]):
        anchor = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[anchor, column] < 0:
            eigenvectors[:, column] *= -1
    eigenvectors /= np.maximum(eigenvectors.std(axis=0, keepdims=True), 1e-8)
    eigenvectors /= np.maximum(
        np.linalg.norm(eigenvectors, axis=1, keepdims=True),
        1e-8,
    )
    return eigenvectors.astype(np.float32), eigenvalues.astype(np.float32)


def xyz_embedding(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - np.median(vertices, axis=0, keepdims=True)
    scale = np.percentile(np.abs(centered), 95, axis=0, keepdims=True)
    normalized = centered / np.maximum(scale, 1e-6)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-8)
    return normalized.astype(np.float32)


def backproject(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    camera = np.stack(
        [
            (x - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
            (y - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
            depth,
        ],
        axis=1,
    )
    return (camera - extrinsic[:3, 3]) @ extrinsic[:3, :3]


def project(
    world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    camera = world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    homogeneous = camera @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def support_vertex_groups(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        mapping = pickle.load(handle)
    groups: dict[str, list[int]] = {}
    for vertex_id, name in mapping.items():
        groups.setdefault(str(name), []).append(int(vertex_id))
    return {
        name: np.asarray(sorted(vertex_ids), dtype=np.int32)
        for name, vertex_ids in groups.items()
    }


def descriptor_colors(embedding: np.ndarray) -> np.ndarray:
    values = embedding[:, :3].copy()
    low = np.percentile(values, 1, axis=0, keepdims=True)
    high = np.percentile(values, 99, axis=0, keepdims=True)
    values = np.clip((values - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
    return (values * 255.0).astype(np.uint8)


def recover_keypoint(
    codebook: np.ndarray,
    support_ids: np.ndarray,
    available_ids: np.ndarray,
    available_uv: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(codebook[available_ids])
    distance, index = tree.query(codebook[support_ids], k=1, workers=-1)
    predicted_uv = available_uv[index]
    finite = np.isfinite(distance) & np.isfinite(predicted_uv).all(axis=1)
    if not finite.any():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    distance = distance[finite]
    predicted_uv = predicted_uv[finite]
    positive = distance[distance > 1e-8]
    scale = float(np.median(positive)) if positive.size else 1e-3
    weights = np.exp(-distance / max(scale, 1e-3))
    return np.average(predicted_uv, axis=0, weights=weights).astype(np.float32)


def summarize_errors(records: list[dict], method: str, group: str) -> dict:
    values = [
        float(record[f"{method}_error_px"])
        for record in records
        if record["group"] == group and np.isfinite(record[f"{method}_error_px"])
    ]
    return {
        "count": len(values),
        "mean_px": float(np.mean(values)) if values else None,
        "median_px": float(np.median(values)) if values else None,
        "p90_px": float(np.percentile(values, 90)) if values else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = np.load(args.mesh_npz)
    canonical = mesh["canonical_vertices"].astype(np.float64)
    posed = mesh["posed_world_vertices"].astype(np.float64)
    edges = mesh["edges"].astype(np.int32)
    spectral, eigenvalues = spectral_embedding(canonical, edges, args.embedding_dim)
    xyz = xyz_embedding(canonical)
    np.savez_compressed(
        args.output_dir / "mouse_surface_codebook.npz",
        spectral_embedding=spectral,
        xyz_embedding=xyz,
        eigenvalues=eigenvalues,
        canonical_vertices=canonical.astype(np.float32),
        edges=edges,
    )

    groups = support_vertex_groups(args.vertex_keypoint_map)
    vertex_tree = cKDTree(posed)
    spectral_colors = descriptor_colors(spectral)
    view_ids = [int(value.strip()) for value in args.views.split(",") if value.strip()]
    records: list[dict] = []
    view_summaries: list[dict] = []
    view_images: list[tuple[int, Image.Image, Image.Image, Image.Image]] = []
    covered_sets: dict[int, set[int]] = {}

    for view_id in view_ids:
        annotation_path = args.scene_root / "annotations" / f"{view_id:08d}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        intrinsic = np.asarray(annotation["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(annotation["extrinsic"], dtype=np.float64)
        depth = read_pfm(
            args.scene_root / "rendered_depth_maps" / f"{view_id:08d}.pfm"
        )
        body_mask = np.asarray(
            Image.open(args.scene_root / "body_masks" / f"{view_id:08d}.png").convert("L")
        ) > 0
        valid = body_mask & np.isfinite(depth) & (depth > 1e-6)
        y, x = np.nonzero(valid)
        world = backproject(
            x.astype(np.float64),
            y.astype(np.float64),
            depth[y, x].astype(np.float64),
            intrinsic,
            extrinsic,
        )
        surface_distance, vertex_ids = vertex_tree.query(world, k=1, workers=-1)
        covered_sets[view_id] = set(int(value) for value in np.unique(vertex_ids))

        counts = np.bincount(vertex_ids, minlength=len(posed)).astype(np.float64)
        sum_x = np.bincount(vertex_ids, weights=x, minlength=len(posed))
        sum_y = np.bincount(vertex_ids, weights=y, minlength=len(posed))
        available_ids = np.flatnonzero(counts > 0)
        available_uv = np.stack(
            [
                sum_x[available_ids] / counts[available_ids],
                sum_y[available_ids] / counts[available_ids],
            ],
            axis=1,
        ).astype(np.float32)

        color_map = np.zeros((*depth.shape, 3), dtype=np.uint8)
        color_map[y, x] = spectral_colors[vertex_ids]
        cse_image = Image.fromarray(color_map)
        rgb = Image.open(
            args.scene_root / "blended_images" / f"{view_id:08d}.jpg"
        ).convert("RGB")
        blend = Image.blend(rgb, cse_image, 0.55)
        prediction_overlay = rgb.copy()
        draw = ImageDraw.Draw(prediction_overlay)

        keypoints = {item["name"]: item for item in annotation["keypoints"]}
        for name, support_ids in groups.items():
            item = keypoints.get(name)
            if item is None or not item.get("visible") or not item.get("in_frame"):
                continue
            surface_world = np.asarray(item["surface_world_xyz"], dtype=np.float64)[None]
            target_uv = project(surface_world, intrinsic, extrinsic)[0]
            oracle_uv = project(
                posed[support_ids].mean(axis=0, keepdims=True),
                intrinsic,
                extrinsic,
            )[0]
            spectral_uv = recover_keypoint(
                spectral,
                support_ids,
                available_ids,
                available_uv,
            )
            xyz_uv = recover_keypoint(
                xyz,
                support_ids,
                available_ids,
                available_uv,
            )
            group = "fine_finger" if "Finger" in name else "coarse_anchor"
            record = {
                "view_id": view_id,
                "camera": annotation["camera"],
                "name": name,
                "group": group,
                "target_uv": target_uv.tolist(),
                "oracle_uv": oracle_uv.tolist(),
                "spectral_uv": spectral_uv.tolist(),
                "xyz_uv": xyz_uv.tolist(),
                "oracle_error_px": float(np.linalg.norm(oracle_uv - target_uv)),
                "spectral_error_px": float(np.linalg.norm(spectral_uv - target_uv)),
                "xyz_error_px": float(np.linalg.norm(xyz_uv - target_uv)),
            }
            records.append(record)
            color = (0, 255, 255) if group == "fine_finger" else (255, 190, 0)
            tx, ty = target_uv
            px, py = spectral_uv
            draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), outline=(0, 255, 0), width=2)
            draw.line((px - 4, py, px + 4, py), fill=color, width=2)
            draw.line((px, py - 4, px, py + 4), fill=color, width=2)

        np.savez_compressed(
            args.output_dir / f"view_{view_id:08d}_dense_correspondence.npz",
            x=x.astype(np.int16),
            y=y.astype(np.int16),
            vertex_ids=vertex_ids.astype(np.int32),
            surface_distance=surface_distance.astype(np.float32),
        )
        view_summaries.append(
            {
                "view_id": view_id,
                "camera": annotation["camera"],
                "labeled_pixel_count": int(len(x)),
                "covered_vertex_count": int(len(available_ids)),
                "covered_vertex_fraction": float(len(available_ids) / len(posed)),
                "surface_distance_median": float(np.median(surface_distance)),
                "surface_distance_p90": float(np.percentile(surface_distance, 90)),
                "surface_distance_p99": float(np.percentile(surface_distance, 99)),
                "surface_distance_max": float(surface_distance.max()),
            }
        )
        view_images.append((view_id, rgb, cse_image, blend, prediction_overlay))

    pairwise_overlap = {}
    for index, left in enumerate(view_ids):
        for right in view_ids[index + 1 :]:
            intersection = len(covered_sets[left] & covered_sets[right])
            union = len(covered_sets[left] | covered_sets[right])
            pairwise_overlap[f"{left}-{right}"] = {
                "intersection_vertices": intersection,
                "union_vertices": union,
                "jaccard": intersection / max(union, 1),
                "fraction_of_smaller": intersection
                / max(min(len(covered_sets[left]), len(covered_sets[right])), 1),
            }

    error_summary = {}
    for method in ("oracle", "spectral", "xyz"):
        error_summary[method] = {
            group: summarize_errors(records, method, group)
            for group in ("coarse_anchor", "fine_finger")
        }
        all_values = [
            float(record[f"{method}_error_px"])
            for record in records
            if np.isfinite(record[f"{method}_error_px"])
        ]
        error_summary[method]["all"] = {
            "count": len(all_values),
            "mean_px": float(np.mean(all_values)),
            "median_px": float(np.median(all_values)),
            "p90_px": float(np.percentile(all_values, 90)),
        }

    union_coverage = set().union(*(covered_sets[view_id] for view_id in view_ids))
    report = {
        "format": "mousegeoforge_mouse_cse_pilot_report_v1",
        "mesh_npz": str(args.mesh_npz.resolve()),
        "scene_root": str(args.scene_root.resolve()),
        "frame": int(mesh["frame"]),
        "vertex_count": int(len(canonical)),
        "embedding_dim": int(args.embedding_dim),
        "eigenvalues": eigenvalues.tolist(),
        "views": view_summaries,
        "union_covered_vertex_count": len(union_coverage),
        "union_covered_vertex_fraction": len(union_coverage) / len(canonical),
        "pairwise_coverage": pairwise_overlap,
        "keypoint_error_summary": error_summary,
        "keypoint_records": records,
    }
    (args.output_dir / "cse_pilot_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    tile_width, tile_height = 600, 400
    contact = Image.new(
        "RGB",
        (tile_width * len(view_images), tile_height * 4),
        (16, 16, 16),
    )
    for column, (view_id, rgb, cse, blend, overlay) in enumerate(view_images):
        for row, image in enumerate((rgb, cse, blend, overlay)):
            contact.paste(
                image.resize((tile_width, tile_height), Image.Resampling.LANCZOS),
                (column * tile_width, row * tile_height),
            )
    contact.save(args.output_dir / "cse_dense_correspondence_contact_sheet.jpg", quality=94)

    groups_for_plot = ("coarse_anchor", "fine_finger", "all")
    x_axis = np.arange(len(groups_for_plot))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4.6))
    spectral_values = [
        error_summary["spectral"][group]["median_px"] for group in groups_for_plot
    ]
    xyz_values = [
        error_summary["xyz"][group]["median_px"] for group in groups_for_plot
    ]
    axis.bar(x_axis - width / 2, spectral_values, width, label="16D spectral CSE")
    axis.bar(x_axis + width / 2, xyz_values, width, label="canonical XYZ")
    axis.set_xticks(x_axis, ["coarse anchors", "finger joints", "all"])
    axis.set_ylabel("Median surface-query error (px)")
    axis.set_title("Recovering keypoint groups from one dense surface map")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "cse_keypoint_query_error.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report["keypoint_error_summary"], indent=2))
    print(
        "cse_pilot_complete",
        f"union_coverage={report['union_covered_vertex_fraction']:.4f}",
        f"records={len(records)}",
        f"output={args.output_dir}",
    )


if __name__ == "__main__":
    main()
