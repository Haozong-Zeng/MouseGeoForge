#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "vggt-omega"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_blendedmvs_depth_consistency import (  # noqa: E402
    FrameData,
    aggregate,
    choose_pairs,
    discover_scenes,
    draw_depth_reprojection,
    load_scene,
    make_contact_sheet,
    read_pfm,
    reprojection_metrics,
    write_csv,
    write_html,
)
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def camera_center(extrinsic_3x4: np.ndarray) -> np.ndarray:
    r = extrinsic_3x4[:3, :3]
    t = extrinsic_3x4[:3, 3]
    return -r.T @ t


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t such that dst ~= s * R @ src + t."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 arrays, got {src.shape} and {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_mat[-1, -1] = -1
    r = u @ s_mat @ vt
    var = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.trace(np.diag(d) @ s_mat) / max(var, 1e-12))
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def align_predicted_to_gt_world(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Align predicted camera world to GT world.

    VGGT-Omega predicts camera-from-world extrinsics in an arbitrary world gauge.
    Estimate X_pred = s * A * X_gt + b from camera centers, then rewrite each
    predicted camera as camera-from-GT-world with GT-scale depth:
        R_aligned = R_pred @ A
        t_aligned = (R_pred @ b + t_pred) / s
    """
    pred_centers = np.stack([camera_center(e) for e in pred_extrinsics], axis=0)
    gt_centers = np.stack([camera_center(e) for e in gt_extrinsics], axis=0)
    scale, a_rot, b = umeyama(gt_centers, pred_centers)
    aligned = np.zeros_like(pred_extrinsics)
    aligned_centers = []
    for idx, pred in enumerate(pred_extrinsics):
        r_pred = pred[:3, :3]
        t_pred = pred[:3, 3]
        aligned[idx, :3, :3] = r_pred @ a_rot
        aligned[idx, :3, 3] = (r_pred @ b + t_pred) / scale
        aligned_centers.append(camera_center(aligned[idx]))
    aligned_centers = np.stack(aligned_centers, axis=0)
    center_err = np.linalg.norm(aligned_centers - gt_centers, axis=1)
    rot_err = []
    for pred_aligned, gt in zip(aligned, gt_extrinsics):
        r_delta = pred_aligned[:3, :3] @ gt[:3, :3].T
        cos_theta = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
        rot_err.append(float(np.degrees(np.arccos(cos_theta))))
    diagnostics = {
        "align_scale_gt_to_pred": scale,
        "align_center_rmse": float(np.sqrt(np.mean(center_err**2))),
        "align_center_median": float(np.median(center_err)),
        "align_rot_median_deg": float(np.median(rot_err)),
        "align_rot_mean_deg": float(np.mean(rot_err)),
    }
    return aligned, diagnostics


def scale_intrinsics_to_original(
    pred_intrinsics: np.ndarray,
    input_hw: tuple[int, int],
    frames: list[FrameData],
) -> np.ndarray:
    in_h, in_w = input_hw
    out = pred_intrinsics.copy()
    for idx, frame in enumerate(frames):
        with Image.open(frame.image_path) as img:
            orig_w, orig_h = img.size
        sx = orig_w / float(in_w)
        sy = orig_h / float(in_h)
        out[idx, 0, 0] *= sx
        out[idx, 0, 2] *= sx
        out[idx, 1, 1] *= sy
        out[idx, 1, 2] *= sy
    return out


def select_view_ids(
    frames: dict[int, FrameData],
    pair_map: dict[int, list[int]],
    max_views: int,
    srcs_per_ref: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for ref in sorted(pair_map):
        if ref not in frames:
            continue
        for view_id in [ref] + [src for src in pair_map[ref][:srcs_per_ref] if src in frames]:
            if view_id not in seen:
                selected.append(view_id)
                seen.add(view_id)
            if len(selected) >= max_views:
                return selected
    for view_id in sorted(frames):
        if view_id not in seen:
            selected.append(view_id)
            seen.add(view_id)
        if len(selected) >= max_views:
            break
    return selected


def state_dict_from_payload(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def load_model(checkpoint: Path, device: torch.device) -> VGGTOmega:
    model = VGGTOmega().eval()
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and payload.get("save_full") is False:
        base_checkpoint = payload.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError(f"Adapter checkpoint {checkpoint} does not record base_checkpoint")
        base_payload = torch.load(base_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict_from_payload(base_payload), strict=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys in {checkpoint}: {unexpected[:8]}")
        print(f"loaded_base_checkpoint={base_checkpoint}")
        print(f"loaded_adapter_checkpoint={checkpoint} missing_after_adapter={len(missing)}")
    else:
        model.load_state_dict(state_dict_from_payload(payload), strict=True)
        print(f"loaded_checkpoint={checkpoint}")
    return model.to(device)


def run_vggt(
    model: VGGTOmega,
    image_paths: list[str],
    image_resolution: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    input_hw = tuple(int(v) for v in images.shape[-2:])
    with torch.inference_mode():
        predictions = model(images)
        extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    extrinsic_np = extrinsic.detach().float().cpu().numpy()[0]
    intrinsic_np = intrinsic.detach().float().cpu().numpy()[0]
    del images, predictions, extrinsic, intrinsic
    torch.cuda.empty_cache()
    return extrinsic_np, intrinsic_np, input_hw


def add_metrics_rows(
    rows: list[dict[str, Any]],
    scene: str,
    candidate: str,
    description: str,
    frames: dict[int, FrameData],
    depth_cache: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    extrinsics: dict[int, np.ndarray],
    intrinsics: dict[int, np.ndarray],
    max_depth_points: int,
    rng: np.random.Generator,
    extra: dict[str, float] | None = None,
) -> None:
    extra = extra or {}
    for view_id, frame in frames.items():
        row = {
            "scene": scene,
            "candidate": candidate,
            "description": description,
            "kind": "self_depth_roundtrip",
            "src_view": view_id,
            "dst_view": view_id,
            **extra,
        }
        frame_for_intr = FrameData(
            scene=frame.scene,
            view_id=frame.view_id,
            image_path=frame.image_path,
            masked_image_path=frame.masked_image_path,
            depth_path=frame.depth_path,
            cam_path=frame.cam_path,
            extrinsic=frame.extrinsic,
            intrinsic=intrinsics[view_id],
            depth_params=frame.depth_params,
        )
        row.update(
            reprojection_metrics(
                frame_for_intr,
                frame_for_intr,
                extrinsics[view_id],
                extrinsics[view_id],
                depth_cache[view_id],
                depth_cache[view_id],
                max_points=max_depth_points,
                rng=rng,
                prefix="self_",
            )
        )
        rows.append(row)

    for src, dst in pairs:
        src_frame = frames[src]
        dst_frame = frames[dst]
        src_eval = FrameData(
            scene=src_frame.scene,
            view_id=src_frame.view_id,
            image_path=src_frame.image_path,
            masked_image_path=src_frame.masked_image_path,
            depth_path=src_frame.depth_path,
            cam_path=src_frame.cam_path,
            extrinsic=src_frame.extrinsic,
            intrinsic=intrinsics[src],
            depth_params=src_frame.depth_params,
        )
        dst_eval = FrameData(
            scene=dst_frame.scene,
            view_id=dst_frame.view_id,
            image_path=dst_frame.image_path,
            masked_image_path=dst_frame.masked_image_path,
            depth_path=dst_frame.depth_path,
            cam_path=dst_frame.cam_path,
            extrinsic=dst_frame.extrinsic,
            intrinsic=intrinsics[dst],
            depth_params=dst_frame.depth_params,
        )
        row = {
            "scene": scene,
            "candidate": candidate,
            "description": description,
            "kind": "cross_depth_reprojection",
            "src_view": src,
            "dst_view": dst,
            **extra,
        }
        row.update(
            reprojection_metrics(
                src_eval,
                dst_eval,
                extrinsics[src],
                extrinsics[dst],
                depth_cache[src],
                depth_cache[dst],
                max_points=max_depth_points,
                rng=rng,
                prefix="cross_",
            )
        )
        rows.append(row)


def prefixed_alignment(extra: dict[str, float]) -> dict[str, float]:
    return {f"vggt_{key}": value for key, value in extra.items()}


def write_report(path: Path, summary: list[dict[str, Any]], detail_csv: Path, images: list[Path]) -> None:
    headers = [
        "rank",
        "candidate",
        "diagnostic_score",
        "cross_valid_target_rate",
        "cross_median_relative_error",
        "cross_agree_1pct_rate",
        "cross_agree_5pct_rate",
        "cross_agree_10pct_rate",
        "self_median_relative_error",
        "vggt_align_center_rmse",
        "vggt_align_rot_median_deg",
        "vggt_align_scale_gt_to_pred",
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
<title>VGGT-Omega on BlendedMVS camera evaluation</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
th {{ background: #eee; }}
code {{ background: #f3f3f3; padding: 1px 3px; }}
</style>
<h1>VGGT-Omega on BlendedMVS Camera Evaluation</h1>
<p><code>gt_as_is_w2c</code> is the BlendedMVS camera/depth positive control. VGGT-Omega cameras are globally aligned to GT camera centers by a single Sim(3) before depth reprojection.</p>
<p><code>vggt_pose_predK</code> uses both predicted pose and predicted intrinsics. <code>vggt_pose_gtK</code> uses predicted pose but GT intrinsics, which helps separate pose drift from intrinsic/preprocessing drift.</p>
<p>Detailed metrics: <code>{html.escape(detail_csv.name)}</code></p>
<table><thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead><tbody>
{''.join(rows_html)}
</tbody></table>
{image_html}
"""
    path.write_text(doc)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model = load_model(Path(args.checkpoint), device)
    all_rows: list[dict[str, Any]] = []
    image_outputs: list[Path] = []

    for scene_root in discover_scenes(Path(args.scene_root))[: args.max_scenes]:
        frames_all, pair_map = load_scene(scene_root)
        view_ids = select_view_ids(frames_all, pair_map, args.max_views, args.srcs_per_ref)
        frames = {view_id: frames_all[view_id] for view_id in view_ids}
        filtered_pair_map = {
            ref: [src for src in pair_map.get(ref, []) if src in frames]
            for ref in frames
        }
        pairs = choose_pairs(frames, filtered_pair_map, args.srcs_per_ref, args.max_pairs_per_scene)
        if len(frames) < 3 or not pairs:
            print(f"{scene_root.name}: skipped frames={len(frames)} pairs={len(pairs)}")
            continue
        print(f"{scene_root.name}: views={len(frames)} pairs={len(pairs)}")

        depth_cache = {
            view_id: read_pfm(frame.depth_path, flip_y=args.pfm_flip_y)
            for view_id, frame in frames.items()
        }
        gt_extrinsics_arr = np.stack([frames[view_id].extrinsic[:3, :4] for view_id in view_ids], axis=0)
        gt_extrinsics = {view_id: frames[view_id].extrinsic for view_id in view_ids}
        gt_intrinsics = {view_id: frames[view_id].intrinsic for view_id in view_ids}

        add_metrics_rows(
            all_rows,
            scene_root.name,
            "gt_as_is_w2c",
            "BlendedMVS GT camera positive control.",
            frames,
            depth_cache,
            pairs,
            gt_extrinsics,
            gt_intrinsics,
            args.max_depth_points,
            rng,
        )

        image_paths = [str(frames[view_id].image_path) for view_id in view_ids]
        pred_extrinsics, pred_intrinsics_input, input_hw = run_vggt(
            model,
            image_paths,
            image_resolution=args.image_resolution,
            device=device,
        )
        pred_intrinsics_orig = scale_intrinsics_to_original(
            pred_intrinsics_input,
            input_hw=input_hw,
            frames=[frames[view_id] for view_id in view_ids],
        )
        pred_aligned, align_diag = align_predicted_to_gt_world(pred_extrinsics, gt_extrinsics_arr)
        print(
            f"  vggt_align scale={align_diag['align_scale_gt_to_pred']:.5g} "
            f"center_rmse={align_diag['align_center_rmse']:.5g} "
            f"rot_med={align_diag['align_rot_median_deg']:.3f}deg"
        )

        pred_extrinsics_dict = {view_id: pred_aligned[idx] for idx, view_id in enumerate(view_ids)}
        pred_intrinsics_dict = {view_id: pred_intrinsics_orig[idx] for idx, view_id in enumerate(view_ids)}
        extra = prefixed_alignment(align_diag)
        add_metrics_rows(
            all_rows,
            scene_root.name,
            "vggt_pose_predK",
            "VGGT-Omega predicted pose and predicted intrinsics, Sim(3)-aligned to GT world.",
            frames,
            depth_cache,
            pairs,
            pred_extrinsics_dict,
            pred_intrinsics_dict,
            args.max_depth_points,
            rng,
            extra=extra,
        )
        add_metrics_rows(
            all_rows,
            scene_root.name,
            "vggt_pose_gtK",
            "VGGT-Omega predicted pose with BlendedMVS GT intrinsics, Sim(3)-aligned to GT world.",
            frames,
            depth_cache,
            pairs,
            pred_extrinsics_dict,
            gt_intrinsics,
            args.max_depth_points,
            rng,
            extra=extra,
        )

        if args.visualize:
            src, dst = pairs[0]
            images = []
            for candidate_name, extrinsics, intrinsics in [
                ("gt_as_is_w2c", gt_extrinsics, gt_intrinsics),
                ("vggt_pose_predK", pred_extrinsics_dict, pred_intrinsics_dict),
                ("vggt_pose_gtK", pred_extrinsics_dict, gt_intrinsics),
            ]:
                src_frame = frames[src]
                dst_frame = frames[dst]
                src_eval = FrameData(
                    src_frame.scene,
                    src_frame.view_id,
                    src_frame.image_path,
                    src_frame.masked_image_path,
                    src_frame.depth_path,
                    src_frame.cam_path,
                    src_frame.extrinsic,
                    intrinsics[src],
                    src_frame.depth_params,
                )
                dst_eval = FrameData(
                    dst_frame.scene,
                    dst_frame.view_id,
                    dst_frame.image_path,
                    dst_frame.masked_image_path,
                    dst_frame.depth_path,
                    dst_frame.cam_path,
                    dst_frame.extrinsic,
                    intrinsics[dst],
                    dst_frame.depth_params,
                )
                images.append(
                    draw_depth_reprojection(
                        src_eval,
                        dst_eval,
                        extrinsics[src],
                        extrinsics[dst],
                        depth_cache[src],
                        depth_cache[dst],
                        f"{scene_root.name} {src}->{dst} | {candidate_name}",
                        max_points=args.max_depth_points,
                        max_width=args.overlay_width,
                        rng=rng,
                    )
                )
            sheet = make_contact_sheet(images, columns=1)
            out_path = output_dir / f"{scene_root.name}_{src:08d}_to_{dst:08d}_vggt_depth_reprojection.jpg"
            sheet.save(out_path, quality=92)
            image_outputs.append(out_path)

        pred_path = output_dir / f"{scene_root.name}_vggt_predictions.npz"
        np.savez(
            pred_path,
            view_ids=np.asarray(view_ids, dtype=np.int32),
            input_hw=np.asarray(input_hw, dtype=np.int32),
            pred_extrinsics=pred_extrinsics,
            pred_extrinsics_aligned=pred_aligned,
            pred_intrinsics_input=pred_intrinsics_input,
            pred_intrinsics_orig=pred_intrinsics_orig,
            gt_extrinsics=gt_extrinsics_arr,
            gt_intrinsics=np.stack([gt_intrinsics[view_id] for view_id in view_ids], axis=0),
            **align_diag,
        )

    summary = aggregate(all_rows)
    detail_csv = output_dir / "per_pair_metrics.csv"
    summary_csv = output_dir / "summary_by_candidate.csv"
    write_csv(detail_csv, all_rows)
    write_csv(summary_csv, summary)
    write_report(output_dir / "index.html", summary, detail_csv, image_outputs)
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
    parser.add_argument("--checkpoint", default="vggt-omega/vggt_omega_1b_512.pt")
    parser.add_argument("--output", default="output/vggt_omega_blendedmvs_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--max-pairs-per-scene", type=int, default=12)
    parser.add_argument("--srcs-per-ref", type=int, default=1)
    parser.add_argument("--max-depth-points", type=int, default=8000)
    parser.add_argument("--overlay-width", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pfm-flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
