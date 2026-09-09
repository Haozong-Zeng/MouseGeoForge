#!/usr/bin/env python3
"""Run the sample VGGT-Omega camera calibration pipeline end to end."""

from __future__ import annotations

import argparse
import os
import queue
import random
import shutil
import string
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from vggt_calibration_pipeline.pipeline_utils import (  # noqa: E402
    extrinsics_to_rvec_tvec,
    intrinsics_for_output_sizes,
    load_model,
    parse_distortions,
    write_anipose_toml,
    evaluate_prediction_files,
    maybe_plot,
    write_csv,
    write_summary,
)
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from vggt_calibration_pipeline.extract_frames import (  # noqa: E402
    VideoInfo,
    common_frame_count,
    discover_videos,
    extract_frame_sets,
    iter_extract_frame_sets,
    parse_frame_spec,
    write_manifest,
)


def frame_dir_for_index(frames_dir: Path, frame_index: int) -> Path:
    return frames_dir / f"frame_{frame_index:06d}"


def image_paths_for_frame(frames_dir: Path, frame_index: int, camera_names: Sequence[str]) -> list[Path]:
    frame_dir = frame_dir_for_index(frames_dir, frame_index)
    return [frame_dir / f"p{name}.png" for name in camera_names]


def run_vggt_calibrations(
    frames_dir: Path,
    calibration_dir: Path,
    frame_indices: Sequence[int],
    camera_names: Sequence[str],
    output_sizes: Sequence[tuple[int, int]],
    checkpoint: Path,
    device_name: str,
    model_resolution: int,
    preprocess_mode: str,
    translation_scale: float,
    distortions: Sequence[float],
    fisheye: bool,
    skip_existing: bool,
    attention_viz_dir: Path | None = None,
) -> list[Path]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu to force CPU execution.")
    if device.type != "cuda":
        print("Warning: VGGT-Omega CPU inference will be slow.", file=sys.stderr)

    calibration_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading VGGT-Omega checkpoint: {checkpoint}", flush=True)
    model = load_model(checkpoint, device)

    written: list[Path] = []
    for count, frame_index in enumerate(frame_indices, start=1):
        output_path = write_vggt_calibration_for_frame(
            model=model,
            device=device,
            frames_dir=frames_dir,
            calibration_dir=calibration_dir,
            frame_index=frame_index,
            camera_names=camera_names,
            output_sizes=output_sizes,
            model_resolution=model_resolution,
            preprocess_mode=preprocess_mode,
            translation_scale=translation_scale,
            distortions=distortions,
            fisheye=fisheye,
            skip_existing=skip_existing,
            attention_viz_dir=attention_viz_dir,
        )
        written.append(output_path)

        if count == 1 or count == len(frame_indices) or count % 25 == 0:
            print(f"Wrote {count}/{len(frame_indices)} calibrations: {output_path}")

    return written


def write_vggt_calibration_for_frame(
    model: torch.nn.Module,
    device: torch.device,
    frames_dir: Path,
    calibration_dir: Path,
    frame_index: int,
    camera_names: Sequence[str],
    output_sizes: Sequence[tuple[int, int]],
    model_resolution: int,
    preprocess_mode: str,
    translation_scale: float,
    distortions: Sequence[float],
    fisheye: bool,
    skip_existing: bool,
    attention_viz_dir: Path | None = None,
) -> Path:
    output_path = calibration_dir / f"frame_{frame_index:06d}" / "calibration.toml"
    if skip_existing and output_path.is_file():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = image_paths_for_frame(frames_dir, frame_index, camera_names)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing extracted image(s): " + ", ".join(missing))

    extrinsics_batch, pose_encoding_batch = predict_cameras_batch(
        model=model,
        device=device,
        image_path_sets=[image_paths],
        model_resolution=model_resolution,
        preprocess_mode=preprocess_mode,
        attention_viz_dir=attention_viz_dir,
        batch_frame_indices=[frame_index],
    )
    extrinsics = extrinsics_batch[0]
    pose_encoding = pose_encoding_batch[0]
    intrinsics = intrinsics_for_output_sizes(pose_encoding, output_sizes)
    rvecs, tvecs = extrinsics_to_rvec_tvec(extrinsics, translation_scale)
    write_anipose_toml(
        output_path=output_path,
        names=camera_names,
        sizes=output_sizes,
        matrices=intrinsics,
        rvecs=rvecs,
        tvecs=tvecs,
        distortions=distortions,
        fisheye=fisheye,
    )
    return output_path


def write_vggt_calibration_batch(
    model: torch.nn.Module,
    device: torch.device,
    frames_dir: Path,
    calibration_dir: Path,
    frame_indices: Sequence[int],
    camera_names: Sequence[str],
    output_sizes: Sequence[tuple[int, int]],
    model_resolution: int,
    preprocess_mode: str,
    translation_scale: float,
    distortions: Sequence[float],
    fisheye: bool,
    skip_existing: bool,
    attention_viz_dir: Path | None = None,
) -> list[Path]:
    output_paths = [calibration_dir / f"frame_{frame_index:06d}" / "calibration.toml" for frame_index in frame_indices]
    pending: list[tuple[int, Path, list[Path]]] = []

    for frame_index, output_path in zip(frame_indices, output_paths):
        if skip_existing and output_path.is_file():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_paths = image_paths_for_frame(frames_dir, frame_index, camera_names)
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing extracted image(s): " + ", ".join(missing))
        pending.append((frame_index, output_path, image_paths))

    if not pending:
        return output_paths

    extrinsics_batch, pose_encoding_batch = predict_cameras_batch(
        model=model,
        device=device,
        image_path_sets=[item[2] for item in pending],
        model_resolution=model_resolution,
        preprocess_mode=preprocess_mode,
        attention_viz_dir=attention_viz_dir,
        batch_frame_indices=[item[0] for item in pending],
    )

    for batch_idx, (_, output_path, _) in enumerate(pending):
        intrinsics = intrinsics_for_output_sizes(pose_encoding_batch[batch_idx], output_sizes)
        rvecs, tvecs = extrinsics_to_rvec_tvec(extrinsics_batch[batch_idx], translation_scale)
        write_anipose_toml(
            output_path=output_path,
            names=camera_names,
            sizes=output_sizes,
            matrices=intrinsics,
            rvecs=rvecs,
            tvecs=tvecs,
            distortions=distortions,
            fisheye=fisheye,
        )

    return output_paths


def predict_cameras_batch(
    model: torch.nn.Module,
    device: torch.device,
    image_path_sets: Sequence[Sequence[Path]],
    model_resolution: int,
    preprocess_mode: str,
    attention_viz_dir: Path | None = None,
    batch_frame_indices: Sequence[int] | None = None,
) -> tuple[object, object]:
    """Predict cameras for a batch of independent camera sets."""
    if not image_path_sets:
        raise ValueError("At least one image set is required")

    image_tensors = [
        load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=preprocess_mode,
            image_resolution=model_resolution,
        )
        for image_paths in image_path_sets
    ]
    images = torch.stack(image_tensors, dim=0).to(device)

    if attention_viz_dir is not None:
        from vggt_calibration_pipeline.attention_viz_weights import patch_attention_module
        if not hasattr(model, "_attn_store"):
            model._attn_store = {}
            patch_attention_module(model.aggregator.inter_frame_blocks[-1].attn, "inter_frame_block_last", model._attn_store)
            for i in range(len(model.aggregator.frame_blocks) - 6, len(model.aggregator.frame_blocks)):
                patch_attention_module(model.aggregator.frame_blocks[i].attn, f"frame_block_{i}", model._attn_store)
        model._attn_store.clear()

    with torch.inference_mode():
        try:
            predictions = model(images)
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory during VGGT batch inference. "
                    "Reduce --batch-size, or use a smaller --model-resolution for scouting runs."
                ) from exc
            raise

    if attention_viz_dir is not None and batch_frame_indices is not None:
        from vggt_calibration_pipeline.attention_viz_weights import generate_and_save_visualizations
        camera_names = [p.stem[1:] for p in image_path_sets[0]] # e.g. pA -> A
        
        csv_path = attention_viz_dir.parent / "attention_scores.csv"
        if not csv_path.exists():
            with open(csv_path, "w") as f:
                f.write("frame,attention_sharpness_score\n")
                
        for batch_idx, frame_index in enumerate(batch_frame_indices):
            frame_dir = attention_viz_dir / f"frame_{frame_index:06d}"
            score = generate_and_save_visualizations(
                image_path_sets[batch_idx], 
                images, 
                model._attn_store, 
                camera_names, 
                frame_dir, 
                batch_idx=batch_idx
            )
            with open(csv_path, "a") as f:
                f.write(f"{frame_index},{score:.6f}\n")

    extrinsics, _ = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
        build_intrinsics=False,
    )

    return (
        extrinsics.detach().float().cpu().numpy(),
        predictions["pose_enc"].detach().float().cpu().numpy(),
    )


def iter_prefetched_frame_sets(
    videos: Sequence[VideoInfo],
    frame_indices: Sequence[int],
    frames_dir: Path,
    max_prefetch: int,
) -> Iterable[tuple[int, list[dict[str, object]]]]:
    if max_prefetch <= 0:
        yield from iter_extract_frame_sets(videos, frame_indices, frames_dir)
        return

    sentinel = object()
    work_queue: queue.Queue[object] = queue.Queue(maxsize=max_prefetch)

    def producer() -> None:
        try:
            for item in iter_extract_frame_sets(videos, frame_indices, frames_dir):
                work_queue.put(item)
        except BaseException as exc:  # Propagate producer failures to the consumer thread.
            work_queue.put(exc)
        finally:
            work_queue.put(sentinel)

    thread = threading.Thread(target=producer, name="vggt-frame-prefetch", daemon=True)
    thread.start()

    try:
        while True:
            item = work_queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        thread.join(timeout=1.0)


def run_streaming_vggt_calibrations(
    videos: Sequence[VideoInfo],
    frames_dir: Path,
    calibration_dir: Path,
    frame_indices: Sequence[int],
    camera_names: Sequence[str],
    output_sizes: Sequence[tuple[int, int]],
    checkpoint: Path,
    device_name: str,
    model_resolution: int,
    preprocess_mode: str,
    translation_scale: float,
    distortions: Sequence[float],
    fisheye: bool,
    skip_existing: bool,
    batch_size: int,
    prefetch: int,
    attention_viz_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Extract each frame set and immediately run VGGT to avoid long GPU-idle phases."""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu to force CPU execution.")
    if device.type != "cuda":
        print("Warning: VGGT-Omega CPU inference will be slow.", file=sys.stderr)

    calibration_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading VGGT-Omega checkpoint: {checkpoint}", flush=True)
    model = load_model(checkpoint, device)

    all_records: list[dict[str, object]] = []
    total = len(frame_indices)
    start = time.perf_counter()
    batch_size = max(1, int(batch_size))
    pending: list[tuple[int, list[dict[str, object]]]] = []

    def flush_batch(batch: Sequence[tuple[int, list[dict[str, object]]]], count_done: int) -> None:
        if not batch:
            return
        batch_frame_indices = [item[0] for item in batch]
        batch_start = time.perf_counter()
        first, last = batch_frame_indices[0], batch_frame_indices[-1]
        print(
            f"[{count_done - len(batch) + 1}-{count_done}/{total}] "
            f"frames {first}..{last}: extracted, running VGGT batch={len(batch)}...",
            flush=True,
        )

        output_paths = write_vggt_calibration_batch(
            model=model,
            device=device,
            frames_dir=frames_dir,
            calibration_dir=calibration_dir,
            frame_indices=batch_frame_indices,
            camera_names=camera_names,
            output_sizes=output_sizes,
            model_resolution=model_resolution,
            preprocess_mode=preprocess_mode,
            translation_scale=translation_scale,
            distortions=distortions,
            fisheye=fisheye,
            skip_existing=skip_existing,
            attention_viz_dir=attention_viz_dir,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - batch_start
        total_elapsed = time.perf_counter() - start
        print(
            f"[{count_done}/{total}] wrote {len(output_paths)} calibration(s) "
            f"in {elapsed:.1f}s; elapsed {total_elapsed:.1f}s",
            flush=True,
        )

    for count, item in enumerate(
        iter_prefetched_frame_sets(videos, frame_indices, frames_dir, prefetch),
        start=1,
    ):
        _, frame_records = item
        all_records.extend(frame_records)
        pending.append(item)
        if len(pending) >= batch_size:
            flush_batch(pending, count)
            pending = []

    if pending:
        flush_batch(pending, total)

    return all_records


def source_model_resolution(videos: Sequence[VideoInfo]) -> int:
    """Use the largest source image dimension for no-resize max_size preprocessing."""
    return max(max(video.width, video.height) for video in videos)


def describe_size_plan(
    videos: Sequence[VideoInfo],
    preprocess_mode: str,
    model_resolution: int,
    patch_size: int = 16,
) -> None:
    source_sizes = ", ".join(f"{video.camera}={video.width}x{video.height}" for video in videos)
    print(f"Source/intrinsics sizes: {source_sizes}", flush=True)
    print(f"VGGT preprocessing: mode={preprocess_mode}, model_resolution={model_resolution}", flush=True)
    keeps_source_size = (
        preprocess_mode == "max_size"
        and all(max(video.width, video.height) == model_resolution for video in videos)
        and all(video.width % patch_size == 0 and video.height % patch_size == 0 for video in videos)
    )
    if keeps_source_size:
        print("VGGT input keeps the source frame size for cameras whose dimensions are patch-compatible.", flush=True)
    else:
        print("VGGT input will be resized for inference; intrinsics are still decoded at the source sizes above.", flush=True)


def _resolve_output_dir(output_root: Path, folder_name: str, dataset: Path) -> Path:
    """Return the output directory for this run.

    Reuses an existing folder only if its source.txt points to the same dataset.
    Otherwise appends a short random suffix so nothing is ever overwritten.
    """
    candidate = output_root / folder_name
    if candidate.exists():
        src_file = candidate / "source.txt"
        if src_file.exists():
            for line in src_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("dataset:"):
                    existing = line.split(":", 1)[1].strip()
                    if Path(existing) == dataset.resolve():
                        return candidate  # same source → resume
        # Folder exists but belongs to a different dataset → new suffixed folder
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        new_dir = output_root / f"{folder_name}_{suffix}"
        print(f"Output folder '{candidate}' belongs to a different dataset. Using '{new_dir}' instead.")
        return new_dir
    return candidate


def _write_source_txt(output_dir: Path, dataset: Path, video_dir: Path) -> None:
    src_file = output_dir / "source.txt"
    src_file.write_text(
        f"dataset:   {dataset.resolve()}\n"
        f"video_dir: {video_dir.resolve()}\n",
        encoding="utf-8",
    )



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("temp_data/videos"))
    parser.add_argument("--video-dir", type=Path, help="Defaults to DATASET/videos-raw.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for outputs. A subfolder named after the dataset is created inside, "
             "e.g. --output-root /media/drive/vggt --dataset temp_data/videos "
             "→ writes to /media/drive/vggt/videos/",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override the output subfolder name (default: dataset folder name). "
             "Use this when two datasets share the same folder name, "
             "e.g. --output-name videos_proj2 → writes to OUTPUT_ROOT/videos_proj2/",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("VGGT_OMEGA_CHECKPOINT", "checkpoints/vggt_omega_1b_512.pt")))
    parser.add_argument("--camera-names", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--frames",
        default="0",
        help="Frame list/range, e.g. 0,10,20 or 0:405:5 or all. Default: 0.",
    )
    parser.add_argument(
        "--model-resolution",
        type=int,
        default=512,
        help="VGGT input resolution. Default: 512.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("balanced", "max_size"),
        default="balanced",
        help="Default balanced resizes to --model-resolution before inference.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--distortions", type=parse_distortions)
    parser.add_argument("--fisheye", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Independent frame sets per VGGT forward pass. Increase cautiously; source-sized images use much more VRAM.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Number of extracted frame sets to queue while VGGT is running. Use 0 to disable.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-attention-viz", action=argparse.BooleanOptionalAction, default=False, help="Save attention visualization PNGs for each frame (default: False; shown to have near-zero correlation with calibration quality)")
    parser.add_argument(
        "--quality-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After calibration, compute gradient-based frame quality scores "
            "(requires an extra backward pass per frame; ~2× slower). "
            "Saves quality_scores.csv in the output directory."
        ),
    )
    parser.add_argument(
        "--save-intermediates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save model intermediates for every processed frame to "
            "<output-dir>/intermediates/frame_XXXXXX/. "
            "Compact set (~3 MB/frame): camera token embeddings at 5 key blocks, "
            "patch importance heatmap, GradCAM PNGs, and pose outputs. "
            "Disables batching (one frame set at a time). "
            "Use --intermediates-full to also save all-token activations (~130 MB/frame extra)."
        ),
    )
    parser.add_argument(
        "--intermediates-full",
        action="store_true",
        default=False,
        help="With --save-intermediates: also save full [F, N, D] token activations at key blocks.",
    )
    parser.add_argument(
        "--intermediates-no-grads",
        action="store_true",
        default=False,
        help="With --save-intermediates: skip the backward pass (no patch importance / GradCAM).",
    )
    parser.add_argument(
        "--intermediates-skip-done",
        action="store_true",
        default=False,
        help="With --save-intermediates: skip frames whose output dir already has patch_importance.npy.",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--ground-truth", "--gt", type=Path, help="Defaults to DATASET/calibration/calibration.toml.")
    parser.add_argument("--threshold-deg", type=float, default=5.0)
    parser.add_argument("--origin", default="ALL")
    parser.add_argument("--include-origin", action="store_true")
    parser.add_argument("--no-scale-align", dest="scale_align", action="store_false")
    parser.set_defaults(scale_align=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    folder_name = args.output_name or args.dataset.name
    output_dir = _resolve_output_dir(args.output_root, folder_name, args.dataset)
    comparison_dir = output_dir / "comparisons"
    gt_path = args.ground_truth or args.dataset / "calibration" / "calibration.toml"

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_source_txt(output_dir, args.dataset, args.video_dir or args.dataset / "videos-raw")

    if args.compare_only:
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground-truth calibration not found: {gt_path}")
        rows = evaluate_prediction_files(
            ground_truth_path=gt_path,
            predictions_path=output_dir,
            threshold_deg=args.threshold_deg,
            origin_spec=args.origin,
            include_origin=args.include_origin,
            scale_align=args.scale_align,
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        write_csv(rows, comparison_dir / "calibration_comparison_results.csv")
        write_summary(rows, comparison_dir / "calibration_comparison_summary.json")
        maybe_plot(rows, comparison_dir / "calibration_metrics.png")
        print(f"Wrote comparison results to {comparison_dir}")
        return 0

    video_dir = args.video_dir or args.dataset / "videos-raw"
    videos = discover_videos(video_dir, args.camera_names)
    frame_count = common_frame_count(videos)
    frame_indices = parse_frame_spec(args.frames, frame_count)
    output_sizes = [(video.width, video.height) for video in videos]
    describe_size_plan(videos, args.preprocess_mode, args.model_resolution)

    if args.distortions is None:
        args.distortions = [0.0, 0.0, 0.0, 0.0] if args.fisheye else [0.0, 0.0, 0.0, 0.0, 0.0]
    if args.fisheye and len(args.distortions) != 4:
        raise ValueError("--fisheye requires exactly four distortion coefficients")

    attention_viz_dir = output_dir / "attention_viz" if args.save_attention_viz else None

    with tempfile.TemporaryDirectory(prefix="vggt_frames_") as _tmp_frames:
        frames_dir = Path(_tmp_frames)

        if args.extract_only:
            extract_frame_sets(videos, frame_indices, frames_dir)
            print(f"Extracted {len(frame_indices)} frame set(s) (temp dir, not saved)")
            return 0

        run_streaming_vggt_calibrations(
            videos=videos,
            frames_dir=frames_dir,
            calibration_dir=output_dir,
            frame_indices=frame_indices,
            camera_names=args.camera_names,
            output_sizes=output_sizes,
            checkpoint=args.checkpoint,
            device_name=args.device,
            model_resolution=args.model_resolution,
            preprocess_mode=args.preprocess_mode,
            translation_scale=args.translation_scale,
            distortions=args.distortions,
            fisheye=args.fisheye,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            prefetch=args.prefetch,
            attention_viz_dir=attention_viz_dir,
        )

        if args.save_intermediates:
            import gc
            from vggt_calibration_pipeline.model_inspector import save_frame_intermediates
            device = torch.device(args.device)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"\nSaving model intermediates → {output_dir}/frame_XXXXXX/")
            print(f"  save_gradients={not args.intermediates_no_grads}  full_activations={args.intermediates_full}")
            _insp_model = load_model(args.checkpoint, device)
            _insp_model.eval()
            total = len(frame_indices)
            for count, frame_index in enumerate(frame_indices, start=1):
                frame_dir = frames_dir / f"frame_{frame_index:06d}"
                image_paths = [str(frame_dir / f"p{name}.png") for name in args.camera_names]
                if not all(Path(p).is_file() for p in image_paths):
                    continue
                out = output_dir / f"frame_{frame_index:06d}"
                if args.intermediates_skip_done and (out / "patch_importance.npy").exists():
                    continue
                images_tensor = load_and_preprocess_images(
                    image_paths, mode=args.preprocess_mode, image_resolution=args.model_resolution
                ).to(device)
                save_frame_intermediates(
                    model=_insp_model,
                    images_tensor=images_tensor,
                    device=device,
                    output_dir=out,
                    save_gradients=not args.intermediates_no_grads,
                    save_full_activations=args.intermediates_full,
                )
                if count == 1 or count == total or count % 25 == 0:
                    print(f"  [{count}/{total}] saved {out}", flush=True)
            del _insp_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Done. Intermediates saved alongside calibrations in {output_dir}/frame_XXXXXX/")

        if args.quality_scores:
            from vggt_calibration_pipeline.frame_quality import score_frame_directory, write_quality_csv
            print("\nComputing gradient-based frame quality scores (backward pass per frame)...")
            _model_q = load_model(args.checkpoint, torch.device(args.device))
            _model_q.eval()
            quality_records = score_frame_directory(
                frames_dir=frames_dir,
                frame_indices=frame_indices,
                camera_names=args.camera_names,
                model=_model_q,
                device=torch.device(args.device),
                model_resolution=args.model_resolution,
                preprocess_mode=args.preprocess_mode,
            )
            write_quality_csv(quality_records, comparison_dir / "quality_scores.csv", args.camera_names)
            print(f"Wrote {len(quality_records)} quality scores → {comparison_dir / 'quality_scores.csv'}")
            del _model_q
            torch.cuda.empty_cache()

    # Frames temp dir has been deleted here.

    if gt_path.is_file():
        rows = evaluate_prediction_files(
            ground_truth_path=gt_path,
            predictions_path=output_dir,
            threshold_deg=args.threshold_deg,
            origin_spec=args.origin,
            include_origin=args.include_origin,
            scale_align=args.scale_align,
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        write_csv(rows, comparison_dir / "calibration_comparison_results.csv")
        write_summary(rows, comparison_dir / "calibration_comparison_summary.json")
        maybe_plot(rows, comparison_dir / "calibration_metrics.png")
        print(f"Wrote comparison results to {comparison_dir}")
    else:
        print(f"Skipped comparison — ground-truth calibration not found: {gt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
