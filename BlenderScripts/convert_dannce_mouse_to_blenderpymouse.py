#!/usr/bin/env python3
"""Convert the official DANNCE markerless-mouse demos to BlenderPyMouse.

The converter uses the full ``label3d_dannce.mat`` annotations, decodes only
the labeled source frames, removes the supplied Brown-Conrady lens distortion
from every image, and reprojects the 3D labels with the matching pinhole camera.
No depth or visibility labels are invented: their availability masks remain
false so downstream losses can distinguish real labels from missing labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat


SOURCE_MOUSE22 = (
    "EarL",
    "EarR",
    "Snout",
    "SpineF",
    "SpineM",
    "Tail(base)",
    "Tail(mid)",
    "Tail(end)",
    "ForepawL",
    "WristL",
    "ElbowL",
    "ShoulderL",
    "ForepawR",
    "WristR",
    "ElbowR",
    "ShoulderR",
    "HindpawL",
    "AnkleL",
    "KneeL",
    "HindpawR",
    "AnkleR",
    "KneeR",
)

CORE23 = (
    "nose",
    "head_center",
    "left_ear",
    "right_ear",
    "neck",
    "spine_center",
    "lumbar_spine",
    "tail_base",
    "tail_q1",
    "tail_q2",
    "tail_q3",
    "tail_q4",
    "tail_q5",
    "tail_tip",
    "left_forepaw_center",
    "right_forepaw_center",
    "left_hindpaw_center",
    "right_hindpaw_center",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "abdomen_center",
)

SOURCE_TO_CORE23 = {
    "Snout": "nose",
    "EarL": "left_ear",
    "EarR": "right_ear",
    "SpineF": "neck",
    "SpineM": "spine_center",
    "Tail(base)": "tail_base",
    "Tail(mid)": "tail_q3",
    "Tail(end)": "tail_tip",
    "ForepawL": "left_forepaw_center",
    "ForepawR": "right_forepaw_center",
    "HindpawL": "left_hindpaw_center",
    "HindpawR": "right_hindpaw_center",
    "ElbowL": "left_elbow",
    "ElbowR": "right_elbow",
    "KneeL": "left_knee",
    "KneeR": "right_knee",
}

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("dataset/DANNCE/raw"),
        help="Directory containing markerless_mouse_1 and markerless_mouse_2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset"),
        help=(
            "Destination parent. Real DANNCE data intentionally lives beside, "
            "not inside, the synthetic BlenderPyMouse collection."
        ),
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=("markerless_mouse_1", "markerless_mouse_2"),
    )
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Scale undistorted pinhole focal lengths while preserving resolution.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def _mat_items(value: Any) -> list[Any]:
    return list(np.asarray(value, dtype=object).reshape(-1))


def load_session(mat_path: Path) -> dict[str, Any]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    cameras = [str(value) for value in np.asarray(mat["camnames"]).reshape(-1)]
    labels = _mat_items(mat["labelData"])
    parameters = _mat_items(mat["params"])
    if len(cameras) != len(labels) or len(cameras) != len(parameters):
        raise ValueError(f"Camera-array length mismatch in {mat_path}")

    source_frames = np.asarray(labels[0].data_frame, dtype=np.int64).reshape(-1)
    sample_ids = np.asarray(labels[0].data_sampleID, dtype=np.int64).reshape(-1)
    source_xyz = np.asarray(labels[0].data_3d, dtype=np.float64).reshape(-1, 22, 3)
    for camera_index, label in enumerate(labels[1:], start=1):
        if not np.array_equal(source_frames, np.asarray(label.data_frame).reshape(-1)):
            raise ValueError(f"Unsynchronized label frames for camera {camera_index}")
        candidate = np.asarray(label.data_3d, dtype=np.float64).reshape(-1, 22, 3)
        if not np.allclose(source_xyz, candidate, equal_nan=True):
            raise ValueError(f"Inconsistent 3D labels for camera {camera_index}")

    camera_models = []
    for name, parameter in zip(cameras, parameters, strict=True):
        # DANNCE stores row-vector matrices. BlenderPyMouse/OpenCV use columns.
        intrinsic_distorted = np.asarray(parameter.K, dtype=np.float64).T
        rotation = np.asarray(parameter.r, dtype=np.float64).T
        translation = np.asarray(parameter.t, dtype=np.float64).reshape(3)
        radial = np.asarray(parameter.RDistort, dtype=np.float64).reshape(3)
        tangential = np.asarray(parameter.TDistort, dtype=np.float64).reshape(2)
        camera_models.append(
            {
                "name": name,
                "intrinsic_distorted": intrinsic_distorted,
                "rotation": rotation,
                "translation": translation,
                "radial": radial,
                "tangential": tangential,
            }
        )
    return {
        "camera_names": cameras,
        "camera_models": camera_models,
        "source_frames": source_frames,
        "sample_ids": sample_ids,
        "source_xyz": source_xyz,
    }


def find_videos(session_root: Path, camera_names: list[str]) -> dict[str, list[Path]]:
    candidates = sorted(
        path
        for path in session_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(f"No videos found below {session_root}")
    result: dict[str, list[Path]] = {}
    for camera_name in camera_names:
        matches = [
            path
            for path in candidates
            if path.parent.name.lower() == camera_name.lower()
        ]
        try:
            matches.sort(key=lambda path: int(path.stem))
        except ValueError as error:
            raise ValueError(
                f"DANNCE video segment names must be integer starts: {matches}"
            ) from error
        if not matches:
            raise ValueError(
                f"No segmented videos found for {camera_name} below {session_root}"
            )
        result[camera_name] = matches
    return result


def pinhole_intrinsic(distorted: np.ndarray, focal_scale: float) -> np.ndarray:
    if focal_scale <= 0:
        raise ValueError("--focal-scale must be positive")
    result = np.asarray(distorted, dtype=np.float64).copy()
    result[0, 0] *= focal_scale
    result[1, 1] *= focal_scale
    result[0, 1] = 0.0
    result[1, 0] = 0.0
    result[2] = (0.0, 0.0, 1.0)
    return result


def undistort_maps(
    width: int,
    height: int,
    distorted_intrinsic: np.ndarray,
    pinhole: np.ndarray,
    radial: np.ndarray,
    tangential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an exact DANNCE/Brown-Conrady inverse-remap, including K skew."""
    x_pixel, y_pixel = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (x_pixel - pinhole[0, 2]) / pinhole[0, 0]
    y = (y_pixel - pinhole[1, 2]) / pinhole[1, 1]
    radius2 = x * x + y * y
    k1, k2, k3 = radial
    p1, p2 = tangential
    radial_scale = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted_x = (
        x * radial_scale + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    )
    distorted_y = (
        y * radial_scale + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    )
    map_x = (
        distorted_intrinsic[0, 0] * distorted_x
        + distorted_intrinsic[0, 1] * distorted_y
        + distorted_intrinsic[0, 2]
    )
    map_y = (
        distorted_intrinsic[1, 0] * distorted_x
        + distorted_intrinsic[1, 1] * distorted_y
        + distorted_intrinsic[1, 2]
    )
    return map_x.astype(np.float32), map_y.astype(np.float32)


def project_pinhole(
    xyz: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_xyz = (rotation @ xyz.T).T + translation
    projected = (intrinsic @ camera_xyz.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv, camera_xyz


def core23_from_mouse22(source_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(CORE23), 3), dtype=np.float32)
    valid = np.zeros(len(CORE23), dtype=bool)
    source_index = {name: index for index, name in enumerate(SOURCE_MOUSE22)}
    target_index = {name: index for index, name in enumerate(CORE23)}
    for source_name, target_name in SOURCE_TO_CORE23.items():
        point = source_xyz[source_index[source_name]]
        if np.isfinite(point).all():
            output[target_index[target_name]] = point
            valid[target_index[target_name]] = True
    return output, valid


def write_camera(path: Path, extrinsic: np.ndarray, intrinsic: np.ndarray) -> None:
    lines = ["extrinsic"]
    lines.extend(" ".join(f"{value:.12g}" for value in row) for row in extrinsic)
    lines.extend(("", "intrinsic"))
    lines.extend(" ".join(f"{value:.12g}" for value in row) for row in intrinsic)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pair_file(path: Path, view_count: int) -> None:
    lines = [str(view_count)]
    for reference in range(view_count):
        sources = [index for index in range(view_count) if index != reference]
        scores = [1.0 / (rank + 1) for rank in range(len(sources))]
        lines.extend(
            (
                str(reference),
                f"{len(sources)} "
                + " ".join(
                    f"{source} {score:.6f}"
                    for source, score in zip(sources, scores, strict=True)
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def convert_session(
    raw_root: Path,
    output_root: Path,
    session_name: str,
    focal_scale: float,
    jpeg_quality: int,
) -> dict[str, Any]:
    session_root = raw_root / session_name
    mat_path = session_root / "label3d_dannce.mat"
    session = load_session(mat_path)
    videos = find_videos(session_root, session["camera_names"])
    output_name = (
        "DANNCE-MarkerlessMouse1-Core23-RealMultiview"
        if session_name.endswith("_1")
        else "DANNCE-MarkerlessMouse2-Core23-RealMultiview"
    )
    target_root = output_root / output_name
    target_root.mkdir(parents=True, exist_ok=True)

    camera_runtime = []
    for view_id, model in enumerate(session["camera_models"]):
        video_paths = videos[model["name"]]
        video_path = video_paths[0]
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {video_path}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_segment_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        segment_starts = [int(path.stem) for path in video_paths]
        if segment_starts[0] != 0 or segment_starts != sorted(segment_starts):
            raise ValueError(f"Unexpected segment starts for {model['name']}: {segment_starts}")
        if int(session["source_frames"].max()) >= segment_starts[-1] + first_segment_count:
            raise ValueError(
                f"Label frame exceeds {model['name']} segments: "
                f"{session['source_frames'].max()}"
            )
        intrinsic = pinhole_intrinsic(model["intrinsic_distorted"], focal_scale)
        map_x, map_y = undistort_maps(
            width,
            height,
            model["intrinsic_distorted"],
            intrinsic,
            model["radial"],
            model["tangential"],
        )
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = model["rotation"]
        extrinsic[:3, 3] = model["translation"]
        camera_runtime.append(
            {
                **model,
                "view_id": view_id,
                "video_paths": video_paths,
                "segment_starts": segment_starts,
                "width": width,
                "height": height,
                "first_segment_count": first_segment_count,
                "fps": fps,
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "map_x": map_x,
                "map_y": map_y,
            }
        )

    widths = {camera["width"] for camera in camera_runtime}
    heights = {camera["height"] for camera in camera_runtime}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError(f"Mixed camera resolutions are not supported: {widths}x{heights}")
    width, height = widths.pop(), heights.pop()

    scene_records = []
    scene_data = []
    for label_index, (source_frame, sample_id, source_xyz) in enumerate(
        zip(
            session["source_frames"],
            session["sample_ids"],
            session["source_xyz"],
            strict=True,
        ),
        start=1,
    ):
        scene_name = f"frame_{label_index:06d}"
        scene_root = target_root / scene_name
        for directory in ("blended_images", "cams", "annotations"):
            (scene_root / directory).mkdir(parents=True, exist_ok=True)
        core_world, world_valid = core23_from_mouse22(source_xyz)
        source_valid = np.isfinite(source_xyz).all(axis=-1)
        uv = np.zeros((len(camera_runtime), 1, len(CORE23), 2), dtype=np.float32)
        in_frame = np.zeros((len(camera_runtime), 1, len(CORE23)), dtype=bool)
        camera_xyz_all = np.zeros(
            (len(camera_runtime), len(CORE23), 3), dtype=np.float32
        )
        views = []
        for camera in camera_runtime:
            view_id = camera["view_id"]
            projected, camera_xyz = project_pinhole(
                core_world,
                camera["rotation"],
                camera["translation"],
                camera["intrinsic"],
            )
            valid_projection = world_valid & np.isfinite(projected).all(axis=-1)
            valid_projection &= camera_xyz[:, 2] > 1e-6
            inside = valid_projection.copy()
            inside &= projected[:, 0] >= 0
            inside &= projected[:, 0] < width
            inside &= projected[:, 1] >= 0
            inside &= projected[:, 1] < height
            uv[view_id, 0, valid_projection] = projected[valid_projection]
            in_frame[view_id, 0] = inside
            camera_xyz_all[view_id] = camera_xyz
            write_camera(
                scene_root / "cams" / f"{view_id:08d}_cam.txt",
                camera["extrinsic"],
                camera["intrinsic"],
            )
            keypoints = []
            for keypoint_index, name in enumerate(CORE23):
                valid = bool(world_valid[keypoint_index])
                keypoints.append(
                    {
                        "name": name,
                        "joint_world_xyz": core_world[keypoint_index].tolist(),
                        "surface_world_xyz": core_world[keypoint_index].tolist(),
                        "camera_xyz": camera_xyz[keypoint_index].tolist() if valid else [0.0] * 3,
                        "uv": projected[keypoint_index].tolist() if valid else [0.0, 0.0],
                        "x": float(projected[keypoint_index, 0]) if valid else 0.0,
                        "y": float(projected[keypoint_index, 1]) if valid else 0.0,
                        "camera_depth": float(camera_xyz[keypoint_index, 2]) if valid else 0.0,
                        "geometry_valid": valid,
                        "in_frame": bool(inside[keypoint_index]),
                        "visible": False,
                        "visibility_annotation_available": False,
                    }
                )
            annotation = {
                "format": "dannce_mouse22_to_blenderpymouse_core23_v1",
                "session": session_name,
                "label_index": label_index - 1,
                "source_video_frame_zero_based": int(source_frame),
                "source_sample_id": int(sample_id),
                "view_id": view_id,
                "camera": camera["name"],
                "image": f"blended_images/{view_id:08d}.jpg",
                "cam": f"cams/{view_id:08d}_cam.txt",
                "intrinsic": camera["intrinsic"].tolist(),
                "extrinsic": camera["extrinsic"].tolist(),
                "source_distorted_intrinsic": camera["intrinsic_distorted"].tolist(),
                "source_radial_distortion": camera["radial"].tolist(),
                "source_tangential_distortion": camera["tangential"].tolist(),
                "distortion_corrected": True,
                "depth_available": False,
                "visibility_annotation_available": False,
                "keypoints": keypoints,
            }
            (scene_root / "annotations" / f"{view_id:08d}.json").write_text(
                json.dumps(annotation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            views.append(annotation)

        np.savez_compressed(
            scene_root / "keypoints.npz",
            format=np.asarray("dannce_mouse22_to_blenderpymouse_core23_v1"),
            names=np.asarray(CORE23),
            view_ids=np.arange(len(camera_runtime), dtype=np.int32),
            camera_names=np.asarray(session["camera_names"]),
            instance_ids=np.asarray([0], dtype=np.int32),
            instance_names=np.asarray(["mouse"]),
            joint_world_xyz=core_world[None],
            surface_world_xyz=core_world[None],
            world_valid=world_valid[None],
            uv=uv,
            in_frame=in_frame,
            surface_visible=np.zeros_like(in_frame),
            visibility_valid=np.zeros_like(in_frame),
            source_mouse22_names=np.asarray(SOURCE_MOUSE22),
            source_joint_world_xyz=np.nan_to_num(source_xyz, nan=0.0).astype(np.float32),
            source_world_valid=source_valid,
            source_video_frame_zero_based=np.asarray(source_frame, dtype=np.int64),
            source_sample_id=np.asarray(sample_id, dtype=np.int64),
            distortion_corrected=np.asarray(True),
        )
        write_pair_file(scene_root / "cams" / "pair.txt", len(camera_runtime))
        scene_meta = {
            "format": "dannce_mouse22_to_blenderpymouse_core23_v1",
            "session": session_name,
            "scene": scene_name,
            "label_index": label_index - 1,
            "source_video_frame_zero_based": int(source_frame),
            "source_sample_id": int(sample_id),
            "views": views,
            "mapped_core23_count": int(world_valid.sum()),
            "mapped_core23_valid": {
                name: bool(valid) for name, valid in zip(CORE23, world_valid, strict=True)
            },
        }
        (scene_root / "scene_meta.json").write_text(
            json.dumps(scene_meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        scene_records.append(scene_name)
        scene_data.append((scene_root, int(source_frame)))

    # Decode per camera so each compressed stream is opened only once.
    for camera in camera_runtime:
        starts_and_paths = list(
            zip(camera["segment_starts"], camera["video_paths"], strict=True)
        )
        for segment_index, (segment_start, video_path) in enumerate(starts_and_paths):
            segment_end = (
                starts_and_paths[segment_index + 1][0]
                if segment_index + 1 < len(starts_and_paths)
                else 2**63 - 1
            )
            selected = [
                (scene_root, source_frame)
                for scene_root, source_frame in scene_data
                if segment_start <= source_frame < segment_end
            ]
            if not selected:
                continue
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open {video_path}")
                for scene_root, source_frame in selected:
                    local_frame = source_frame - segment_start
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame):
                        raise RuntimeError(
                            f"Could not seek {video_path} to frame {local_frame}"
                        )
                    ok, distorted = capture.read()
                    if not ok or distorted is None:
                        raise RuntimeError(
                            f"Could not decode {video_path} frame {local_frame}"
                        )
                    corrected = cv2.remap(
                        distorted,
                        camera["map_x"],
                        camera["map_y"],
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    image_path = (
                        scene_root
                        / "blended_images"
                        / f"{camera['view_id']:08d}.jpg"
                    )
                    if not cv2.imwrite(
                        str(image_path),
                        corrected,
                        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
                    ):
                        raise RuntimeError(f"Could not write {image_path}")
            finally:
                capture.release()

    metadata = {
        "format": "dannce_mouse22_to_blenderpymouse_core23_v1",
        "data_origin": "real_multiview_video",
        "dataset_name": output_name,
        "source_session": session_name,
        "source_project": "DANNCE",
        "source_repository": "https://github.com/spoonsso/dannce",
        "source_repository_commit": "cefd782aee1fce2044c7cef6237dfc207034fc68",
        "source_annotation": str(mat_path.resolve()),
        "source_annotation_sha256": sha256(mat_path),
        "source_videos": {
            name: [str(path.resolve()) for path in paths]
            for name, paths in videos.items()
        },
        "scene_count": len(scene_records),
        "scene_folders": scene_records,
        "views_per_scene": len(camera_runtime),
        "cameras": session["camera_names"],
        "instance_count": 1,
        "keypoint_schema": "Core23",
        "keypoint_names": list(CORE23),
        "source_keypoint_schema": "DANNCE mouse22",
        "source_keypoint_names": list(SOURCE_MOUSE22),
        "source_to_core23": SOURCE_TO_CORE23,
        "mapped_core23_names": list(SOURCE_TO_CORE23.values()),
        "unmapped_core23_names": [
            name for name in CORE23 if name not in SOURCE_TO_CORE23.values()
        ],
        "render_resolution": [width, height],
        "image_format": "JPEG",
        "lens_distortion_correction": {
            "applied": True,
            "model": "DANNCE Brown-Conrady (3 radial + 2 tangential, full skew K)",
            "focal_scale": focal_scale,
            "output_camera_model": "pinhole, zero skew",
        },
        "depth_available": False,
        "visibility_annotation_available": False,
        "world_coordinate_unit": "millimeter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    reports = []
    for session_name in args.sessions:
        report = convert_session(
            args.raw_root.resolve(),
            args.output_root.resolve(),
            session_name,
            args.focal_scale,
            args.jpeg_quality,
        )
        reports.append(report)
        print(
            f"Converted {report['source_session']}: {report['scene_count']} scenes, "
            f"{report['views_per_scene']} views each -> {report['dataset_name']}"
        )


if __name__ == "__main__":
    main()
