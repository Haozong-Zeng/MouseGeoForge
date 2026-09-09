"""Windowed loader for compact MouseGeoForge Core23 temporal sequences.

Camera views and time remain separate axes. Static camera parameters are
returned once per clip rather than repeated for every timestep.
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_TEMPORAL_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "behavior_llac": (np.dtype(np.uint8), ()),
    "behavior_hlac": (np.dtype(np.uint8), ()),
    "behavior_lljc": (np.dtype(np.uint8), ()),
    "behavior_hljc": (np.dtype(np.uint8), ()),
    "embedding_action": (np.dtype(np.float32), (2,)),
    "embedding_joint": (np.dtype(np.float32), (2,)),
}


class Core23TemporalSequenceDataset(Dataset):
    """Load fixed-length pose windows from one or more sequence manifests.

    Pose-only windows work for both SBeA and s-DANNCE. Set ``require_rgb`` to
    restrict SBeA windows to frames with aligned AVI data, and ``load_images``
    to decode those views into ``[T,V,3,H,W]`` tensors.
    """

    def __init__(
        self,
        manifests: str | Path | Iterable[str | Path],
        *,
        window_size: int = 16,
        frame_step: int = 1,
        window_stride: int = 1,
        split: str | None = None,
        require_rgb: bool = False,
        load_images: bool = False,
        image_size: tuple[int, int] = (512, 288),
        max_instances: int | None = None,
        max_temporal_speed_mm_s: float | None = 5000.0,
        project_root: str | Path = PROJECT_ROOT,
        cache_size: int = 2,
    ) -> None:
        if window_size <= 0 or frame_step <= 0 or window_stride <= 0:
            raise ValueError("window_size, frame_step, and window_stride must be positive")
        if load_images and not require_rgb:
            raise ValueError("load_images=True requires require_rgb=True")
        if image_size[0] % 16 or image_size[1] % 16:
            raise ValueError("image_size must be divisible by VGGT patch size 16")
        self.project_root = Path(project_root)
        self.window_size = int(window_size)
        self.frame_step = int(frame_step)
        self.window_stride = int(window_stride)
        self.require_rgb = bool(require_rgb)
        self.load_images = bool(load_images)
        if max_temporal_speed_mm_s is not None and max_temporal_speed_mm_s <= 0:
            raise ValueError("max_temporal_speed_mm_s must be positive or None")
        self.max_temporal_speed_mm_s = max_temporal_speed_mm_s
        self.image_width, self.image_height = map(int, image_size)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        records: list[dict[str, Any]] = []
        for manifest in manifests:
            path = Path(manifest)
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if split is not None and record.get("split") != split:
                        continue
                    if require_rgb and not record.get("rgb_available", False):
                        continue
                    records.append(record)
        if not records:
            raise ValueError("No temporal sequences match the requested manifests/split")
        self.records = records
        inferred_max = max(int(record["instance_count"]) for record in records)
        self.max_instances = inferred_max if max_instances is None else int(max_instances)
        if self.max_instances < inferred_max:
            raise ValueError(
                f"max_instances={self.max_instances} is smaller than dataset maximum {inferred_max}"
            )

        self.window_counts: list[int] = []
        self.cumulative: list[int] = []
        total = 0
        span = (self.window_size - 1) * self.frame_step + 1
        for record in records:
            available = int(
                record["rgb_frame_count"] if require_rgb else record["frame_count"]
            )
            count = 0 if available < span else (available - span) // self.window_stride + 1
            self.window_counts.append(count)
            total += count
            self.cumulative.append(total)
        if total == 0:
            raise ValueError("No sequence is long enough for the requested temporal window")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def _load_sequence(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        path = self._resolve(record["sequence_npz"])
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        self._cache[path] = arrays
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[sequence_index - 1] if sequence_index else 0
        return sequence_index, index - previous

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, local_window = self._locate(index)
        record = self.records[sequence_index]
        arrays = self._load_sequence(record)
        start = local_window * self.window_stride
        local_indices = start + np.arange(self.window_size) * self.frame_step
        if self.require_rgb:
            pose_indices = arrays["rgb_pose_indices"][local_indices].astype(np.int64)
        else:
            pose_indices = local_indices.astype(np.int64)

        actual_instances = int(record["instance_count"])
        keypoints = np.zeros(
            (self.window_size, self.max_instances, 23, 3), dtype=np.float32
        )
        world_valid = np.zeros(
            (self.window_size, self.max_instances, 23), dtype=bool
        )
        keypoints[:, :actual_instances] = arrays["joint_world_xyz"][pose_indices]
        world_valid[:, :actual_instances] = arrays["world_valid"][pose_indices]
        instance_ids = np.full(self.max_instances, -1, dtype=np.int64)
        instance_ids[:actual_instances] = arrays["instance_ids"]
        instance_valid = np.zeros((self.window_size, self.max_instances), dtype=bool)
        instance_valid[:, :actual_instances] = True
        timestamps = arrays["timestamps_seconds"][pose_indices].astype(np.float64)
        delta_seconds = np.zeros(self.window_size, dtype=np.float32)
        delta_seconds[1:] = np.diff(timestamps).astype(np.float32)
        transition_valid = np.zeros_like(world_valid)
        if self.window_size > 1:
            paired_valid = world_valid[1:] & world_valid[:-1]
            if self.max_temporal_speed_mm_s is not None:
                displacement = np.linalg.norm(
                    np.diff(keypoints, axis=0), axis=-1
                )
                speed = displacement / delta_seconds[1:, None, None]
                paired_valid &= speed <= self.max_temporal_speed_mm_s
            transition_valid[1:] = paired_valid

        sample: dict[str, Any] = {
            "sample_id": f"{record['sequence_id']}:{int(pose_indices[0])}",
            "sequence_id": record["sequence_id"],
            "source": record["source"],
            "split": record.get("split"),
            "frame_indices": torch.from_numpy(arrays["frame_indices"][pose_indices].copy()),
            "timestamps_seconds": torch.from_numpy(timestamps.astype(np.float32)),
            "delta_seconds": torch.from_numpy(delta_seconds),
            "fps": torch.tensor(float(record["fps"]), dtype=torch.float32),
            "instance_ids": torch.from_numpy(instance_ids),
            "instance_valid": torch.from_numpy(instance_valid),
            "keypoint_world_xyz": torch.from_numpy(keypoints),
            "world_valid": torch.from_numpy(world_valid),
            "temporal_transition_valid": torch.from_numpy(transition_valid),
            "fixed_camera": torch.tensor(bool(record.get("fixed_camera", False))),
            "camera_available": torch.tensor(bool(record.get("camera_available", False))),
            "rgb_available": torch.tensor(bool(record.get("rgb_available", False))),
            "pose_to_camera_alignment_valid": torch.tensor(
                bool(record.get("pose_to_camera_alignment_valid", False))
            ),
        }
        for key, (dtype, trailing_shape) in OPTIONAL_TEMPORAL_FIELDS.items():
            padded = np.zeros(
                (self.window_size, self.max_instances, *trailing_shape), dtype=dtype
            )
            value_valid = np.zeros(
                (self.window_size, self.max_instances), dtype=bool
            )
            if (
                key in arrays
                and arrays[key].ndim >= 2
                and arrays[key].shape[0] > int(pose_indices[-1])
            ):
                values = arrays[key][pose_indices]
                expected = (self.window_size, actual_instances, *trailing_shape)
                if values.shape != expected:
                    raise ValueError(
                        f"Unexpected {key} shape for {record['sequence_id']}: "
                        f"{values.shape}, expected {expected}"
                    )
                padded[:, :actual_instances] = values
                value_valid[:, :actual_instances] = True
            sample[key] = torch.from_numpy(padded)
            sample[f"{key}_valid"] = torch.from_numpy(value_valid)

        if self.require_rgb:
            sample.update(self._rgb_labels(record, arrays, local_indices, actual_instances))
        return sample

    def _rgb_labels(
        self,
        record: dict[str, Any],
        arrays: dict[str, np.ndarray],
        rgb_indices: np.ndarray,
        actual_instances: int,
    ) -> dict[str, Any]:
        view_count = int(record["camera_count"])
        boxes = np.zeros(
            (self.window_size, view_count, self.max_instances, 4), dtype=np.float32
        )
        box_valid = np.zeros(
            (self.window_size, view_count, self.max_instances), dtype=bool
        )
        boxes[:, :, :actual_instances] = arrays["boxes_cxcywh"][rgb_indices]
        box_valid[:, :, :actual_instances] = arrays["box_valid"][rgb_indices]
        result: dict[str, Any] = {
            "boxes_cxcywh": torch.from_numpy(boxes),
            "box_valid": torch.from_numpy(box_valid),
            "projection_matrices": torch.from_numpy(
                arrays["projection_matrices"].astype(np.float32)
            ),
            "intrinsics": torch.from_numpy(arrays["intrinsics"].astype(np.float32)),
            "extrinsics": torch.from_numpy(arrays["extrinsics"].astype(np.float32)),
        }
        if self.load_images:
            video_indices = arrays["video_frame_indices"][rgb_indices].astype(np.int64)
            result["images"] = self._decode_images(record, video_indices)
        return result

    def _decode_images(
        self, record: dict[str, Any], video_indices: np.ndarray
    ) -> torch.Tensor:
        views = []
        for video in record["video_paths"]:
            path = self._resolve(video)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {path}")
            frames = []
            try:
                for frame_index in video_indices:
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
                        raise RuntimeError(f"Could not seek {path} to {frame_index}")
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Could not decode {path} frame {frame_index}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame,
                        (self.image_width, self.image_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    frames.append(frame)
            finally:
                capture.release()
            views.append(np.stack(frames))
        images = np.stack(views, axis=1).astype(np.float32) / 255.0
        return torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()


def inspect_temporal_sample(
    manifests: str | Path | Iterable[str | Path], **kwargs: Any
) -> dict[str, Any]:
    dataset = Core23TemporalSequenceDataset(manifests, **kwargs)
    sample = dataset[0]
    return {
        key: tuple(value.shape) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
