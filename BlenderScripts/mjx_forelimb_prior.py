import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontlimb_quality_check as qc


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "track-mjx" / "data" / "mouse_arm"


def add_python_deps():
    configured = os.environ.get("MOUSEGEOFORGE_BLENDER_PYTHON_DEPS", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path(__file__).resolve().parent / "python_deps")
    for deps in candidates:
        if deps.is_dir():
            sys.path.insert(0, str(deps.resolve()))
            return


def import_h5py():
    add_python_deps()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py
    import numpy as np

    return h5py, np


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Blender forelimbs against a lightweight MJX mouse-arm pose prior."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--quality-json", default=None)
    parser.add_argument("--warning-frames-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nearest-top", type=int, default=5)
    parser.add_argument("--distance-warning-threshold", type=float, default=3.0)
    return parser.parse_args(args_after_dash())


def safe_normalized(vector):
    if vector.length < 1.0e-8:
        return None
    return vector.normalized()


def angle_degrees(a, b):
    if a.length < 1.0e-8 or b.length < 1.0e-8:
        return None
    return math.degrees(a.angle(b))


def chain_features(shoulder, elbow, wrist):
    upper = elbow - shoulder
    lower = wrist - elbow
    reach = wrist - shoulder
    upper_len = upper.length
    lower_len = lower.length
    chain_len = upper_len + lower_len
    if upper_len < 1.0e-8 or lower_len < 1.0e-8 or chain_len < 1.0e-8:
        return None
    elbow_angle = angle_degrees(upper, lower)
    upper_dir = safe_normalized(upper)
    lower_dir = safe_normalized(lower)
    bend_cos = None if upper_dir is None or lower_dir is None else float(upper_dir.dot(lower_dir))
    return {
        "reach_ratio": float(reach.length / chain_len),
        "elbow_angle_deg": None if elbow_angle is None else float(elbow_angle),
        "bend_cos": bend_cos,
        "length_ratio": float(lower_len / upper_len),
        "upper_length": float(upper_len),
        "lower_length": float(lower_len),
        "chain_length": float(chain_len),
    }


def feature_vector(features):
    if features is None or features.get("elbow_angle_deg") is None:
        return None
    return [
        float(features["reach_ratio"]),
        float(features["elbow_angle_deg"]) / 180.0,
        float(features["bend_cos"]),
        float(features["length_ratio"]),
    ]


class MJXForelimbPrior:
    def __init__(self, records, np_module):
        self.np = np_module
        self.records = records
        self.matrix = self.np.array([record["feature_vector"] for record in records], dtype=self.np.float64)
        self.mean = self.matrix.mean(axis=0)
        self.std = self.matrix.std(axis=0)
        self.std[self.std < 1.0e-8] = 1.0
        self.standardized = (self.matrix - self.mean) / self.std
        self.percentiles = {}
        for key in ("reach_ratio", "elbow_angle_deg", "bend_cos", "length_ratio"):
            values = self.np.array([record["features"][key] for record in records], dtype=self.np.float64)
            self.percentiles[key] = {
                "p01": float(self.np.percentile(values, 1)),
                "p05": float(self.np.percentile(values, 5)),
                "p50": float(self.np.percentile(values, 50)),
                "p95": float(self.np.percentile(values, 95)),
                "p99": float(self.np.percentile(values, 99)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    @classmethod
    def from_data_dir(cls, data_dir):
        h5py, np_module = import_h5py()
        data_dir = Path(data_dir)
        records = []
        for h5_path in sorted(data_dir.glob("*.h5")):
            with h5py.File(h5_path, "r", locking=False) as handle:
                if "marker_sites" not in handle:
                    continue
                marker_sites = handle["marker_sites"][:]
                for frame_index, points in enumerate(marker_sites):
                    if points.shape[0] < 3:
                        continue
                    shoulder = Vector([float(v) for v in points[0]])
                    elbow = Vector([float(v) for v in points[1]])
                    wrist = Vector([float(v) for v in points[2]])
                    features = chain_features(shoulder, elbow, wrist)
                    vector = feature_vector(features)
                    if vector is None:
                        continue
                    records.append(
                        {
                            "clip": h5_path.name,
                            "frame_index": int(frame_index),
                            "features": features,
                            "feature_vector": vector,
                        }
                    )
        if not records:
            raise RuntimeError(f"No MJX marker-site records found in {data_dir}")
        return cls(records, np_module)

    def score_features(self, features, top=5):
        vector = feature_vector(features)
        if vector is None:
            return {"available": False}
        vector_np = self.np.array(vector, dtype=self.np.float64)
        standardized = (vector_np - self.mean) / self.std
        distances = self.np.linalg.norm(self.standardized - standardized, axis=1)
        order = self.np.argsort(distances)[: max(int(top), 1)]
        nearest = []
        for index in order:
            record = self.records[int(index)]
            nearest.append(
                {
                    "distance": float(distances[int(index)]),
                    "clip": record["clip"],
                    "frame_index": record["frame_index"],
                    "features": record["features"],
                }
            )
        z = {
            "reach_ratio": float(standardized[0]),
            "elbow_angle": float(standardized[1]),
            "bend_cos": float(standardized[2]),
            "length_ratio": float(standardized[3]),
        }
        return {
            "available": True,
            "distance": float(distances[int(order[0])]),
            "z": z,
            "nearest": nearest,
        }

    def summary(self):
        return {
            "record_count": len(self.records),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "percentiles": self.percentiles,
        }


def blender_forelimb_features(armature, side):
    shoulder = qc.pose_bone_head_world(armature, f"Scapula_{side}")
    elbow = qc.pose_bone_tail_world(armature, f"Scapula_{side}")
    wrist = qc.pose_bone_tail_world(armature, f"Wrist_{side}")
    if shoulder is None or elbow is None or wrist is None:
        return None
    return chain_features(shoulder, elbow, wrist)


def frame_list(action, explicit_frames, sample_step, quality_json, warning_only):
    if explicit_frames:
        return sorted({float(frame) for frame in explicit_frames})
    if quality_json and warning_only:
        quality = json.loads(Path(quality_json).read_text(encoding="utf-8"))
        action_record = next(row for row in quality["actions"] if row["action"] == action.name)
        return sorted({float(row["frame"]) for row in action_record["warning_frames"]})
    frames = qc.action_frames(action)
    step = max(int(sample_step), 1)
    return frames[::step]


def main():
    args = parse_args()
    prior = MJXForelimbPrior.from_data_dir(args.data_dir)
    armature = qc.find_armature(args.armature)
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Action {args.action!r} not found.")
    qc.set_action(armature, action)
    frames = frame_list(
        action,
        args.frame,
        args.sample_step,
        args.quality_json,
        args.warning_frames_only,
    )

    rows = []
    for frame in frames:
        qc.set_frame(frame)
        for side in ("L", "R"):
            features = blender_forelimb_features(armature, side)
            score = prior.score_features(features, args.nearest_top)
            flags = []
            if not score.get("available"):
                flags.append(f"{side}_mjx_prior_unavailable")
            elif score["distance"] > args.distance_warning_threshold:
                flags.append(f"{side}_mjx_prior_distance")
            rows.append(
                {
                    "frame": float(frame),
                    "side": side,
                    "features": features,
                    "score": score,
                    "flags": flags,
                }
            )

    warning_rows = [row for row in rows if row["flags"]]
    worst = sorted(
        [row for row in rows if row["score"].get("available")],
        key=lambda row: row["score"]["distance"],
        reverse=True,
    )[:50]
    report = {
        "file": bpy.data.filepath,
        "action": action.name,
        "armature": armature.name,
        "data_dir": str(Path(args.data_dir).resolve()),
        "prior": prior.summary(),
        "settings": {
            "sample_step": args.sample_step,
            "nearest_top": args.nearest_top,
            "distance_warning_threshold": args.distance_warning_threshold,
            "quality_json": args.quality_json,
            "warning_frames_only": args.warning_frames_only,
        },
        "frame_count": len(frames),
        "row_count": len(rows),
        "warning_count": len(warning_rows),
        "warning_rows": warning_rows,
        "worst": worst,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOUSE_GEO_FORGE_MJX_PRIOR_BEGIN")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "record_count": prior.summary()["record_count"],
                "frame_count": len(frames),
                "warning_count": len(warning_rows),
                "worst_distance": worst[0]["score"]["distance"] if worst else None,
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_MJX_PRIOR_END")


if __name__ == "__main__":
    main()
