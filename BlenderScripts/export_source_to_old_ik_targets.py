"""Export source-bone endpoints as old-IK bridge targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import migrate_animations_to_ik as mig  # noqa: E402


FINGER_CHAINS = {
    "index": ("IndexFinger1", "IndexFinger2", "IndexFinger3"),
    "middle": ("MiddleFinger1", "MiddleFinger2", "MiddleFinger3"),
    "ring": ("RingFinger1", "RingFinger2", "RingFinger3"),
    "pinky": ("PinkyFinger1", "PinkyFinger2", "PinkyFinger3"),
}


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--target-armature", default="DeformationSystem")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-global-scale", action="store_true")
    return parser.parse_args(args_after_dash())


def vec_json(vector, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in vector]


def mapped_point(source_point, source_to_target_world, target):
    return target.matrix_world.inverted() @ (source_to_target_world @ source_point)


def mapped_head(frame_payload, source_to_target_world, target, name):
    point = frame_payload["world_heads"].get(name)
    return None if point is None else mapped_point(point, source_to_target_world, target)


def mapped_tail(frame_payload, source_to_target_world, target, name):
    point = frame_payload["world_tails"].get(name)
    return None if point is None else mapped_point(point, source_to_target_world, target)


def maybe_vec(vector):
    return None if vector is None else vec_json(vector)


def chain_payload(frame_payload, source_to_target_world, target, side: str) -> dict:
    scapula = f"Scapula_{side}"
    wrist = f"Wrist_{side}"
    fingers = f"Fingers1_{side}"
    shoulder = mapped_head(frame_payload, source_to_target_world, target, scapula)
    elbow = mapped_head(frame_payload, source_to_target_world, target, wrist)
    palm = mapped_head(frame_payload, source_to_target_world, target, fingers)
    out = {
        "arm": {
            "shoulder": maybe_vec(shoulder),
            "elbow": maybe_vec(elbow),
            "palm": maybe_vec(palm),
            "scapula_tail": maybe_vec(mapped_tail(frame_payload, source_to_target_world, target, scapula)),
            "wrist_tail": maybe_vec(mapped_tail(frame_payload, source_to_target_world, target, wrist)),
            "fingers1_tail": maybe_vec(mapped_tail(frame_payload, source_to_target_world, target, fingers)),
        },
        "fingers": {},
    }
    for label, chain in FINGER_CHAINS.items():
        points = []
        for base in chain:
            name = f"{base}_{side}"
            points.append({"bone": name, "head": maybe_vec(mapped_head(frame_payload, source_to_target_world, target, name))})
        tip_name = f"{chain[-1]}_{side}"
        out["fingers"][label] = {
            "points": points,
            "tip": maybe_vec(mapped_tail(frame_payload, source_to_target_world, target, tip_name)),
        }
    return out


def main() -> None:
    args = parse_args()
    clip = mig.collect_source_clip(
        args.source.resolve(),
        args.source_armature,
        args.frame_step,
        args.max_frames,
    )

    bpy.ops.wm.open_mainfile(filepath=str(args.target.resolve()))
    target = mig.find_armature(args.target_armature)
    target_reference_local = {pb.name: pb.matrix.copy() for pb in target.pose.bones}
    target_base_world_heads = {
        name: target.matrix_world @ matrix.translation
        for name, matrix in target_reference_local.items()
    }
    source_to_target_world, scale, count = mig.similarity_matrix_from_matrices(
        clip["per_frame"][0]["world_heads"],
        target_base_world_heads,
        mig.BODY_TAIL_ALIGNMENT_BONES,
        not args.no_global_scale,
    )

    frames = []
    for frame_payload in clip["per_frame"]:
        frames.append(
            {
                "frame": float(frame_payload["frame"]),
                "left": chain_payload(frame_payload, source_to_target_world, target, "L"),
                "right": chain_payload(frame_payload, source_to_target_world, target, "R"),
            }
        )

    payload = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_armature": clip["source_armature"],
        "source_action": clip["source_action"],
        "target_armature": target.name,
        "frame_count": len(frames),
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "source_to_target_world_scale": float(scale),
        "source_to_target_world_point_count": int(count),
        "source_to_target_world_matrix": [float(value) for row in source_to_target_world for value in row],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(frames)}, indent=2))


if __name__ == "__main__":
    main()
