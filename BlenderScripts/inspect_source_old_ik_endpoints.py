"""Inspect source and old-IK forelimb endpoint correspondence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


FORELIMB_BONES = [
    "Scapula",
    "Wrist",
    "Fingers1",
    "IndexFinger1",
    "IndexFinger2",
    "IndexFinger3",
    "MiddleFinger1",
    "MiddleFinger2",
    "MiddleFinger3",
    "RingFinger1",
    "RingFinger2",
    "RingFinger3",
    "PinkyFinger1",
    "PinkyFinger2",
    "PinkyFinger3",
]


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--target-armature", default="DeformationSystem")
    parser.add_argument("--frame", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args_after_dash())


def vec_json(vector: Vector) -> list[float]:
    return [round(float(value), 6) for value in vector]


def set_frame(frame: float) -> None:
    whole = int(frame)
    bpy.context.scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def pose_head_tail_world(armature, name: str) -> dict | None:
    pb = armature.pose.bones.get(name)
    if pb is None:
        return None
    head = armature.matrix_world @ pb.matrix.translation
    tail = armature.matrix_world @ (pb.matrix @ Vector((0.0, pb.bone.length, 0.0)))
    return {
        "head_world": head,
        "tail_world": tail,
        "length_world": float((tail - head).length),
    }


def rest_head_tail_local(armature, name: str) -> dict | None:
    bone = armature.data.bones.get(name)
    if bone is None:
        return None
    return {
        "parent": bone.parent.name if bone.parent else None,
        "use_connect": bool(bone.use_connect),
        "head_local": bone.head_local.copy(),
        "tail_local": bone.tail_local.copy(),
        "length": float(bone.length),
    }


def collect(path: Path, armature_name: str, frame: float) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise RuntimeError(f"Missing armature {armature_name!r} in {path}")
    set_frame(frame)
    rows = {}
    for side in ("L", "R"):
        for base in FORELIMB_BONES:
            name = f"{base}_{side}"
            rest = rest_head_tail_local(armature, name)
            pose = pose_head_tail_world(armature, name)
            if rest is None or pose is None:
                continue
            rows[name] = {
                "parent": rest["parent"],
                "use_connect": rest["use_connect"],
                "rest_length": rest["length"],
                "rest_head_local": vec_json(rest["head_local"]),
                "rest_tail_local": vec_json(rest["tail_local"]),
                "pose_head_world": vec_json(pose["head_world"]),
                "pose_tail_world": vec_json(pose["tail_world"]),
                "pose_length_world": pose["length_world"],
            }
    return {
        "path": str(path.resolve()),
        "armature": armature.name,
        "frame": frame,
        "bones": rows,
    }


def add_chain_gaps(payload: dict) -> None:
    gaps = {}
    for side in ("L", "R"):
        chains = [
            ["Scapula", "Wrist", "Fingers1"],
            ["Fingers1", "IndexFinger1", "IndexFinger2", "IndexFinger3"],
            ["Fingers1", "MiddleFinger1", "MiddleFinger2", "MiddleFinger3"],
            ["Fingers1", "RingFinger1", "RingFinger2", "RingFinger3"],
            ["Fingers1", "PinkyFinger1", "PinkyFinger2", "PinkyFinger3"],
        ]
        for chain in chains:
            for parent, child in zip(chain, chain[1:]):
                p_name = f"{parent}_{side}"
                c_name = f"{child}_{side}"
                p_row = payload["bones"].get(p_name)
                c_row = payload["bones"].get(c_name)
                if not p_row or not c_row:
                    continue
                rest_gap = (
                    Vector(p_row["rest_tail_local"]) - Vector(c_row["rest_head_local"])
                ).length
                pose_gap = (
                    Vector(p_row["pose_tail_world"]) - Vector(c_row["pose_head_world"])
                ).length
                gaps[f"{p_name}->{c_name}"] = {
                    "rest_tail_to_child_head": float(rest_gap),
                    "pose_tail_to_child_head": float(pose_gap),
                }
    payload["chain_gaps"] = gaps


def main() -> None:
    args = parse_args()
    source = collect(args.source, args.source_armature, args.frame)
    target = collect(args.target, args.target_armature, args.frame)
    add_chain_gaps(source)
    add_chain_gaps(target)
    payload = {"source": source, "target": target}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
