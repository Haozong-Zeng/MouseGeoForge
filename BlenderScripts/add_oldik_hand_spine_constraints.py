"""Add hand and spine stretch/rotation constraints to the old IK blend.

The active constraints are conservative: LIMIT_SCALE locks bone stretch to the
current pose scale, and LIMIT_ROTATION windows are sampled from the existing
action with a configurable margin.  Optional muted STRETCH_TO constraints link
palm/fingertip bones to the visible hand contact sphere proxies for later
controller refinement without changing the current animation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


SPINE_BONES = ("Spine1_M", "Spine2_M", "Spine3_M", "Spine4_M", "Chest_M")
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--action", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--constraint-prefix", default="MGF")
    parser.add_argument("--finger-margin-deg", type=float, default=35.0)
    parser.add_argument("--palm-margin-deg", type=float, default=45.0)
    parser.add_argument(
        "--finger-rotation-mode",
        choices=["orbbec", "full-turn", "sampled", "none"],
        default="orbbec",
        help="Finger LIMIT_ROTATION mode. orbbec uses narrow local X/Y/Z windows matching the non-finger Orbbec convention.",
    )
    parser.add_argument("--finger-orbbec-xz-deg", type=float, default=30.0)
    parser.add_argument("--finger-orbbec-y-deg", type=float, default=10.0)
    parser.add_argument("--spine-margin-x-deg", type=float, default=30.0)
    parser.add_argument("--spine-margin-y-deg", type=float, default=10.0)
    parser.add_argument("--spine-margin-z-deg", type=float, default=30.0)
    parser.add_argument(
        "--finger-full-turn-rotation-window",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag: same as --finger-rotation-mode full-turn.",
    )
    parser.add_argument(
        "--sample-finger-rotation-window",
        dest="sample_finger_rotation_window",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag: same as --finger-rotation-mode sampled.",
    )
    parser.add_argument(
        "--no-finger-rotation-window",
        dest="no_finger_rotation_window",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag: same as --finger-rotation-mode none.",
    )
    parser.add_argument(
        "--legacy-full-turn-finger-rotation-window",
        action="store_true",
        default=False,
        help="Use [-180, 180] local rotation limits on finger bones to prevent multi-turn twists without changing the current baked pose.",
    )
    parser.add_argument("--add-muted-stretch-to-proxies", action="store_true", default=True)
    parser.add_argument("--no-muted-stretch-to-proxies", dest="add_muted_stretch_to_proxies", action="store_false")
    parser.add_argument("--proxy-prefix", default="MGF_HandSphere")
    return parser.parse_args(args_after_dash())


def set_frame(frame: int) -> None:
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


def choose_action(armature, action_name: str | None):
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action {action_name!r} not found.")
    elif armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
    elif bpy.data.actions:
        action = sorted(bpy.data.actions, key=lambda item: item.name)[0]
    else:
        raise RuntimeError("No action found.")
    armature.animation_data_create()
    armature.animation_data.action = action
    return action


def action_frames(action, scene, start: int | None, end: int | None, step: int) -> list[int]:
    first = int(math.floor(action.frame_range[0])) if start is None else int(start)
    last = int(math.ceil(action.frame_range[1])) if end is None else int(end)
    if last < first:
        first, last = int(scene.frame_start), int(scene.frame_end)
    return list(range(first, last + 1, max(int(step), 1)))


def all_finger_bones() -> list[str]:
    names = []
    for side in ("L", "R"):
        names.append(f"Fingers1_{side}")
        for chain in FINGER_CHAINS.values():
            for base in chain:
                names.append(f"{base}_{side}")
    return names


def remove_named_constraint(pb, name: str) -> None:
    for constraint in list(pb.constraints):
        if constraint.name == name:
            pb.constraints.remove(constraint)


def configure_limit_scale(pb, name: str, scale_xyz) -> dict:
    remove_named_constraint(pb, name)
    constraint = pb.constraints.new(type="LIMIT_SCALE")
    constraint.name = name
    constraint.influence = 1.0
    constraint.mute = False
    for attr in ("use_min_x", "use_min_y", "use_min_z", "use_max_x", "use_max_y", "use_max_z"):
        if hasattr(constraint, attr):
            setattr(constraint, attr, True)
    sx, sy, sz = [float(v) for v in scale_xyz]
    constraint.min_x = sx
    constraint.max_x = sx
    constraint.min_y = sy
    constraint.max_y = sy
    constraint.min_z = sz
    constraint.max_z = sz
    constraint.owner_space = "LOCAL"
    return {
        "name": constraint.name,
        "type": constraint.type,
        "scale": [sx, sy, sz],
        "mute": bool(constraint.mute),
        "influence": float(constraint.influence),
    }


def configure_limit_rotation(pb, name: str, bounds: dict[str, tuple[float, float]]) -> dict:
    remove_named_constraint(pb, name)
    constraint = pb.constraints.new(type="LIMIT_ROTATION")
    constraint.name = name
    constraint.influence = 1.0
    constraint.mute = False
    for axis in ("x", "y", "z"):
        use_attr = f"use_limit_{axis}"
        if hasattr(constraint, use_attr):
            setattr(constraint, use_attr, True)
        setattr(constraint, f"min_{axis}", float(bounds[axis][0]))
        setattr(constraint, f"max_{axis}", float(bounds[axis][1]))
    constraint.owner_space = "LOCAL"
    if hasattr(constraint, "use_transform_limit"):
        constraint.use_transform_limit = True
    return {
        "name": constraint.name,
        "type": constraint.type,
        "bounds_deg": {
            axis: [math.degrees(bounds[axis][0]), math.degrees(bounds[axis][1])]
            for axis in ("x", "y", "z")
        },
        "mute": bool(constraint.mute),
        "influence": float(constraint.influence),
    }


def euler_samples_for_bone(armature, bone_name: str, frames: list[int]) -> list[tuple[float, float, float]]:
    pb = armature.pose.bones[bone_name]
    previous = None
    samples = []
    for frame in frames:
        set_frame(frame)
        euler = pb.matrix_basis.to_euler("XYZ")
        if previous is not None:
            euler.make_compatible(previous)
        samples.append((float(euler.x), float(euler.y), float(euler.z)))
        previous = euler.copy()
    return samples


def sampled_rotation_bounds(samples: list[tuple[float, float, float]], margins: tuple[float, float, float]) -> dict[str, tuple[float, float]]:
    bounds = {}
    for index, axis in enumerate(("x", "y", "z")):
        values = [row[index] for row in samples]
        margin = margins[index]
        low = min(values) - margin
        high = max(values) + margin
        # Limit Rotation behaves best when the window stays in one turn.
        low = max(low, -math.pi)
        high = min(high, math.pi)
        if low > high:
            low, high = -math.pi, math.pi
        bounds[axis] = (low, high)
    return bounds


def proxy_name_for(side: str, role: str, prefix: str) -> str:
    return f"{prefix}_{side}_{role}"


def configure_muted_stretch_to(pb, name: str, target_obj) -> dict:
    remove_named_constraint(pb, name)
    constraint = pb.constraints.new(type="STRETCH_TO")
    constraint.name = name
    constraint.target = target_obj
    constraint.influence = 0.0
    constraint.mute = True
    if hasattr(constraint, "volume"):
        constraint.volume = "NO_VOLUME"
    return {
        "name": constraint.name,
        "type": constraint.type,
        "target": target_obj.name,
        "mute": bool(constraint.mute),
        "influence": float(constraint.influence),
    }


def constraint_payload(constraint) -> dict:
    row = {
        "name": constraint.name,
        "type": constraint.type,
        "target": getattr(getattr(constraint, "target", None), "name", None),
        "subtarget": getattr(constraint, "subtarget", ""),
        "mute": bool(getattr(constraint, "mute", False)),
        "influence": float(getattr(constraint, "influence", 0.0)),
    }
    for attr in (
        "owner_space",
        "target_space",
        "use_limit_x",
        "use_limit_y",
        "use_limit_z",
        "use_min_x",
        "use_min_y",
        "use_min_z",
        "use_max_x",
        "use_max_y",
        "use_max_z",
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "volume",
    ):
        if hasattr(constraint, attr):
            value = getattr(constraint, attr)
            row[attr] = float(value) if isinstance(value, (int, float)) else value
    return row


def main() -> None:
    args = parse_args()
    if args.finger_full_turn_rotation_window or args.legacy_full_turn_finger_rotation_window:
        args.finger_rotation_mode = "full-turn"
    if args.sample_finger_rotation_window:
        args.finger_rotation_mode = "sampled"
    if args.no_finger_rotation_window:
        args.finger_rotation_mode = "none"
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"Armature {args.armature!r} not found.")
    action = choose_action(armature, args.action)
    frames = action_frames(action, bpy.context.scene, args.frame_start, args.frame_end, args.frame_step)

    length_name = f"{args.constraint_prefix}_StretchLengthLock"
    rotation_name = f"{args.constraint_prefix}_RotationWindow"
    stretch_name = f"{args.constraint_prefix}_StretchToProxy_MUTED"

    changed = {}
    spine_margins = (
        math.radians(args.spine_margin_x_deg),
        math.radians(args.spine_margin_y_deg),
        math.radians(args.spine_margin_z_deg),
    )
    finger_margin = math.radians(args.finger_margin_deg)
    palm_margin = math.radians(args.palm_margin_deg)

    constrained_bones = [name for name in SPINE_BONES if name in armature.pose.bones]
    constrained_bones.extend(name for name in all_finger_bones() if name in armature.pose.bones)

    for bone_name in constrained_bones:
        pb = armature.pose.bones[bone_name]
        pb.rotation_mode = "XYZ" if bone_name in SPINE_BONES else pb.rotation_mode
        margins = spine_margins if bone_name in SPINE_BONES else (
            (palm_margin, palm_margin, palm_margin)
            if bone_name.startswith("Fingers1_")
            else (finger_margin, finger_margin, finger_margin)
        )
        samples = euler_samples_for_bone(armature, bone_name, frames)
        is_finger = bone_name not in SPINE_BONES
        if is_finger and args.finger_rotation_mode == "full-turn":
            bounds = {axis: (-math.pi, math.pi) for axis in ("x", "y", "z")}
        elif is_finger and args.finger_rotation_mode == "orbbec":
            xz = math.radians(args.finger_orbbec_xz_deg)
            y = math.radians(args.finger_orbbec_y_deg)
            bounds = {"x": (-xz, xz), "y": (-y, y), "z": (-xz, xz)}
        else:
            bounds = sampled_rotation_bounds(samples, margins)
        changed[bone_name] = {
            "limit_scale": configure_limit_scale(pb, length_name, pb.scale),
        }
        if not (is_finger and args.finger_rotation_mode == "none"):
            changed[bone_name]["limit_rotation"] = configure_limit_rotation(pb, rotation_name, bounds)

    if args.add_muted_stretch_to_proxies:
        proxy_roles = {
            "Fingers1": "palm",
            "IndexFinger3": "index_tip",
            "MiddleFinger3": "middle_tip",
            "RingFinger3": "ring_tip",
            "PinkyFinger3": "pinky_tip",
        }
        for side in ("L", "R"):
            for bone_base, role in proxy_roles.items():
                bone_name = f"{bone_base}_{side}"
                pb = armature.pose.bones.get(bone_name)
                target = bpy.data.objects.get(proxy_name_for(side, role, args.proxy_prefix))
                if pb is None or target is None:
                    continue
                changed.setdefault(bone_name, {})["stretch_to_proxy"] = configure_muted_stretch_to(
                    pb,
                    stretch_name,
                    target,
                )

    set_frame(frames[0])
    bpy.context.view_layer.update()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    final_constraints = {}
    for bone_name in constrained_bones:
        pb = armature.pose.bones[bone_name]
        final_constraints[bone_name] = [constraint_payload(constraint) for constraint in pb.constraints]

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "armature": armature.name,
        "action": action.name,
        "frames": frames,
        "constraint_names": {
            "length": length_name,
            "rotation": rotation_name,
            "stretch_to_proxy": stretch_name,
        },
        "finger_rotation_mode": args.finger_rotation_mode,
        "finger_orbbec_limits_deg": {
            "x": [-args.finger_orbbec_xz_deg, args.finger_orbbec_xz_deg],
            "y": [-args.finger_orbbec_y_deg, args.finger_orbbec_y_deg],
            "z": [-args.finger_orbbec_xz_deg, args.finger_orbbec_xz_deg],
        },
        "spine_bones": [name for name in SPINE_BONES if name in armature.pose.bones],
        "finger_bones": [name for name in all_finger_bones() if name in armature.pose.bones],
        "changed": changed,
        "final_constraints": final_constraints,
    }
    report_path = args.report or args.output.with_suffix(".hand-spine-constraints-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
