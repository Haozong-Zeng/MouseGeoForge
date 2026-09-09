"""Blend old-IK forelimbs using source hand-mouth semantics.

This is a targeted repair step for frames where the migrated pose drives the
palm/wrist too close to the mouth.  It blends only the requested forelimb bones
from a candidate action, chooses a per-frame alpha from source-relative hand
distances, then optionally smooths the alpha sequence before keyframing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_hand_mouth_and_finger_alignment as audit  # noqa: E402
import blend_forelimb_actions as blend_actions  # noqa: E402
import frontlimb_quality_check as qc  # noqa: E402
import probe_forelimb_interpolation as interp  # noqa: E402


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_csv_floats(values: list[str], default: list[float]) -> list[float]:
    if not values:
        return default
    out = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                out.append(float(text))
    return out or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--base-blend", type=Path, required=True)
    parser.add_argument("--candidate-blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-armature", default="DeformationSystem")
    parser.add_argument("--source-action", default=None)
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument("--base-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--candidate-action", default="MIG_BrowmFur_FGcamB")
    parser.add_argument("--side", action="append", choices=("L", "R"), default=[])
    parser.add_argument("--frame", type=float, action="append", default=[])
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--alpha", action="append", default=[])
    parser.add_argument("--include-fingers", action="store_true")
    parser.add_argument("--palm-min-source-ratio", type=float, default=0.68)
    parser.add_argument("--wrist-min-source-ratio", type=float, default=0.62)
    parser.add_argument("--fingertip-max-source-ratio", type=float, default=1.65)
    parser.add_argument("--fingers-tip-max-source-ratio", type=float, default=1.85)
    parser.add_argument("--alpha-smooth-radius", type=int, default=1)
    parser.add_argument("--alpha-change-cost", type=float, default=0.015)
    args = parser.parse_args(args_after_dash())
    args.side = args.side or ["R"]
    args.alpha = sorted(set(parse_csv_floats(args.alpha, [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.65, 0.8])))
    if 0.0 not in args.alpha:
        args.alpha.insert(0, 0.0)
    return args


def action_fcurves(action):
    fcurves = []
    if getattr(action, "fcurves", None) is not None:
        fcurves.extend(action.fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def action_frames(action) -> list[float]:
    frames = set()
    for fcurve in action_fcurves(action):
        for key in fcurve.keyframe_points:
            frames.add(round(float(key.co.x), 6))
    if frames:
        return sorted(frames)
    start, end = action.frame_range
    return [float(frame) for frame in range(math.ceil(start), math.floor(end) + 1)]


def choose_frames(args: argparse.Namespace, source_action) -> list[float]:
    if args.frame:
        return sorted(set(float(frame) for frame in args.frame))
    frames = action_frames(source_action)
    if args.frame_start is not None:
        frames = [frame for frame in frames if frame >= args.frame_start]
    if args.frame_end is not None:
        frames = [frame for frame in frames if frame <= args.frame_end]
    return frames


def frame_semantics(armature, side: str) -> dict:
    mouth, mouth_source = audit.mouth_proxy(armature)
    head, head_source = audit.head_proxy(armature)
    points = audit.hand_points(armature, side)
    return {
        "mouth_proxy": mouth_source,
        "head_proxy": head_source,
        "distance_to_mouth": {
            name: audit.distance(point, mouth) for name, point in points.items()
        },
        "distance_to_head": {
            name: audit.distance(point, head) for name, point in points.items()
        },
        "arm_chain": audit.arm_chain_metrics(points),
    }


def ratio(value, ref):
    if value is None or ref in (None, 0):
        return None
    return float(value / ref)


def collect_source_semantics(args: argparse.Namespace):
    bpy.ops.wm.open_mainfile(filepath=str(args.source_blend.resolve()))
    armature = bpy.data.objects.get(args.source_armature)
    if armature is None:
        raise RuntimeError(f"Missing source armature {args.source_armature!r}")
    action = audit.set_action(armature, args.source_action)
    frames = choose_frames(args, action)
    rows = {}
    for frame in frames:
        qc.set_frame(frame)
        key = int(round(frame))
        rows[key] = {
            side: frame_semantics(armature, side)
            for side in args.side
        }
    return action.name, frames, rows


def semantic_ratios(target_sem: dict, source_sem: dict) -> dict:
    target_mouth = target_sem["distance_to_mouth"]
    source_mouth = source_sem["distance_to_mouth"]
    target_chain = target_sem["arm_chain"]
    source_chain = source_sem["arm_chain"]
    return {
        "palm_mouth": ratio(target_mouth.get("palm"), source_mouth.get("palm")),
        "wrist_mouth": ratio(target_mouth.get("wrist_head"), source_mouth.get("wrist_head")),
        "fingers_tip_mouth": ratio(target_mouth.get("fingers_tip"), source_mouth.get("fingers_tip")),
        "fingertip_center_mouth": ratio(target_mouth.get("fingertip_center"), source_mouth.get("fingertip_center")),
        "reach": ratio(target_chain.get("reach_ratio"), source_chain.get("reach_ratio")),
        "elbow_angle": ratio(target_chain.get("elbow_angle_deg"), source_chain.get("elbow_angle_deg")),
    }


def alpha_flags(ratios: dict, args: argparse.Namespace) -> list[str]:
    flags = []
    palm = ratios.get("palm_mouth")
    wrist = ratios.get("wrist_mouth")
    fingertip = ratios.get("fingertip_center_mouth")
    fingers_tip = ratios.get("fingers_tip_mouth")
    if palm is not None and palm < args.palm_min_source_ratio:
        flags.append("palm_too_close_to_mouth")
    if wrist is not None and wrist < args.wrist_min_source_ratio:
        flags.append("wrist_too_close_to_mouth")
    if fingertip is not None and fingertip > args.fingertip_max_source_ratio:
        flags.append("fingertip_too_far_from_mouth")
    if fingers_tip is not None and fingers_tip > args.fingers_tip_max_source_ratio:
        flags.append("fingers_tip_too_far_from_mouth")
    return flags


def score_candidate(ratios: dict, alpha: float, args: argparse.Namespace) -> float:
    score = args.alpha_change_cost * float(alpha)
    palm = ratios.get("palm_mouth")
    wrist = ratios.get("wrist_mouth")
    fingertip = ratios.get("fingertip_center_mouth")
    fingers_tip = ratios.get("fingers_tip_mouth")
    reach = ratios.get("reach")

    if palm is not None:
        score += 30.0 * max(0.0, args.palm_min_source_ratio - palm) ** 2
        score += 0.12 * max(0.0, palm - 1.35) ** 2
    if wrist is not None:
        score += 10.0 * max(0.0, args.wrist_min_source_ratio - wrist) ** 2
    if fingertip is not None:
        score += 3.0 * max(0.0, fingertip - args.fingertip_max_source_ratio) ** 2
        score += 0.15 * abs(fingertip - 1.0)
    if fingers_tip is not None:
        score += 1.2 * max(0.0, fingers_tip - args.fingers_tip_max_source_ratio) ** 2
    if reach is not None:
        score += 0.02 * abs(min(reach, 8.0) - 1.0)
    return float(score)


def choose_alpha(candidates: list[dict], args: argparse.Namespace) -> dict:
    acceptable = [row for row in candidates if not row["flags"]]
    pool = acceptable or candidates
    return min(pool, key=lambda row: (row["score"], row["alpha"]))


def smooth_alpha_records(records: dict[tuple[int, str], dict], frames: list[float], sides: list[str], radius: int) -> None:
    if radius <= 0:
        for row in records.values():
            row["smoothed_alpha"] = row["selected_alpha"]
        return
    frame_keys = [int(round(frame)) for frame in frames]
    for side in sides:
        for key in frame_keys:
            values = []
            for other in frame_keys:
                if abs(other - key) <= radius:
                    row = records.get((other, side))
                    if row is not None:
                        values.append(float(row["selected_alpha"]))
            records[(key, side)]["smoothed_alpha"] = float(sum(values) / len(values)) if values else 0.0


def apply_alpha_for_frame(
    armature,
    base_action,
    candidate_action,
    frame: float,
    side: str,
    alpha: float,
    include_fingers: bool,
) -> int:
    bone_names = interp.side_bone_names(side, include_fingers)
    qc.set_action(armature, base_action)
    qc.set_frame(frame)
    base_bases = interp.decompose_pose_bases(armature, bone_names)
    qc.set_action(armature, candidate_action)
    qc.set_frame(frame)
    candidate_bases = interp.decompose_pose_bases(armature, bone_names)
    qc.set_action(armature, base_action)
    qc.set_frame(frame)
    interp.apply_interpolated_bases(armature, base_bases, candidate_bases, alpha)
    blend_actions.keyframe_bones(armature, bone_names, frame)
    return len(bone_names)


def evaluate_candidates(
    armature,
    base_action,
    candidate_action,
    frame: float,
    side: str,
    source_sem: dict,
    args: argparse.Namespace,
) -> list[dict]:
    bone_names = interp.side_bone_names(side, args.include_fingers)
    qc.set_action(armature, base_action)
    qc.set_frame(frame)
    base_bases = interp.decompose_pose_bases(armature, bone_names)
    qc.set_action(armature, candidate_action)
    qc.set_frame(frame)
    candidate_bases = interp.decompose_pose_bases(armature, bone_names)

    rows = []
    for alpha in args.alpha:
        qc.set_action(armature, base_action)
        qc.set_frame(frame)
        interp.apply_interpolated_bases(armature, base_bases, candidate_bases, alpha)
        target_sem = frame_semantics(armature, side)
        ratios = semantic_ratios(target_sem, source_sem)
        flags = alpha_flags(ratios, args)
        rows.append(
            {
                "alpha": float(alpha),
                "score": score_candidate(ratios, alpha, args),
                "flags": flags,
                "ratios": ratios,
                "target_distance_to_mouth": target_sem["distance_to_mouth"],
            }
        )
    qc.set_action(armature, base_action)
    qc.set_frame(frame)
    return rows


def main() -> None:
    args = parse_args()
    source_action, frames, source_rows = collect_source_semantics(args)

    bpy.ops.wm.open_mainfile(filepath=str(args.base_blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None:
        raise RuntimeError(f"Missing armature {args.armature!r}")
    base_action = bpy.data.actions.get(args.base_action)
    if base_action is None:
        raise RuntimeError(f"Missing base action {args.base_action!r}")
    candidate_action = blend_actions.load_candidate_action(args.candidate_blend, args.candidate_action)

    records: dict[tuple[int, str], dict] = {}
    for frame in frames:
        frame_key = int(round(frame))
        for side in args.side:
            candidates = evaluate_candidates(
                armature,
                base_action,
                candidate_action,
                frame,
                side,
                source_rows[frame_key][side],
                args,
            )
            selected = choose_alpha(candidates, args)
            records[(frame_key, side)] = {
                "frame": float(frame),
                "side": side,
                "selected_alpha": float(selected["alpha"]),
                "selected_score": float(selected["score"]),
                "selected_flags": selected["flags"],
                "selected_ratios": selected["ratios"],
                "candidate_count": len(candidates),
                "candidates": candidates,
            }

    smooth_alpha_records(records, frames, args.side, args.alpha_smooth_radius)

    keyframed = []
    for frame in frames:
        frame_key = int(round(frame))
        for side in args.side:
            row = records[(frame_key, side)]
            alpha = float(row["smoothed_alpha"])
            if alpha <= 1.0e-7:
                row["bone_count"] = 0
                continue
            row["bone_count"] = apply_alpha_for_frame(
                armature,
                base_action,
                candidate_action,
                frame,
                side,
                alpha,
                args.include_fingers,
            )
            keyframed.append({"frame": float(frame), "side": side, "alpha": alpha})

    qc.set_action(armature, base_action)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))

    report = {
        "source_blend": str(args.source_blend.resolve()),
        "source_action": source_action,
        "base_blend": str(args.base_blend.resolve()),
        "candidate_blend": str(args.candidate_blend.resolve()),
        "output": str(args.output.resolve()),
        "base_action": base_action.name,
        "candidate_action": candidate_action.name,
        "frames": [float(frame) for frame in frames],
        "sides": args.side,
        "include_fingers": bool(args.include_fingers),
        "alpha_values": [float(alpha) for alpha in args.alpha],
        "thresholds": {
            "palm_min_source_ratio": float(args.palm_min_source_ratio),
            "wrist_min_source_ratio": float(args.wrist_min_source_ratio),
            "fingertip_max_source_ratio": float(args.fingertip_max_source_ratio),
            "fingers_tip_max_source_ratio": float(args.fingers_tip_max_source_ratio),
            "alpha_smooth_radius": int(args.alpha_smooth_radius),
        },
        "keyframed": keyframed,
        "records": [records[key] for key in sorted(records)],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SEMANTIC_BLEND_FORELIMB_BEGIN")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frame_count": len(frames),
                "keyframed_count": len(keyframed),
                "include_fingers": bool(args.include_fingers),
            },
            indent=2,
        )
    )
    print("SEMANTIC_BLEND_FORELIMB_END")


if __name__ == "__main__":
    main()
