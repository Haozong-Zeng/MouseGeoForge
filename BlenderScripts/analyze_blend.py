import json
import sys
from pathlib import Path

import bpy


def fcurve_summary(action):
    if not action:
        return None
    fcurves = []
    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        fcurves.extend(legacy_fcurves)
    elif getattr(action, "is_action_layered", False):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)

    frames = []
    data_paths = {}
    for fc in fcurves:
        if fc.keyframe_points:
            frames.append(fc.range()[0])
            frames.append(fc.range()[1])
        data_paths[fc.data_path] = data_paths.get(fc.data_path, 0) + 1
    return {
        "name": action.name,
        "layered": bool(getattr(action, "is_action_layered", False)),
        "slots": len(getattr(action, "slots", [])),
        "layers": len(getattr(action, "layers", [])),
        "fcurves": len(fcurves),
        "frame_range": [min(frames), max(frames)] if frames else None,
        "data_paths_sample": sorted(data_paths.items())[:20],
    }


def armature_summary(obj, compact=False):
    action = obj.animation_data.action if obj.animation_data else None
    constraints = {}
    for pb in obj.pose.bones:
        if pb.constraints:
            constraints[pb.name] = [
                {"name": c.name, "type": c.type, "target": getattr(getattr(c, "target", None), "name", None)}
                for c in pb.constraints
            ]
    return {
        "name": obj.name,
        "data": obj.data.name,
        "bones": len(obj.data.bones),
        "pose_bones": len(obj.pose.bones),
        "bone_names": [] if compact else [b.name for b in obj.data.bones],
        "bone_names_sample": [b.name for b in obj.data.bones[:20]],
        "root_bones": [b.name for b in obj.data.bones if not b.parent],
        "matrix_world": [round(v, 6) for row in obj.matrix_world for v in row],
        "constraints": constraints,
        "action": fcurve_summary(action),
    }


def main():
    compact = "--compact" in sys.argv
    blend = Path(bpy.data.filepath)
    summary = {
        "file": str(blend),
        "objects": [] if compact else [
            {"name": obj.name, "type": obj.type, "data": getattr(obj.data, "name", None)}
            for obj in bpy.data.objects
        ],
        "armatures": [
            armature_summary(obj, compact=compact)
            for obj in bpy.data.objects
            if obj.type == "ARMATURE"
        ],
        "actions": [fcurve_summary(action) for action in bpy.data.actions],
    }
    print("MOUSE_GEO_FORGE_SUMMARY_BEGIN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("MOUSE_GEO_FORGE_SUMMARY_END")


if __name__ == "__main__":
    main()
