import json

import bpy


def flat_matrix(matrix):
    return [round(value, 6) for row in matrix for value in row]


bindings = []
for obj in bpy.data.objects:
    if obj.type not in {"MESH", "CURVES"}:
        continue
    mods = []
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE":
            mods.append(
                {
                    "name": modifier.name,
                    "object": modifier.object.name if modifier.object else None,
                    "use_vertex_groups": getattr(modifier, "use_vertex_groups", None),
                }
            )
    if mods or obj.name.startswith("MS_Mouse"):
        bindings.append(
            {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "parent_type": obj.parent_type,
                "modifiers": mods,
                "vertex_groups": len(getattr(obj, "vertex_groups", [])),
                "hide_render": obj.hide_render,
                "hide_viewport": obj.hide_viewport,
            }
        )

cameras = []
for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        cameras.append(
            {
                "name": obj.name,
                "lens": obj.data.lens,
                "matrix_world": flat_matrix(obj.matrix_world),
                "action": obj.animation_data.action.name
                if obj.animation_data and obj.animation_data.action
                else None,
            }
        )

armatures = []
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        armatures.append(
            {
                "name": obj.name,
                "bones": len(obj.data.bones),
                "action": obj.animation_data.action.name
                if obj.animation_data and obj.animation_data.action
                else None,
                "hide_render": obj.hide_render,
                "hide_viewport": obj.hide_viewport,
                "matrix_world": flat_matrix(obj.matrix_world),
            }
        )

payload = {
    "file": bpy.data.filepath,
    "scene_camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
    "frame_start": bpy.context.scene.frame_start,
    "frame_end": bpy.context.scene.frame_end,
    "bindings": bindings,
    "cameras": cameras,
    "armatures": armatures,
}

print("MOUSE_GEO_FORGE_SCENE_INSPECT_BEGIN")
print(json.dumps(payload, ensure_ascii=False, indent=2))
print("MOUSE_GEO_FORGE_SCENE_INSPECT_END")
