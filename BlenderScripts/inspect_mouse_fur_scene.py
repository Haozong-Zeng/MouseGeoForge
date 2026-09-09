import json

import bpy


def socket_value(socket):
    if not hasattr(socket, "default_value"):
        return None
    value = socket.default_value
    try:
        return [float(item) for item in value]
    except TypeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


def material_summary(material):
    nodes = []
    if material and material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            row = {"name": node.name, "type": node.type}
            if node.type in {"BSDF_PRINCIPLED", "BSDF_HAIR_PRINCIPLED", "RGB"}:
                row["inputs"] = {
                    socket.name: socket_value(socket)
                    for socket in node.inputs
                    if socket.name
                    in {
                        "Base Color",
                        "Color",
                        "Roughness",
                        "Radial Roughness",
                        "Melanin",
                        "Melanin Redness",
                        "Coat Weight",
                        "IOR",
                    }
                }
            if node.type == "TEX_IMAGE":
                row["image"] = node.image.filepath if node.image else None
            nodes.append(row)
    return {"name": material.name if material else None, "nodes": nodes}


def modifier_summary(modifier):
    row = {
        "name": modifier.name,
        "type": modifier.type,
        "show_viewport": modifier.show_viewport,
        "show_render": modifier.show_render,
    }
    if modifier.type == "NODES" and modifier.node_group:
        row["node_group"] = modifier.node_group.name
        values = {}
        for key in modifier.keys():
            if key == "_RNA_UI":
                continue
            value = modifier[key]
            if isinstance(value, bpy.types.ID):
                values[key] = value.name
            else:
                try:
                    values[key] = [float(item) for item in value]
                except TypeError:
                    try:
                        values[key] = float(value)
                    except (TypeError, ValueError):
                        values[key] = str(value)
        row["inputs"] = values
    return row


def main():
    scene = bpy.context.scene
    armatures = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        action = obj.animation_data.action if obj.animation_data else None
        armatures.append(
            {
                "name": obj.name,
                "bones": len(obj.data.bones),
                "action": action.name if action else None,
                "action_range": list(action.frame_range) if action else None,
            }
        )

    fur_objects = []
    for obj in bpy.data.objects:
        name = obj.name.lower()
        if obj.type != "CURVES" and not any(
            token in name for token in ("fur", "hair", "whisker", "mouth")
        ):
            continue
        surface = getattr(obj.data, "surface", None) if obj.data else None
        fur_objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "surface": surface.name if surface else None,
                "parent": obj.parent.name if obj.parent else None,
                "hide_viewport": obj.hide_viewport,
                "hide_render": obj.hide_render,
                "materials": [
                    material_summary(material) for material in obj.data.materials
                ]
                if obj.data and hasattr(obj.data, "materials")
                else [],
                "modifiers": [modifier_summary(modifier) for modifier in obj.modifiers],
                "attributes": [
                    {
                        "name": attribute.name,
                        "domain": attribute.domain,
                        "data_type": attribute.data_type,
                    }
                    for attribute in obj.data.attributes
                ]
                if obj.data and hasattr(obj.data, "attributes")
                else [],
            }
        )

    mouse_meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not (
            obj.find_armature()
            or any(token in obj.name.lower() for token in ("mouse", "body", "skin"))
        ):
            continue
        mouse_meshes.append(
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "armature": obj.find_armature().name if obj.find_armature() else None,
                "materials": [material.name for material in obj.data.materials],
                "vertex_groups": [group.name for group in obj.vertex_groups],
            }
        )

    print("MOUSE_FUR_SCENE_BEGIN")
    print(
        json.dumps(
            {
                "filepath": bpy.data.filepath,
                "frame_range": [scene.frame_start, scene.frame_end],
                "armatures": armatures,
                "fur_objects": fur_objects,
                "mouse_meshes": mouse_meshes,
            },
            indent=2,
        )
    )
    print("MOUSE_FUR_SCENE_END")


if __name__ == "__main__":
    main()
