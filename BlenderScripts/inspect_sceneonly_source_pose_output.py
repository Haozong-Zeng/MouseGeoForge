#!/usr/bin/env python3

import bpy


scene = bpy.context.scene
mouse = bpy.data.objects["MS_Mouse.001"]
placement = bpy.data.objects["MGF_SourcePose_PerFramePlacement"]
for frame in (1, 2, 10, 100, 200):
    scene.frame_set(frame)
    evaluated = mouse.evaluated_get(bpy.context.evaluated_depsgraph_get())
    print(
        "POSE_AUDIT",
        frame,
        "placement",
        tuple(round(value, 3) for value in placement.location),
        "location",
        tuple(round(value, 3) for value in evaluated.matrix_world.translation),
        "dimensions",
        tuple(round(value, 3) for value in evaluated.dimensions),
        flush=True,
    )
