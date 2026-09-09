import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


DEFAULT_MOVED = {
    "MGF_CamB_PhysicalModel",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=63)
    parser.add_argument("--end-frame", type=int, default=163)
    parser.add_argument("--step", type=int, default=5)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def world_aabb(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
    return (
        Vector(tuple(min(point[i] for point in points) for i in range(3))),
        Vector(tuple(max(point[i] for point in points) for i in range(3))),
    )


def aabb_overlap(left, right, padding=0.0):
    left_min, left_max = left
    right_min, right_max = right
    return all(
        left_min[i] <= right_max[i] + padding
        and left_max[i] + padding >= right_min[i]
        for i in range(3)
    )


def world_bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        triangles = [tuple(loop.vertices) for loop in mesh.loop_triangles]
        if not vertices or not triangles:
            return None
        return BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    finally:
        evaluated.to_mesh_clear()


def is_obstacle(obj, moved):
    if obj.type != "MESH" or obj.name in moved:
        return False
    if obj.hide_render or obj.hide_viewport:
        return False
    if obj.name.endswith(".001") and obj.name.startswith("MS_Mouse"):
        return False
    return True


def main():
    args = parse_args()
    scene = bpy.context.scene
    moved_objects = [bpy.data.objects[name] for name in DEFAULT_MOVED if name in bpy.data.objects]
    if not moved_objects:
        raise RuntimeError("Animated CamB mesh objects were not found")
    obstacles = [obj for obj in bpy.data.objects if is_obstacle(obj, DEFAULT_MOVED)]
    frames = list(range(args.start_frame, args.end_frame + 1, max(args.step, 1)))
    if frames[-1] != args.end_frame:
        frames.append(args.end_frame)

    broadphase = []
    collisions = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        moved_aabbs = {obj.name: world_aabb(obj) for obj in moved_objects}
        obstacle_aabbs = {obj.name: world_aabb(obj) for obj in obstacles}
        candidates = []
        for moved in moved_objects:
            for obstacle in obstacles:
                if aabb_overlap(moved_aabbs[moved.name], obstacle_aabbs[obstacle.name]):
                    candidates.append((moved, obstacle))
                    broadphase.append(
                        {"frame": frame, "moved": moved.name, "obstacle": obstacle.name}
                    )
        moved_bvhs = {}
        obstacle_bvhs = {}
        for moved, obstacle in candidates:
            moved_bvh = moved_bvhs.get(moved.name)
            if moved.name not in moved_bvhs:
                moved_bvh = world_bvh(moved)
                moved_bvhs[moved.name] = moved_bvh
            obstacle_bvh = obstacle_bvhs.get(obstacle.name)
            if obstacle.name not in obstacle_bvhs:
                obstacle_bvh = world_bvh(obstacle)
                obstacle_bvhs[obstacle.name] = obstacle_bvh
            if moved_bvh is None or obstacle_bvh is None:
                continue
            overlap = moved_bvh.overlap(obstacle_bvh)
            if overlap:
                collisions.append(
                    {
                        "frame": frame,
                        "moved": moved.name,
                        "obstacle": obstacle.name,
                        "triangle_pair_count": len(overlap),
                    }
                )

    report = {
        "blend": bpy.data.filepath,
        "frame_range": [args.start_frame, args.end_frame],
        "step": args.step,
        "frames_checked": frames,
        "moved_objects": [obj.name for obj in moved_objects],
        "obstacle_count": len(obstacles),
        "broadphase_candidate_count": len(broadphase),
        "collision_count": len(collisions),
        "collisions": collisions,
        "broadphase_candidates": broadphase,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MGF_CAMB_COLLISION_AUDIT_BEGIN")
    print(
        json.dumps(
            {
                "output": str(output),
                "frames_checked": len(frames),
                "obstacle_count": len(obstacles),
                "broadphase_candidate_count": len(broadphase),
                "collision_count": len(collisions),
                "collisions": collisions,
            },
            indent=2,
        )
    )
    print("MGF_CAMB_COLLISION_AUDIT_END")


if __name__ == "__main__":
    main()
