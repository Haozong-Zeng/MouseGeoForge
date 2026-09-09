"""
This script is able to generate synthetic dataset for training DensePose CSE, PointRend, and Depth Anything with 
the Blender 4.0 PyMouse Studio project. 
Written by Haozong Zeng, The University of British Columbia, in Dr. Tim Murphy's Lab.
"""

import bpy
import bpy_extras
import bmesh
import argparse
import mathutils
import pickle
import sys
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
import random
from PIL import Image, ImageDraw
import math
import os
import cv2
import OpenEXR
import Imath
import numpy as np
from scipy.spatial import KDTree, ConvexHull, distance_matrix
from sklearn.neighbors import KDTree
import shutil
import json
import time
import copy
from math import tan
import re
from pathlib import Path
    
def read_exr(exr_path):
    # 打开EXR文件
    exr_file = OpenEXR.InputFile(exr_path)
    
    # 获取数据窗口大小
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 获取所有通道名称
    channels = list(exr_file.header()['channels'].keys())

    # Prefer explicit depth-like channels. File Output depth sockets may name
    # the single channel differently across Blender/OpenEXR builds.
    selected_channel = None
    for candidate in ["Z", "Depth", "V", "Y", "R"]:
        if candidate in channels:
            selected_channel = candidate
            break
    if selected_channel is None and channels:
        selected_channel = channels[0]

    # 如果没有找到通道，则返回 None
    if not selected_channel:
        print("No channels found in the EXR file.")
        return None

    # 读取选中的通道
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel(selected_channel, pt)
    
    # 将通道数据转换为numpy数组
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])  # 调整数组形状

    return depth

def remove_close_points(points, ids, min_dist):
        tree = KDTree(points)
        to_remove = set()
        for i in range(points.shape[0]):
            if i in to_remove:
                continue
            ind = tree.query_radius(points[i:i+1], r=min_dist, count_only=False, return_distance=False)
            for j in ind[0]:
                if j != i:
                    to_remove.add(j)
        remaining_ids = np.delete(ids, list(to_remove), axis=0)
        points = np.delete(points, list(to_remove), axis=0)
        return points, remaining_ids

def process_points(image_id, 
                    depth_exr_path, 
                    depth_png_path, 
                    vertex_obj, 
                    temp_mesh, 
                    cam, 
                    render_width, 
                    render_height, 
                    scene=None,
                    vertex_id_to_body_parts=None, 
                    min_dist_to_edge=8, 
                    min_dist_between_points=12):
    """
    Process the given points to calculate the bounding box, filter points that are too close to each other or to the edge,
    and calculate the area inside the contour.
    Outputs include vertex IDs, bounding box, segmentation coordinates, area, and coordinates corresponding to the vertex IDs.
    """

    vertex_pixel_dict = {}
    vertex_cam_dist_dict = {}
    scene = scene or bpy.context.scene
    # 获取相机的世界矩阵
    cam_matrix_world = cam.matrix_world
    # 计算相机的逆矩阵
    cam_matrix_world_inv = cam_matrix_world.inverted()
    # 计算顶点在图片中的像素位置
    for vertex_id in range(len(temp_mesh.vertices)):
        vertex = temp_mesh.vertices[vertex_id].co
        world_vertex_loc = vertex_obj.matrix_world @ vertex
        
        # 提取前向向量（负的Z轴方向）
        forward_vector = -cam_matrix_world.to_3x3().col[2]

        # 将顶点坐标转换到相机坐标系
        cam_vertex_loc = cam_matrix_world_inv @ world_vertex_loc
        
        # get the Z component as the depth value. 
        # this value needs to be multiplied by the length of camera forward vector 
        # to obtain the same absolute depth as the rendered depth map.
        depth = -cam_vertex_loc.z * (Vector(forward_vector).length)
        # 使用world_to_camera_view转换世界坐标到相机视图坐标
        camera_coord = world_to_camera_view(scene, cam, world_vertex_loc)
        # 转换为像素坐标
        pixel_x = round(camera_coord.x * render_width)
        pixel_y = round(render_height - camera_coord.y * render_height)
        
        vertex_pixel_dict[vertex_id] = (pixel_x, pixel_y)
        vertex_cam_dist_dict[vertex_id] = depth

    depth_map = read_exr(depth_exr_path)
    vertex_pixel_dict_copy = copy.deepcopy(vertex_pixel_dict)

    # loop through every vertex and verify if it's visible
    for vertex_id, (pixel_x, pixel_y) in vertex_pixel_dict_copy.items():
        # get the depth value from the depth map
        # pay attention to the different definitions of the x and y axes
        if pixel_y >= render_height or pixel_x >= render_width or pixel_y <= 0 or pixel_x <= 0: # remove vertex out of view
            del vertex_pixel_dict[vertex_id]
        else:
            depth_value = depth_map[pixel_y, pixel_x]
            ##print(f"depth_value is {depth_value}")
            # the calculated depth value of the vertex
            vertex_depth = vertex_cam_dist_dict[vertex_id]
            ##print(f"vertex_depth is {vertex_depth}")

            # remove invisible vertices, allow a small error and exclude the edge error
            if vertex_depth <= depth_value * 1.01 and depth_value <= 0.8 * np.max(depth_map):
                pass
            else:
                del vertex_pixel_dict[vertex_id]
        
    # convert the input dictionary to a numpy array for processing
    points = np.array(list(vertex_pixel_dict.values())).astype(np.float32)
    vertex_ids = np.array(list(vertex_pixel_dict.keys())).astype(np.float32)

    return vertex_pixel_dict


def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)  # 将 float32 转换为标准的 float
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
def get_bone_head_pixel_coordinates(bone_name, camera_name, armature_name='DeformationSystem'):
    """
    获取骨骼头部在相机平面的像素坐标

    参数:
    bone_name (str): 骨骼的名称
    camera_name (str): 相机对象的名称
    armature_name (str): 骨骼对象（骨架）的名称

    返回:
    tuple: 骨骼头部在相机平面的像素坐标 (x, y)
    """
    
    # 获取骨骼和相机对象
    armature = bpy.data.objects[armature_name]
    camera = bpy.data.objects[camera_name]
    
    # 获取骨骼头部的世界坐标
    pose_bone = armature.pose.bones[bone_name]
    fraction = keypoint_position_fractions.get(bone_name, 0.0)
    bone_point = pose_bone.head.lerp(pose_bone.tail, fraction)
    bone_head_world = armature.matrix_world @ bone_point

    # 将世界坐标转换为相机视图坐标
    scene = bpy.context.scene
    coord_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, bone_head_world)
    
    # 获取渲染分辨率
    render = scene.render
    render_width = render.resolution_x * render.resolution_percentage / 100
    render_height = render.resolution_y * render.resolution_percentage / 100
    
    # 将归一化的相机视图坐标转换为像素坐标
    pixel_x = coord_2d.x * render_width
    pixel_y = (1.0 - coord_2d.y) * render_height  # y坐标需要反转

    return pixel_x, pixel_y

def check_bone_visibility(visible_vertices_dict, vertices_keypoints_dict):
    # Step 1: 提取可见的顶点ID
    visible_vertex_ids = list(set(visible_vertices_dict.keys()))
    
    # Step 2: 创建一个字典来保存每个骨骼ID是否可见
    bone_visibility = {bone_name: False for bone_name in list(vertices_keypoints_dict.values())}
    
    # Step 3: 遍历每个顶点ID，检查是否在可见顶点集合中
    for vertex_id, bone_name in zip(list(vertices_keypoints_dict.keys()), list(vertices_keypoints_dict.values())):
        if vertex_id in visible_vertex_ids:
            bone_visibility[bone_name] = True
    
    return bone_visibility

###############################################################################################################
####################################### BlendedMVS-style Export ###############################################
###############################################################################################################

right_hand_keypoints = [
    'PinkyFinger3_R', 'PinkyFinger2_R', 'PinkyFinger1_R',
    'RingFinger3_R', 'RingFinger2_R', 'RingFinger1_R',
    'MiddleFinger3_R', 'MiddleFinger2_R', 'MiddleFinger1_R',
    'IndexFinger3_R', 'IndexFinger2_R', 'IndexFinger1_R', 'Fingers1_R'
]

left_hand_keypoints = [
    'PinkyFinger3_L', 'PinkyFinger2_L', 'PinkyFinger1_L',
    'RingFinger3_L', 'RingFinger2_L', 'RingFinger1_L',
    'MiddleFinger3_L', 'MiddleFinger2_L', 'MiddleFinger1_L',
    'IndexFinger3_L', 'IndexFinger2_L', 'IndexFinger1_L', 'Fingers1_L'
]

additional_keypoints = ['Wrist_L', 'Wrist_R', 'HeadEnd_M']
keypoint_position_fractions = {'HeadEnd_M': 0.27}
DEFAULT_VERTEX_KEYPOINT_MAP_NAME = "mouse_hand_vertex_keypoint_map.pkl"
PHYSICAL_CAMERA_OBJECT_PREFIXES = (
    "CameraItems.",
    "Part_CameraIO.",
    "Part_CameraLens.",
    "Part_CameraMount.",
    "Part_CameraPCB.",
    "Part_CamLen.",
)


def parse_blender_args():
    parser = argparse.ArgumentParser(
        description="Render water-reaching animation frames as BlendedMVS-style multi-view scenes."
    )
    parser.add_argument("--output-root", default="//water_reaching_blendedmvs")
    parser.add_argument(
        "--cameras",
        default="AUTO",
        help=(
            "Comma-separated camera object names. AUTO prefers CameraA-E when present; "
            "otherwise it discovers non-reference camera objects and orders them left-to-right."
        ),
    )
    parser.add_argument("--camera-model-collections", default="CamA,CamB,CamC")
    parser.add_argument(
        "--hide-physical-camera-prefixes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the legacy global object-name fallback for physical camera models.",
    )
    parser.add_argument(
        "--per-camera-hide-collections",
        default="",
        help=(
            "Semicolon-separated camera=collection mapping. The mapped physical rig is "
            "hidden only while rendering from that camera."
        ),
    )
    parser.add_argument(
        "--depth-ignore-collections",
        default="Bbox",
        help=(
            "Comma-separated collections visible in RGB but excluded from depth supervision. "
            "Pixels covered by these objects are written as zero depth by default."
        ),
    )
    parser.add_argument(
        "--per-camera-depth-ignore-collections",
        default="",
        help=(
            "Semicolon-separated camera=collection mapping. Objects in the mapped "
            "collection are excluded only from that camera's depth pass."
        ),
    )
    parser.add_argument(
        "--per-camera-depth-show-collections",
        default="",
        help=(
            "Semicolon-separated camera=collection mapping. Objects in the mapped "
            "collection are temporarily made renderable only for that camera's depth pass."
        ),
    )
    parser.add_argument(
        "--depth-ignore-mode",
        choices=["mask-invalid", "hide-only", "none"],
        default="mask-invalid",
        help=(
            "mask-invalid renders an ignore mask and writes covered depth pixels as 0; "
            "hide-only only hides ignored objects during the depth pass; none keeps them in depth."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--frames", default="",
        help="Optional comma-separated exact frames within start/end; useful for filling gaps.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame in the requested range (default: 1).",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, skip complete frame folders and restart from the first incomplete frame. "
            "Use --no-resume to render every requested frame."
        ),
    )
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--resolution-x", type=int, default=None)
    parser.add_argument("--resolution-y", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--engine", choices=["KEEP", "CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"], default="CYCLES")
    parser.add_argument("--cycles-samples", type=int, default=64)
    parser.add_argument("--depth-num", type=int, default=128)
    parser.add_argument("--save-depth-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-depth-exr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--depth-exclude-hair-curves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude CURVES hair from depth while retaining it in RGB.",
    )
    parser.add_argument("--save-marked-keypoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mouse-model", default="MS_Mouse")
    parser.add_argument("--armature", default="DeformationSystem")
    parser.add_argument(
        "--vertex-keypoint-map",
        default="AUTO",
        help=(
            "Vertex-to-keypoint pickle. AUTO searches beside the current .blend, "
            "its keypoint_mapping directory, and the renderer script directory. "
            "Pass an empty string to disable keypoint export."
        ),
    )
    parser.add_argument("--metadata-only", action=argparse.BooleanOptionalAction, default=False)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def blender_path(path_text):
    return Path(bpy.path.abspath(path_text)).resolve()


def ensure_clean_dir(path, overwrite=True):
    if path.exists() and overwrite:
        resolved = path.resolve()
        if len(resolved.parts) < 4 or resolved == resolved.parent:
            raise ValueError(f"Refuse to overwrite suspiciously broad path: {resolved}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collection_objects_recursive(collection):
    objects = list(collection.objects)
    for child in collection.children:
        objects.extend(collection_objects_recursive(child))
    return objects


def parse_name_list(text):
    return [name.strip() for name in text.split(",") if name.strip()]


def parse_camera_collection_map(text):
    result = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        camera_name, separator, collection_name = item.partition("=")
        if not separator or not camera_name.strip() or not collection_name.strip():
            raise ValueError(f"Invalid camera=collection mapping: {item!r}")
        result[camera_name.strip()] = collection_name.strip()
    return result


def resolve_camera_names(text):
    if text.strip().upper() != "AUTO":
        return parse_name_list(text)
    preferred = [
        name
        for name in ("CameraA", "CameraB", "CameraC", "CameraD", "CameraE")
        if bpy.data.objects.get(name) is not None
        and bpy.data.objects[name].type == "CAMERA"
    ]
    if preferred:
        print(f"AUTO cameras: {','.join(preferred)}")
        return preferred

    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "CAMERA" or obj.name.lower().startswith("pasted__"):
            continue
        collection_names = {collection.name.lower() for collection in obj.users_collection}
        if any("reference" in name or "refference" in name for name in collection_names):
            continue
        candidates.append(obj)
    candidates.sort(key=lambda obj: (float(obj.matrix_world.translation.x), obj.name))
    if not candidates and bpy.context.scene.camera is not None:
        candidates = [bpy.context.scene.camera]
    if not candidates:
        raise ValueError("AUTO camera discovery found no usable CAMERA objects")
    names = [obj.name for obj in candidates]
    print(f"AUTO cameras: {','.join(names)}")
    return names


def objects_from_collection_names(collection_names, purpose):
    objects_by_name = {}
    found_collections = []
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            print(f"warning: collection {collection_name!r} not found; skip {purpose}")
            continue
        found_collections.append(collection_name)
        for obj in collection_objects_recursive(collection):
            objects_by_name[obj.name] = obj
    return list(objects_by_name.values()), found_collections


def hide_camera_model_collections(collection_names, hide_prefix_fallback=True):
    """Hide physical camera models from render and return states for restoration."""
    states = []
    visited = set()
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            print(f"warning: collection {collection_name!r} not found; skip render-visibility hiding")
            continue
        pointer = collection.as_pointer()
        if pointer in visited:
            continue
        visited.add(pointer)
        states.append(("collection", collection, collection.hide_render))
        collection.hide_render = True
        for obj in collection_objects_recursive(collection):
            pointer = obj.as_pointer()
            if pointer in visited:
                continue
            visited.add(pointer)
            states.append(("object", obj, obj.hide_render))
            obj.hide_render = True
    fallback_objects = []
    if hide_prefix_fallback:
        for obj in bpy.data.objects:
            if not obj.name.startswith(PHYSICAL_CAMERA_OBJECT_PREFIXES):
                continue
            pointer = obj.as_pointer()
            if pointer in visited:
                continue
            visited.add(pointer)
            states.append(("object", obj, obj.hide_render))
            obj.hide_render = True
            fallback_objects.append(obj.name)
    if fallback_objects:
        print(
            "hidden physical camera model objects by prefix: "
            + ",".join(sorted(fallback_objects))
        )
    return states


def restore_render_visibility(states):
    for kind, datablock, hide_render in reversed(states):
        datablock.hide_render = hide_render


def hide_objects_for_render(objects):
    states = []
    for obj in objects:
        states.append(("object", obj, obj.hide_render))
        obj.hide_render = True
    return states


def show_objects_for_render(objects):
    states = []
    for obj in objects:
        states.append(("object", obj, obj.hide_render))
        obj.hide_render = False
    return states


def get_render_size(scene):
    scale = scene.render.resolution_percentage / 100.0
    return int(scene.render.resolution_x * scale), int(scene.render.resolution_y * scale)


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        return "HORIZONTAL" if size_x >= size_y else "VERTICAL"
    return sensor_fit


def get_sensor_size(sensor_fit, sensor_width, sensor_height):
    return sensor_height if sensor_fit == "VERTICAL" else sensor_width


def camera_intrinsic_matrix(scene, camera_obj):
    cam_data = camera_obj.data
    width_px, height_px = get_render_size(scene)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * width_px,
        scene.render.pixel_aspect_y * height_px,
    )
    sensor_size_mm = get_sensor_size(sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    view_fac_px = width_px if sensor_fit == "HORIZONTAL" else pixel_aspect_ratio * height_px
    pixel_size_mm_per_px = sensor_size_mm / cam_data.lens / view_fac_px
    fx = 1.0 / pixel_size_mm_per_px
    fy = fx / pixel_aspect_ratio
    cx = width_px / 2.0 - cam_data.shift_x * view_fac_px
    cy = height_px / 2.0 + cam_data.shift_y * view_fac_px / pixel_aspect_ratio
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_extrinsic_matrix(camera_obj):
    blender_to_opencv = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    # Camera rigs can inherit a non-unit parent scale. Export only the rigid
    # pose so the rotation remains orthonormal and matches metric depth.
    location, rotation_world, _scale = camera_obj.matrix_world.decompose()
    world_to_camera_rotation = rotation_world.to_matrix().transposed()
    world_to_camera_translation = -(world_to_camera_rotation @ location)
    rot = blender_to_opencv @ world_to_camera_rotation
    trans = blender_to_opencv @ world_to_camera_translation
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = np.array([[rot[row][col] for col in range(3)] for row in range(3)], dtype=np.float64)
    extrinsic[:3, 3] = np.array([trans[idx] for idx in range(3)], dtype=np.float64)
    return extrinsic


def write_cam_file(path, extrinsic, intrinsic, depth, depth_num):
    valid = depth[np.isfinite(depth) & (depth > 1e-8)]
    if valid.size:
        depth_min = float(np.percentile(valid, 0.1))
        depth_max = float(np.percentile(valid, 99.9))
        if depth_max <= depth_min:
            depth_max = float(valid.max())
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0
    else:
        depth_min, depth_max = 0.1, 10.0
    depth_interval = (depth_max - depth_min) / max(depth_num - 1, 1)
    with path.open("w", encoding="utf-8") as f:
        f.write("extrinsic\n")
        for row in extrinsic:
            f.write(" ".join(f"{value:.10g}" for value in row) + "\n")
        f.write("\nintrinsic\n")
        for row in intrinsic:
            f.write(" ".join(f"{value:.10g}" for value in row) + "\n")
        f.write(f"\n{depth_min:.10g} {depth_interval:.10g} {depth_num:d} {depth_max:.10g}\n")


def write_pair_file(path, view_count):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{view_count}\n")
        for ref in range(view_count):
            srcs = [idx for idx in range(view_count) if idx != ref]
            f.write(f"{ref}\n")
            parts = [str(len(srcs))]
            for rank, src in enumerate(srcs):
                parts.extend([str(src), f"{1.0 / (rank + 1):.6f}"])
            f.write(" ".join(parts) + "\n")


def write_pfm(path, image):
    image = np.asarray(image, dtype=np.float32)
    image = np.where(np.isfinite(image) & (image > 0), image, 0.0).astype(np.float32)
    with path.open("wb") as f:
        f.write(b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        f.write(b"-1.0\n")
        np.flipud(image).tofile(f)


def write_depth_preview(path, depth):
    valid = depth[np.isfinite(depth) & (depth > 1e-8)]
    if not valid.size:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        preview = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        preview = ((1.0 - preview) * 255.0).astype(np.uint8)
        preview[~(np.isfinite(depth) & (depth > 1e-8))] = 0
    cv2.imwrite(str(path), preview)


def get_compositor_tree(scene):
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree

    if getattr(scene, "compositing_node_group", None) is None:
        scene.compositing_node_group = bpy.data.node_groups.new(
            f"{scene.name}_MGF_Compositor",
            "CompositorNodeTree",
        )
    return scene.compositing_node_group


def setup_depth_output(scene, output_dir):
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True
    tree = get_compositor_tree(scene)

    render_layers = tree.nodes.get("MGF_RenderLayers")
    if render_layers is None:
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        render_layers.name = "MGF_RenderLayers"

    file_output = tree.nodes.get("MGF_DepthOutput")
    if file_output is None:
        file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        file_output.name = "MGF_DepthOutput"
    file_output.mute = False
    if hasattr(file_output, "base_path"):
        file_output.base_path = str(output_dir)
        file_output.file_slots[0].path = "depth_"
        depth_input = file_output.inputs[0]
    else:
        file_output.directory = str(output_dir)
        file_output.file_name = "depth_"
        file_output.file_output_items.clear()
        depth_item = file_output.file_output_items.new(socket_type="FLOAT", name="Depth")
        depth_item.override_node_format = True
        try:
            depth_item.format.file_format = "OPEN_EXR"
        except TypeError:
            depth_item.format.file_format = "OPEN_EXR_MULTILAYER"
        depth_item.format.color_depth = "32"
        depth_input = file_output.inputs["Depth"]
    try:
        file_output.format.file_format = "OPEN_EXR"
    except TypeError:
        file_output.format.file_format = "OPEN_EXR_MULTILAYER"
    file_output.format.color_depth = "32"

    for link in list(tree.links):
        if link.to_node == file_output:
            tree.links.remove(link)

    depth_socket = render_layers.outputs.get("Depth") or render_layers.outputs.get("Z")
    if depth_socket is None:
        raise RuntimeError("Render Layers node has no Depth/Z output socket")
    tree.links.new(depth_socket, depth_input)


def mute_other_file_output_nodes(scene, keep_node_name="MGF_DepthOutput"):
    tree = get_compositor_tree(scene)
    states = []
    for node in tree.nodes:
        if getattr(node, "bl_idname", "") != "CompositorNodeOutputFile":
            continue
        if node.name == keep_node_name:
            continue
        states.append((node, node.mute))
        node.mute = True
    return states


def restore_node_mute(states):
    for node, mute in reversed(states):
        node.mute = mute


def find_depth_exr(output_dir):
    exr_files = sorted(output_dir.glob("*.exr"), key=lambda p: p.stat().st_mtime)
    if not exr_files:
        raise FileNotFoundError(f"No depth EXR was written under {output_dir}")
    return exr_files[-1]


def sanitize_depth(depth, camera_obj):
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    depth[depth >= float(camera_obj.data.clip_end) * 0.999] = 0.0
    return depth


def render_depth_pass(scene, camera_obj, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_depth_output(scene, output_dir)
    node_mute_states = mute_other_file_output_nodes(scene)
    try:
        discard_path = output_dir / "_discard_rgb"
        scene.render.filepath = str(discard_path)
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
    finally:
        restore_node_mute(node_mute_states)
    for discard in output_dir.glob("_discard_rgb*"):
        try:
            discard.unlink()
        except OSError:
            pass
    depth_exr_path = find_depth_exr(output_dir)
    depth = sanitize_depth(read_exr(str(depth_exr_path)), camera_obj)
    return depth, depth_exr_path


def render_view(
    scene,
    camera_obj,
    image_path,
    depth_tmp_dir,
    image_format,
    jpeg_quality,
    depth_ignore_objects=None,
    depth_show_objects=None,
    depth_ignore_mode="mask-invalid",
):
    scene.camera = camera_obj
    old_filepath = scene.render.filepath
    old_file_format = scene.render.image_settings.file_format
    old_quality = getattr(scene.render.image_settings, "quality", None)
    old_use_compositing = getattr(scene.render, "use_compositing", None)
    depth_ignore_objects = depth_ignore_objects or []
    depth_show_objects = depth_show_objects or []
    front_depth = None

    try:
        # First render the real RGB without compositor side effects. In Blender
        # 5.1 an Output File-only compositor tree can make write_still save a
        # black final image, so RGB and depth are rendered as two passes.
        if hasattr(scene.render, "use_compositing"):
            scene.render.use_compositing = False
        scene.render.filepath = str(image_path)
        scene.render.image_settings.file_format = image_format
        if image_format == "JPEG":
            scene.render.image_settings.quality = jpeg_quality
        bpy.ops.render.render(write_still=True)

        if depth_ignore_mode == "mask-invalid" and depth_ignore_objects:
            # Render full-scene depth before hiding ignored objects. After the
            # final hidden-object depth pass, pixels where the full scene was
            # closer are exactly the pixels where ignored geometry was the
            # front-most surface. This avoids invalidating mouse pixels merely
            # because a large Bbox wall exists behind the mouse.
            front_depth, _ = render_depth_pass(
                scene,
                camera_obj,
                Path(depth_tmp_dir) / "depth_with_ignore_objects",
            )

        # Then render depth through the compositor file output. Any RGB file
        # written during this pass is intentionally discarded. Depth-ignore
        # objects are hidden so their distances are not learned as supervision.
        depth_ignore_states = []
        depth_show_states = []
        try:
            if depth_ignore_mode in {"mask-invalid", "hide-only"} and depth_ignore_objects:
                depth_ignore_states = hide_objects_for_render(depth_ignore_objects)
            if depth_show_objects:
                depth_show_states = show_objects_for_render(depth_show_objects)
            final_depth_dir = Path(depth_tmp_dir) / "depth_without_ignore_objects" if front_depth is not None else Path(depth_tmp_dir)
            depth, depth_exr_path = render_depth_pass(scene, camera_obj, final_depth_dir)
        finally:
            restore_render_visibility(depth_show_states)
            restore_render_visibility(depth_ignore_states)
    finally:
        scene.render.filepath = old_filepath
        scene.render.image_settings.file_format = old_file_format
        if old_quality is not None:
            scene.render.image_settings.quality = old_quality
        if old_use_compositing is not None:
            scene.render.use_compositing = old_use_compositing

    if front_depth is not None:
        if front_depth.shape != depth.shape:
            front_depth = cv2.resize(
                front_depth,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
        tolerance = np.maximum(1e-4, np.abs(depth) * 0.005)
        ignore_mask = (front_depth > 0) & ((depth <= 0) | (front_depth < depth - tolerance))
        cv2.imwrite(str(Path(depth_tmp_dir) / "depth_ignore_front_mask.png"), ignore_mask.astype(np.uint8) * 255)
        depth[ignore_mask] = 0.0
    return depth, depth_exr_path


def vertex_keypoint_map_candidates():
    blend_dir = Path(bpy.data.filepath).resolve().parent if bpy.data.filepath else Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        blend_dir / DEFAULT_VERTEX_KEYPOINT_MAP_NAME,
        blend_dir / "keypoint_mapping" / DEFAULT_VERTEX_KEYPOINT_MAP_NAME,
        script_dir / DEFAULT_VERTEX_KEYPOINT_MAP_NAME,
        script_dir / "keypoint_mapping" / DEFAULT_VERTEX_KEYPOINT_MAP_NAME,
    ]
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def resolve_vertex_keypoint_map(path_text):
    if not path_text:
        return None
    if path_text.strip().upper() == "AUTO":
        candidates = vertex_keypoint_map_candidates()
        for path in candidates:
            if path.is_file():
                return path
        print(
            "warning: AUTO vertex keypoint map not found; searched: "
            + ", ".join(str(path) for path in candidates)
        )
        return None
    return blender_path(path_text)


def load_vertex_keypoint_map(path_text, mouse_model_name):
    path = resolve_vertex_keypoint_map(path_text)
    if path is None:
        return None, None
    if not path.exists():
        print(f"warning: vertex keypoint map does not exist: {path}; skip keypoint annotations")
        return None, path
    with path.open("rb") as f:
        mapping = pickle.load(f)
    if not isinstance(mapping, dict):
        raise TypeError(f"vertex keypoint map must be a dict, got {type(mapping).__name__}: {path}")
    if not mapping:
        raise ValueError(f"vertex keypoint map is empty: {path}")
    if not all(isinstance(vertex_id, int) and isinstance(name, str) for vertex_id, name in mapping.items()):
        raise TypeError(f"vertex keypoint map must contain int -> str entries: {path}")
    mouse_obj = bpy.data.objects.get(mouse_model_name)
    if mouse_obj is None or mouse_obj.type != "MESH":
        raise KeyError(f"Mouse mesh {mouse_model_name!r} not found while validating keypoint map")
    invalid_ids = [
        vertex_id
        for vertex_id in mapping
        if vertex_id < 0 or vertex_id >= len(mouse_obj.data.vertices)
    ]
    if invalid_ids:
        raise ValueError(
            f"vertex keypoint map is incompatible with {mouse_model_name}: "
            f"{len(invalid_ids)} indices exceed vertex count {len(mouse_obj.data.vertices)}"
        )
    print(
        "loaded vertex keypoint map",
        f"path={path}",
        f"landmarks={len(set(mapping.values()))}",
        f"support_vertices={len(mapping)}",
    )
    return mapping, path


def keypoint_annotations(camera_name, vertex_pixel_dict, vertices_keypoints_dict, armature_name):
    if vertices_keypoints_dict is None:
        return []
    try:
        bone_visibility = check_bone_visibility(vertex_pixel_dict, vertices_keypoints_dict)
    except Exception as exc:
        print(f"warning: failed to compute bone visibility for {camera_name}: {exc}")
        return []
    annotations = []
    for bone_name in left_hand_keypoints + right_hand_keypoints + additional_keypoints:
        visible = bool(bone_visibility.get(bone_name, False))
        item = {"name": bone_name, "visible": visible}
        if visible:
            try:
                pixel_x, pixel_y = get_bone_head_pixel_coordinates(
                    bone_name, camera_name, armature_name=armature_name
                )
                item.update({"x": float(pixel_x), "y": float(pixel_y)})
            except Exception as exc:
                item.update({"visible": False, "error": str(exc)})
        annotations.append(item)
    return annotations


def annotate_visible_keypoints(
    scene,
    frame,
    view_id,
    camera_obj,
    image_path,
    depth_exr_path,
    depth_preview_path,
    args,
    vertices_keypoints_dict,
):
    if vertices_keypoints_dict is None:
        return {"keypoints": []}
    mouse_obj = bpy.data.objects.get(args.mouse_model)
    if mouse_obj is None:
        return {"warning": f"mouse model {args.mouse_model!r} not found", "keypoints": []}

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mouse_obj.evaluated_get(depsgraph)
    temp_mesh = evaluated.to_mesh()
    try:
        width, height = get_render_size(scene)
        vertex_pixel_dict = process_points(
            f"{frame}_{view_id}",
            str(depth_exr_path),
            str(depth_preview_path),
            mouse_obj,
            temp_mesh,
            camera_obj,
            width,
            height,
            scene=scene,
        )
    finally:
        evaluated.to_mesh_clear()

    keypoints = keypoint_annotations(camera_obj.name, vertex_pixel_dict, vertices_keypoints_dict, args.armature)
    if args.save_marked_keypoints and keypoints:
        marked_path = image_path.with_name(image_path.stem + "_keypoints.png")
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for item in keypoints:
            if item.get("visible") and "x" in item and "y" in item:
                x, y = item["x"], item["y"]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
        img.save(marked_path)

    return {
        "visible_vertices": len(vertex_pixel_dict),
        "keypoints": keypoints,
    }


def configure_render_engine(scene, args):
    if args.engine != "KEEP":
        engine = args.engine
        engine_items = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
        if engine not in engine_items and engine == "BLENDER_EEVEE_NEXT" and "BLENDER_EEVEE" in engine_items:
            engine = "BLENDER_EEVEE"
        elif engine not in engine_items and engine == "BLENDER_EEVEE" and "BLENDER_EEVEE_NEXT" in engine_items:
            engine = "BLENDER_EEVEE_NEXT"
        scene.render.engine = engine
    if scene.render.engine == "CYCLES":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        selected_backend = None
        backend_errors = {}
        for backend in ("OPTIX", "CUDA"):
            try:
                preferences.compute_device_type = backend
                preferences.get_devices()
                selected_backend = backend
                break
            except Exception as error:
                backend_errors[backend] = str(error)
        if selected_backend is None:
            raise RuntimeError(
                f"Could not initialize an NVIDIA Cycles backend: {backend_errors}"
            )

        backend_devices = [
            device for device in preferences.devices if device.type == selected_backend
        ]
        preferred_devices = [
            device for device in backend_devices if "RTX PRO 6000" in device.name.upper()
        ]
        selected_devices = preferred_devices or backend_devices
        for device in preferences.devices:
            device.use = device in selected_devices
        if not selected_devices:
            available = [
                {"name": device.name, "type": device.type}
                for device in preferences.devices
            ]
            raise RuntimeError(
                f"No {selected_backend} Cycles device available: {available}"
            )
        scene.cycles.device = "GPU"
        scene.cycles.samples = args.cycles_samples
        scene.cycles.use_denoising = True
        print(
            "cycles_device "
            + json.dumps(
                {
                    "backend": selected_backend,
                    "devices": [device.name for device in selected_devices],
                }
            ),
            flush=True,
        )


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def expected_frame_outputs(frame_root, camera_count, image_ext, save_depth_preview, metadata_only):
    expected = [
        frame_root / "scene_meta.json",
        frame_root / "cams" / "pair.txt",
    ]
    for view_id in range(camera_count):
        view_stem = f"{view_id:08d}"
        expected.extend(
            [
                frame_root / "cams" / f"{view_stem}_cam.txt",
                frame_root / "annotations" / f"{view_stem}.json",
            ]
        )
        if not metadata_only:
            expected.extend(
                [
                    frame_root / "blended_images" / f"{view_stem}{image_ext}",
                    frame_root / "blended_images" / f"{view_stem}_masked{image_ext}",
                    frame_root / "rendered_depth_maps" / f"{view_stem}.pfm",
                ]
            )
            if save_depth_preview:
                expected.append(frame_root / "rendered_depth_maps" / f"{view_stem}_preview.png")
    return expected


def is_frame_complete(frame_root, camera_count, image_ext, save_depth_preview, metadata_only):
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_frame_outputs(
        frame_root,
        camera_count,
        image_ext,
        save_depth_preview,
        metadata_only,
    ))


def completed_scene_folders(
    output_root,
    start_frame,
    end_frame,
    frame_step,
    camera_count,
    image_ext,
    save_depth_preview,
    metadata_only,
):
    folders = []
    for frame in range(start_frame, end_frame + 1, frame_step):
        frame_root = output_root / f"frame_{frame:06d}"
        if is_frame_complete(frame_root, camera_count, image_ext, save_depth_preview, metadata_only):
            folders.append(frame_root.name)
    return folders


def refresh_export_metadata(
    metadata,
    output_root,
    start_frame,
    end_frame,
    frame_step,
    camera_count,
    image_ext,
    save_depth_preview,
    metadata_only,
):
    metadata["scene_folders"] = completed_scene_folders(
        output_root,
        start_frame,
        end_frame,
        frame_step,
        camera_count,
        image_ext,
        save_depth_preview,
        metadata_only,
    )
    write_json(output_root / "metadata.json", metadata)


def export_blendedmvs_water_reaching(args):
    scene = bpy.context.scene
    start_time = time.time()
    output_root = blender_path(args.output_root)
    if args.resolution_x is not None:
        scene.render.resolution_x = args.resolution_x
    if args.resolution_y is not None:
        scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    camera_names = resolve_camera_names(args.cameras)
    camera_model_collections = parse_name_list(args.camera_model_collections)
    depth_ignore_collection_names = []
    depth_ignore_objects = []
    found_depth_ignore_collections = []
    if args.depth_ignore_mode != "none":
        depth_ignore_collection_names = parse_name_list(args.depth_ignore_collections)
        depth_ignore_objects, found_depth_ignore_collections = objects_from_collection_names(
            depth_ignore_collection_names,
            purpose="depth-ignore collection",
        )
    depth_excluded_hair_objects = []
    if args.depth_exclude_hair_curves:
        depth_excluded_hair_objects = [
            obj for obj in bpy.data.objects if obj.type == "CURVES" and not obj.hide_render
        ]
        depth_ignore_objects.extend(depth_excluded_hair_objects)
    depth_ignore_objects = list({obj.name: obj for obj in depth_ignore_objects}.values())
    per_camera_hide_collections = parse_camera_collection_map(
        args.per_camera_hide_collections
    )
    per_camera_depth_ignore_collections = parse_camera_collection_map(
        args.per_camera_depth_ignore_collections
    )
    per_camera_depth_show_collections = parse_camera_collection_map(
        args.per_camera_depth_show_collections
    )

    def resolve_per_camera_depth_objects(mapping, purpose):
        resolved = {}
        for camera_name, collection_name in mapping.items():
            objects, found = objects_from_collection_names([collection_name], purpose=purpose)
            if not found:
                raise KeyError(
                    f"{purpose.capitalize()} {collection_name!r} not found for {camera_name}"
                )
            resolved[camera_name] = objects
        return resolved

    per_camera_depth_ignore_objects = resolve_per_camera_depth_objects(
        per_camera_depth_ignore_collections,
        "per-camera depth-ignore collection",
    )
    per_camera_depth_show_objects = resolve_per_camera_depth_objects(
        per_camera_depth_show_collections,
        "per-camera depth-show collection",
    )
    start_frame = args.start_frame if args.start_frame is not None else scene.frame_start
    end_frame = args.end_frame if args.end_frame is not None else scene.frame_end
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be at least 1, got {args.frame_step}")
    if end_frame < start_frame:
        raise ValueError(
            f"--end-frame ({end_frame}) must be greater than or equal to "
            f"--start-frame ({start_frame})"
        )
    selected_frames = list(range(start_frame, end_frame + 1, args.frame_step))
    if getattr(args, "frames", ""):
        requested_frames = sorted({int(value) for value in args.frames.split(",") if value.strip()})
        if not requested_frames or not set(requested_frames).issubset(selected_frames):
            raise ValueError("--frames must be a nonempty subset of the requested frame range/step")
        selected_frames = requested_frames
    sampled_frame_count = len(selected_frames)
    image_ext = ".jpg" if args.image_format == "JPEG" else ".png"

    cameras = []
    for camera_name in camera_names:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None:
            raise KeyError(f"Camera object {camera_name!r} not found")
        if camera_obj.type != "CAMERA":
            raise TypeError(f"Object {camera_name!r} is not a CAMERA")
        cameras.append(camera_obj)

    ensure_clean_dir(output_root, overwrite=args.overwrite)
    vertices_keypoints_dict, vertex_keypoint_map_path = load_vertex_keypoint_map(
        args.vertex_keypoint_map,
        args.mouse_model,
    )
    configure_render_engine(scene, args)
    render_visibility_states = hide_camera_model_collections(
        camera_model_collections,
        hide_prefix_fallback=args.hide_physical_camera_prefixes,
    )

    metadata = {
        "format": "water_reaching_blendedmvs_like",
        "frames": [start_frame, end_frame],
        "frame_step": args.frame_step,
        "sampled_frame_count": sampled_frame_count,
        "cameras_requested": args.cameras,
        "cameras": camera_names,
        "camera_model_collections_hidden_for_render": camera_model_collections,
        "physical_camera_object_prefixes_hidden_for_render": list(
            PHYSICAL_CAMERA_OBJECT_PREFIXES
        ) if args.hide_physical_camera_prefixes else [],
        "depth_ignore_mode": args.depth_ignore_mode,
        "depth_ignore_collections_requested": depth_ignore_collection_names,
        "depth_ignore_collections_found": found_depth_ignore_collections,
        "depth_ignore_object_count": len(depth_ignore_objects),
        "depth_ignore_objects": [obj.name for obj in depth_ignore_objects],
        "depth_exclude_hair_curves": args.depth_exclude_hair_curves,
        "depth_excluded_hair_objects": [obj.name for obj in depth_excluded_hair_objects],
        "per_camera_hide_collections": per_camera_hide_collections,
        "per_camera_depth_ignore_collections": per_camera_depth_ignore_collections,
        "per_camera_depth_show_collections": per_camera_depth_show_collections,
        "render_resolution": list(get_render_size(scene)),
        "image_format": args.image_format,
        "scene_unit_system": scene.unit_settings.system,
        "scene_length_unit": scene.unit_settings.length_unit,
        "blender_unit_scale_meters": float(scene.unit_settings.scale_length),
        "depth_units": (
            "millimeter"
            if scene.unit_settings.system == "METRIC"
            and abs(float(scene.unit_settings.scale_length) - 0.001) < 1e-8
            else "Blender world units"
        ),
        "camera_extrinsics": "OpenCV world-to-camera rigid transform; inherited object scale removed",
        "masked_image_stage": "RGB placeholder copy; replaced by Core23 augmentation",
        "vertex_keypoint_map_requested": args.vertex_keypoint_map,
        "vertex_keypoint_map_resolved": (
            str(vertex_keypoint_map_path) if vertex_keypoint_map_path is not None else None
        ),
        "keypoint_annotation_enabled": vertices_keypoints_dict is not None,
        "keypoint_count": (
            len(set(vertices_keypoints_dict.values()))
            if vertices_keypoints_dict is not None
            else 0
        ),
        "scene_folders": [],
    }
    refresh_export_metadata(
        metadata,
        output_root,
        start_frame,
        end_frame,
        args.frame_step,
        len(cameras),
        image_ext,
        args.save_depth_preview,
        args.metadata_only,
    )

    try:
        for frame in selected_frames:
            frame_root = output_root / f"frame_{frame:06d}"
            if args.resume and is_frame_complete(
                frame_root,
                len(cameras),
                image_ext,
                args.save_depth_preview,
                args.metadata_only,
            ):
                print(f"skip complete frame={frame}")
                continue

            if frame_root.exists():
                print(f"resume from incomplete frame={frame}; clean partial folder {frame_root}")
                ensure_clean_dir(frame_root, overwrite=True)

            scene.frame_set(frame)
            image_dir = frame_root / "blended_images"
            depth_dir = frame_root / "rendered_depth_maps"
            cams_dir = frame_root / "cams"
            ann_dir = frame_root / "annotations"
            tmp_depth_dir = frame_root / "_depth_exr_tmp"
            for path in [image_dir, depth_dir, cams_dir, ann_dir, tmp_depth_dir]:
                path.mkdir(parents=True, exist_ok=True)

            frame_meta = {
                "frame": frame,
                "views": [],
            }
            for view_id, camera_obj in enumerate(cameras):
                view_stem = f"{view_id:08d}"
                image_path = image_dir / f"{view_stem}{image_ext}"
                masked_image_path = image_dir / f"{view_stem}_masked{image_ext}"
                depth_pfm_path = depth_dir / f"{view_stem}.pfm"
                depth_preview_path = depth_dir / f"{view_stem}_preview.png"
                cam_path = cams_dir / f"{view_stem}_cam.txt"
                view_tmp_dir = tmp_depth_dir / view_stem
                ensure_clean_dir(view_tmp_dir, overwrite=True)

                view_collection_name = per_camera_hide_collections.get(camera_obj.name)
                view_collection = (
                    bpy.data.collections.get(view_collection_name)
                    if view_collection_name
                    else None
                )
                if view_collection_name and view_collection is None:
                    raise KeyError(
                        f"Per-camera hide collection {view_collection_name!r} not found "
                        f"for {camera_obj.name}"
                    )
                old_view_collection_hide_render = (
                    view_collection.hide_render if view_collection is not None else None
                )
                if view_collection is not None:
                    view_collection.hide_render = True
                try:
                    if args.metadata_only:
                        depth = np.zeros((get_render_size(scene)[1], get_render_size(scene)[0]), dtype=np.float32)
                        depth_exr_path = view_tmp_dir / "metadata_only_depth.exr"
                    else:
                        depth, depth_exr_path = render_view(
                            scene,
                            camera_obj,
                            image_path,
                            view_tmp_dir,
                            args.image_format,
                            args.jpeg_quality,
                            depth_ignore_objects=list({
                                obj.name: obj
                                for obj in (
                                    depth_ignore_objects
                                    + per_camera_depth_ignore_objects.get(camera_obj.name, [])
                                )
                            }.values()),
                            depth_show_objects=per_camera_depth_show_objects.get(
                                camera_obj.name, []
                            ),
                            depth_ignore_mode=args.depth_ignore_mode,
                        )
                        shutil.copyfile(image_path, masked_image_path)
                        write_pfm(depth_pfm_path, depth)
                        if args.save_depth_preview:
                            write_depth_preview(depth_preview_path, depth)
                finally:
                    if view_collection is not None:
                        view_collection.hide_render = old_view_collection_hide_render

                intrinsic = camera_intrinsic_matrix(scene, camera_obj)
                extrinsic = camera_extrinsic_matrix(camera_obj)
                write_cam_file(cam_path, extrinsic, intrinsic, depth, args.depth_num)

                annotation = {
                    "frame": frame,
                    "view_id": view_id,
                    "camera": camera_obj.name,
                    "image": str(image_path.relative_to(frame_root)),
                    "masked_image": str(masked_image_path.relative_to(frame_root)),
                    "depth_pfm": str(depth_pfm_path.relative_to(frame_root)),
                    "cam": str(cam_path.relative_to(frame_root)),
                    "intrinsic": intrinsic.tolist(),
                    "extrinsic": extrinsic.tolist(),
                    "depth_ignore_mode": args.depth_ignore_mode,
                    "depth_ignore_collections": found_depth_ignore_collections,
                    "per_camera_hidden_collection": view_collection_name,
                    "valid_depth_pixel_count": int(np.count_nonzero(np.isfinite(depth) & (depth > 1e-8))),
                }
                if not args.metadata_only:
                    annotation.update(
                        annotate_visible_keypoints(
                            scene,
                            frame,
                            view_id,
                            camera_obj,
                            image_path,
                            depth_exr_path,
                            depth_preview_path,
                            args,
                            vertices_keypoints_dict,
                        )
                    )
                write_json(ann_dir / f"{view_stem}.json", annotation)
                frame_meta["views"].append(annotation)
                print(f"exported frame={frame} view={view_id} camera={camera_obj.name}")

            write_pair_file(cams_dir / "pair.txt", len(cameras))
            write_json(frame_root / "scene_meta.json", frame_meta)
            if tmp_depth_dir.exists() and not args.keep_depth_exr:
                shutil.rmtree(tmp_depth_dir)
            refresh_export_metadata(
                metadata,
                output_root,
                start_frame,
                end_frame,
                args.frame_step,
                len(cameras),
                image_ext,
                args.save_depth_preview,
                args.metadata_only,
            )
    finally:
        restore_render_visibility(render_visibility_states)

    refresh_export_metadata(
        metadata,
        output_root,
        start_frame,
        end_frame,
        args.frame_step,
        len(cameras),
        image_ext,
        args.save_depth_preview,
        args.metadata_only,
    )
    elapsed = time.time() - start_time
    print(f"BlendedMVS-style water-reaching export finished: {output_root}")
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    export_blendedmvs_water_reaching(parse_blender_args())


