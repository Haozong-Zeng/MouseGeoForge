"""Render/annotate a bounded staging chunk, never a production dataset in place.

Called by tools/complete_subsampled_blender_datasets.py using a frozen JSON job.
The .blend is reopened between RGB, rig annotation and silhouette rendering;
hair/material visibility changes therefore cannot leak into RGB rendering.
"""
from pathlib import Path
import json
import sys
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_water_reaching_rendering as renderer
import annotate_existing_blenderpymouse_keypoints as keypoints
import export_existing_blenderpymouse_body_masks as masks


def main():
    job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
    root = HERE.parent
    staging = (root / job['stage']).resolve()
    if not staging.is_relative_to((root / 'output/temporal_completion/staging').resolve()):
        raise ValueError(f'Not an owned staging path: {staging}')
    source = (root / job['blend']).resolve()
    if Path(bpy.data.filepath).resolve() != source:
        raise ValueError('Wrong source Blender project')
    frames = job['frames']
    width, height = job['resolution']
    sys.argv = ['blender', '--', '--output-root', str(staging),
        '--cameras', ','.join(job['cameras']), '--start-frame', str(min(frames)),
        '--end-frame', str(max(frames)), '--frames', ','.join(map(str, frames)),
        '--resolution-x', str(width), '--resolution-y', str(height),
        '--cycles-samples', '64', '--vertex-keypoint-map=',
        '--depth-ignore-mode', job['depth_ignore_mode'],
        '--depth-ignore-collections', ','.join(job['depth_ignore_collections']), '--resume']
    renderer.export_blendedmvs_water_reaching(renderer.parse_blender_args())
    bpy.ops.wm.open_mainfile(filepath=str(source))
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    sys.argv = ['blender', '--', '--dataset-root', str(staging),
        '--vertex-keypoint-map', str(root / job['keypoint_map']),
        '--vertex-keypoint-audit', str(root / job['keypoint_audit']), '--progress-every', '1']
    keypoints.main()
    bpy.ops.wm.open_mainfile(filepath=str(source))
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    sys.argv = ['blender', '--', '--dataset-root', str(staging),
        '--mouse-collection', job['mouse_collection'], '--progress-every', '1']
    masks.main()
    (staging / 'chunk_complete.json').write_text(json.dumps({
        'frames': frames, 'source_blend': str(source), 'complete': True}, indent=2))
    print('TEMPORAL_CHUNK_COMPLETE', staging, flush=True)


if __name__ == '__main__':
    main()
