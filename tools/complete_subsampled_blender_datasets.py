"""Resumable missing-frame completion queue (run using WSL Python).

Original payload hashes and the requested source frames are frozen in plan.json.
Windows Blender renders into output/, never overwriting production frame folders.
Each chunk receives rig annotations and masks, is validated, and is then moved in.
Style-transfer completion runs only after all base render chunks are published.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image
from core23_temporal_format import write_json_atomic

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'dataset/BlenderPyMouse'
OUT = BASE / 'output/temporal_completion'
BLENDER = Path('/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe')
PYTHON = BASE / '.venv-vggt-omega-eval/bin/python'
WHITE = 'PyMouseStudio-WhiteFur-WaterReaching-FrameToFrame-FGcamB'
TRIALS = ['E_L2_2024-11-18_1_Trial_1', 'FB_R3_2024-03-16_1_Trial_5',
          'FJ_L2_2024-06-24_1_Trial_7', 'FS_L2_2024-06-25_1_Trial_2']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def win(path):
    return str(PureWindowsPath('\\\\wsl.localhost\\Ubuntu') / path.resolve().relative_to('/').as_posix())


def project_relative(path):
    if path.startswith('\\\\wsl.'):
        parts = PureWindowsPath(path).parts
        start = parts.index('MouseGeoForge') + 1
        return Path(*parts[start:]).as_posix()
    return Path(path).resolve().relative_to(BASE).as_posix()


def status(**kwargs):
    old = read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    old.update(kwargs, updated_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    write_json_atomic(OUT/'status.json', old)
    print(json.dumps(kwargs), flush=True)


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', buffering=1) as stream:
        stream.write('\nCOMMAND '+json.dumps(list(map(str, command)))+'\n')
        subprocess.run(list(map(str, command)), cwd=BASE, stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    planpath = OUT/'plan.json'
    if planpath.exists():
        return read(planpath)
    mapping = read(BASE/'configs/blenderpymouse_production_projects.json')
    by_name = {name: item['destination'] for item in mapping['projects'] for name in item['datasets']}
    jobs = []
    # PseudoSynthetic is a pose set, not reliable temporal data (user confirmed).
    for name in [WHITE]:
        root = DATA/name
        meta = read(root/'metadata.json')
        inspection = read(OUT/f'{name}_source_inspection.json')
        assert all(c['world_max_error'] < 1e-6 for c in inspection['checks'])
        assert all(c['K_error'] < 1e-5 and c['E_error'] < 1e-4
                   for item in inspection['checks'] for c in item['cameras'])
        frames = sorted(int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir())
        end = 392 if name == WHITE else 405
        if name == WHITE:
            assert max(r[1] for r in inspection['action_frame_ranges'].values()) == 392
        assert frames[0] == 1 and max(frames) <= end
        missing = sorted(set(range(1, end+1)) - set(frames))
        kp = meta['keypoint_annotations']
        job = dict(dataset=name, blend='BlenderPyMouseProjects/'+by_name[name],
            cameras=meta['cameras'], resolution=meta['render_resolution'],
            depth_ignore_mode=meta.get('depth_ignore_mode', 'mask-invalid'),
            depth_ignore_collections=meta.get('depth_ignore_collections_requested', ['Bbox']),
            keypoint_map=project_relative(kp['vertex_keypoint_map']),
            keypoint_audit=project_relative(kp['vertex_keypoint_audit']),
            mouse_collection=inspection['mouse_collections'][0],
            original_frames=frames, missing_frames=missing, end_frame=end,
            source_fps=inspection['fps']/inspection['fps_base'],
            transferred=None if name == WHITE else name+'Transfered')
        for field in ('blend', 'keypoint_map', 'keypoint_audit'):
            assert (BASE/job[field]).is_file(), job[field]
        for dataset_name in [name] + ([job['transferred']] if job['transferred'] else []):
            target = DATA/dataset_name
            archive = OUT/'original'/dataset_name
            archive.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for f in frames:
                scene = target/f'frame_{f:06d}'
                validate_scene(scene, job, require_mask=False)
                for p in sorted(scene.rglob('*')):
                    if p.is_file():
                        hashes[p.relative_to(target).as_posix()] = digest(p)
            for name_ in ('metadata.json', 'temporal_index.jsonl', 'style_transfer_status.json'):
                if (target/name_).exists():
                    shutil.copy2(target/name_, archive/name_)
            write_json_atomic(archive/'original_payload_sha256.json', hashes)
        jobs.append(job)
        print('Prepared', name, len(missing), 'missing frames', flush=True)
    plan = {'version':1, 'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Fill source animation frames, do not densify selected-pose Sample10/DiversePose datasets.',
        'jobs':jobs, 'original_frames_preserved':True}
    write_json_atomic(planpath, plan)
    return plan


def pfm(path):
    with path.open('rb') as handle:
        assert handle.readline().strip() == b'Pf', path
        width, height = map(int, handle.readline().split())
        scale = float(handle.readline())
        values = np.fromfile(handle, dtype='<f4' if scale < 0 else '>f4')
        assert values.size == width*height, path
        return values.reshape(height, width)


def validate_scene(scene, job, require_mask=True):
    meta = read(scene/'scene_meta.json')
    assert meta['frame'] == int(scene.name[6:]), scene
    assert [v['camera'] for v in meta['views']] == job['cameras'], scene
    assert (scene/'cams/pair.txt').is_file()
    with np.load(scene/'keypoints.npz') as arrays:
        assert len(arrays['names']) == 29, scene
        assert np.isfinite(arrays['joint_world_xyz']).all(), scene
        assert arrays['view_ids'].size == len(job['cameras']), scene
    assert read(scene/'keypoints.json')['frame'] == int(scene.name[6:]), scene
    for vid in range(len(job['cameras'])):
        stem = f'{vid:08d}'
        for suffix in ('', '_masked'):
            with Image.open(scene/f'blended_images/{stem}{suffix}.jpg') as image:
                assert list(image.size) == job['resolution'], scene
                image.load()
        depth = pfm(scene/f'rendered_depth_maps/{stem}.pfm')
        assert list(depth.shape[::-1]) == job['resolution'], scene
        assert np.isfinite(depth).all() and (depth >= 0).all(), scene
        assert (scene/f'cams/{stem}_cam.txt').is_file()
        assert read(scene/f'annotations/{stem}.json')['camera'] == job['cameras'][vid]
        if require_mask:
            with Image.open(scene/f'body_masks/{stem}.png') as image:
                assert list(image.size) == job['resolution'], scene
                assert set(np.unique(np.asarray(image))) <= {0, 255}, scene


def verify_original(name):
    root = DATA/name
    hashes = read(OUT/'original'/name/'original_payload_sha256.json')
    for relative, expected in hashes.items():
        assert digest(root/relative) == expected, f'Original payload changed: {name}/{relative}'
    return len(hashes)


def validate_loader(scenes):
    """Decode newly produced frames, with explicit mm/raw scaling and PFM orientation."""
    sys.path.insert(0, str(BASE/'vggt-omega'))
    import torch
    from training.finetune_blendedmvs import BlendedMVSDataset, load_scene
    torch.set_num_threads(2)
    loaded = [load_scene(path) for path in scenes]
    views = min(2, min(len(scene.frames) for scene in loaded))
    dataset = BlendedMVSDataset(scenes=loaded, num_views=views, image_width=128, image_height=128,
        samples_per_scene=1, seed=5706, use_masked_images=False, pfm_flip_y=True,
        canonicalize_camera=False, scale_mode='raw', top_srcs=views, load_keypoints=True)
    dataset.max_instances = 2
    for index in range(len(loaded)):
        sample = dataset[index]
        assert sample['actual_instance_count'] == 1
        assert not sample['instance_valid'][:, 1].any()
        assert sample['instance_keypoint_valid'][:, 0].any()
        assert torch.isfinite(sample['images']).all()
    print('NEW_FRAME_LOADER_VALIDATED', [p.name for p in scenes], flush=True)


def refresh_metadata(job, name=None, complete=False):
    name = name or job['dataset']
    root = DATA/name
    meta = read(root/'metadata.json')
    frames = sorted(int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir())
    meta.update(dataset_name=name, dataset_root=str(root), frames=[frames[0],frames[-1]],
        frame_numbers=frames, scene_folders=[f'frame_{f:06d}' for f in frames],
        scene_count=len(frames), sampled_frame_count=len(frames), frame_step=1,
        frame_numbering='one-based Blender source frame number; dense 1..N on completion',
        incomplete_scene_count=0, incomplete_scenes=[], view_counts={str(len(job['cameras'])):len(frames)})
    visible = 0
    for frame in frames:
        with np.load(root/f'frame_{frame:06d}/keypoints.npz') as arrays:
            visible += int(arrays['surface_visible'].sum())
    kp = meta['keypoint_annotations']
    kp.update(selected_scene_count=len(frames), complete_scene_count=len(frames),
        complete_view_count=len(frames)*len(job['cameras']), annotated_scene_count=len(frames),
        annotated_view_count=len(frames)*len(job['cameras']), skipped_scene_count=0,
        visible_keypoint_instances=visible, complete_visible_keypoint_instances=visible)
    meta['temporal_completion'] = dict(export_complete=complete, expected_frame_count=job['end_frame'],
        available_frame_count=len(frames), original_frame_count=len(job['original_frames']),
        new_frame_count=len(set(frames)-set(job['original_frames'])), source_frame_step=1,
        source_animation_fps=job['source_fps'], original_payloads_preserved=True,
        provenance='output/temporal_completion/plan.json',
        note='Blender animation timing; not a newly verified real-video acquisition clock.')
    # Do not advertise a complete animation until every requested source frame is present.
    meta['temporal']['frame_count'] = len(frames)
    meta['temporal']['export_complete'] = complete
    mask_count = sum(len(list((root/f'frame_{f:06d}/body_masks').glob('*.png'))) for f in frames)
    meta['body_mask_annotations'] = dict(format='mousegeoforge_blender_visible_body_mask_v1',
        include_hair=False, mask_view_count=mask_count, expected_view_count=len(frames)*len(job['cameras']),
        complete=mask_count == len(frames)*len(job['cameras']), output_layout='frame_xxxxxx/body_masks/00000000.png',
        occlusion_policy='non-mouse scene geometry remains as black foreground occluders')
    meta.setdefault('layout', {})['body_masks'] = 'frame_xxxxxx/body_masks/00000000.png'
    write_json_atomic(root/'metadata.json', meta)


def run_chunk(job, frames):
    key = f'{min(frames):06d}_{max(frames):06d}'
    chunk = {**job, 'frames':frames,
        'stage':f'output/temporal_completion/staging/{job["dataset"]}/{key}'}
    path = OUT/'jobs'/f'{job["dataset"]}_{key}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, chunk)
    log = OUT/'logs'/f'{job["dataset"]}_{key}.log'
    status(phase='rendering_and_annotating', dataset=job['dataset'], chunk_frames=frames,
           active_log=str(log.relative_to(BASE)), complete=False)
    run([BLENDER, '--background', win(BASE/job['blend']), '--python-exit-code', '1',
        '--python', win(BASE/'BlenderScripts/complete_temporal_render_chunk.py'), '--', win(path)], log)
    stage = BASE/chunk['stage']
    assert read(stage/'chunk_complete.json')['frames'] == frames
    # Validate all chunk payloads before publishing any of them.
    for fid in frames:
        validate_scene(stage/f'frame_{fid:06d}', job)
    validate_loader([stage/f'frame_{fid:06d}' for fid in sorted({frames[0],frames[-1]})])
    for fid in frames:
        source = stage/f'frame_{fid:06d}'
        target = DATA/job['dataset']/source.name
        assert source.resolve().is_relative_to((OUT/'staging').resolve())
        assert target.resolve().parent == (DATA/job['dataset']).resolve()
        assert not target.exists(), target
        source.rename(target)
    refresh_metadata(job)
    status(phase='chunk_published', dataset=job['dataset'], published_chunk_frames=frames)


def finalize_base(job):
    root = DATA/job['dataset']
    # Old scenes had no body masks. Add those too, without changing their RGB/depth/NPZ.
    status(phase='backfilling_original_body_masks', dataset=job['dataset'],
           active_log=f'output/temporal_completion/logs/{job["dataset"]}_body_masks.log')
    run([BLENDER, '--background', win(BASE/job['blend']), '--python-exit-code', '1',
         '--python', win(BASE/'BlenderScripts/export_existing_blenderpymouse_body_masks.py'), '--',
         '--dataset-root', win(root), '--mouse-collection', job['mouse_collection']],
        OUT/'logs'/f'{job["dataset"]}_body_masks.log')
    for fid in range(1, job['end_frame']+1):
        validate_scene(root/f'frame_{fid:06d}', job)
    verified = verify_original(job['dataset'])
    refresh_metadata(job, complete=True)
    write_json_atomic(OUT/f'{job["dataset"]}_validation.json',
        {'passed':True, 'frames':job['end_frame'], 'original_files_hash_verified':verified})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--max-chunks', type=int, default=0, help='Bounded integration test; no finalization')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--wait-for-blender-pid', type=int, default=0,
                        help='Adopt the current staging batch after a scheduler-only restart.')
    args = parser.parse_args()
    assert args.chunk_size > 0
    if args.wait_for_blender_pid:
        # The old scheduler is stopped, not the Blender renderer. Let that exact
        # process finish before a resumed worker can access the same staging path.
        status(phase='waiting_for_retained_blender_batch', complete=False,
               retained_blender_pid=args.wait_for_blender_pid)
        subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-Command',
            f'Wait-Process -Id {args.wait_for_blender_pid} -ErrorAction SilentlyContinue'])
    # fcntl prevents concurrent queues from publishing the same missing scenes.
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = prepare()
            if any('PseudoSynthetic' in j['dataset'] for j in plan['jobs']):
                raise ValueError('PseudoSynthetic is not reliable temporal data; remove it from the active plan')
            if args.prepare_only:
                return
            if not args.dataset and not args.max_chunks:
                status(phase='queue_started', complete=False, error=None,
                       started_at_utc=datetime.now(timezone.utc).isoformat())
                for item in plan['jobs']:
                    for name in [item['dataset']] + ([item['transferred']] if item['transferred'] else []):
                        previous = read(DATA/name/'metadata.json').get('temporal_completion', {})
                        refresh_metadata(item, name, complete=previous.get('export_complete',False))
                run([PYTHON, BASE/'tools/organize_temporal_datasets.py'], OUT/'logs/indices.log')
            count = 0
            for job in plan['jobs']:
                if args.dataset and args.dataset != job['dataset']:
                    continue
                root = DATA/job['dataset']
                pending = [f for f in job['missing_frames'] if not (root/f'frame_{f:06d}').exists()]
                for offset in range(0,len(pending),args.chunk_size):
                    run_chunk(job, pending[offset:offset+args.chunk_size])
                    count += 1
                    if args.max_chunks and count >= args.max_chunks:
                        status(phase='bounded_test_finished', complete=False)
                        return
                finalize_base(job)
                run([PYTHON, BASE/'tools/organize_temporal_datasets.py'], OUT/'logs/indices.log')
            if not args.dataset:
                if any(j['transferred'] for j in plan['jobs']):
                    status(phase='style_transfer', complete=False,
                           active_log='output/temporal_completion/logs/style_transfer.log')
                    run([BASE/'img2img-turbo/.venv-mousegeoforge/bin/python',
                         BASE/'tools/complete_subsampled_style_transfer.py'], OUT/'logs/style_transfer.log')
                run([PYTHON, BASE/'tools/organize_temporal_datasets.py'], OUT/'logs/indices.log')
                run([PYTHON, BASE/'tools/validate_blender_temporal_indices.py'], OUT/'logs/index_validation.log')
                run([PYTHON, BASE/'tools/audit_blenderpymouse_multi_instance_loader.py',
                     '--report', OUT/'multi_instance_loader_validation.json'], OUT/'logs/loader_validation.log')
                status(phase='complete', complete=True)
        except Exception:
            status(phase='failed', complete=False, error=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
