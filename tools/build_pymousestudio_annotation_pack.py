"""Export prediction-free raw frames and auditable synthetic Core23 references.

No training, source mutation, lens correction or holdout model inspection.
Uses a predeclared task list; rejects overwrites. Sequential video decoding
avoids ambiguous keyframe seeking. All outputs are generated under output/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'output/annotation_packages/PyMouseStudioRaw-Core23-AnnotationPack-v1'
PRIMARY = {0, 7, 13, 14, 15, 16, 17}
CHINESE = ['鼻尖', '头部内部中心（暂缓）', '左耳骨起点（暂缓）', '右耳骨起点（暂缓）',
           '颈部内部中心（暂缓）', '脊柱内部中心（暂缓）', '腰椎内部中心（暂缓）', '尾根',
           '尾中间骨点1（暂缓）', '尾中间骨点2（暂缓）', '尾中间骨点3（暂缓）',
           '尾中间骨点4（暂缓）', '尾中间骨点5（暂缓）', '尾尖', '左前爪掌/指根中心',
           '右前爪掌/指根中心', '左后爪跖/趾根中心', '右后爪跖/趾根中心',
           '左elbow名/Wrist骨（暂缓）', '右elbow名/Wrist骨（暂缓）',
           '左膝骨点（暂缓）', '右膝骨点（暂缓）', '重复脊柱中心（暂缓）']
SOURCE31 = ['ear_end_l', 'ear_end_r', 'eye_l', 'eye_r', 'nose',
            'fore_paw_digit_2b_l', 'fore_paw_l', 'ulna_l', 'humerus_l',
            'fore_paw_digit_2b_r', 'fore_paw_r', 'ulna_r', 'humerus_r',
            'hind_paw_digit_2b_l', 'hind_paw_l', 'tibia_l',
            'hind_paw_digit_2b_r', 'hind_paw_r', 'tibia_r',
            'Spine0', 'Spine1', 'Spine2', 'Spine3', 'Spine4',
            'tail_root', 'tail_0', 'tail_1', 'tail_2', 'tail_3', 'tail_4', 'tail_end']
MODEL31 = ['EarEnd_L', 'EarEnd_R', 'Eye_L', 'Eye_R', 'Nose_M',
           'NotUsed', 'ForePaw_L', 'Ulna_L', 'NotUsed', 'NotUsed', 'ForePaw_R', 'Ulna_R',
           'NotUsed', 'HindPaw_L', 'Ankle_L', 'NotUsed', 'HindPaw_R', 'Ankle_R', 'NotUsed',
           'Spine0', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'TailRoot_M',
           'Tail_0', 'Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'TailEnd_M']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def synthetic_examples(out):
    dest = out / 'synthetic_examples'
    dest.mkdir()
    sources = []
    canonical = None
    for action in ('WalkAround', 'WashFace', 'ClimbWall'):
        dataset = f'PyMouseStudio-BrownFur-{action}-Core23'
        folder = ROOT / 'dataset/BlenderPyMouse' / dataset / 'frame_000031'
        kpfile = folder / 'keypoints.json'
        kp = json.loads(kpfile.read_text())
        if canonical is None:
            canonical = kp
        assert kp['names'] == canonical['names']
        assert np.array_equal(kp['joint_world_xyz'][5], kp['joint_world_xyz'][22])
        for view_id in ([0, 6, 7] if action == 'WalkAround' else [0, 6]):
            role = {0: 'side', 6: 'bottom', 7: 'top'}[view_id]
            stem = f'{action}_frame000031_view{view_id:02d}_{role}'
            src = folder / 'blended_images' / f'{view_id:08d}.jpg'
            raw = np.array(Image.open(src).convert('RGB'))
            Image.fromarray(raw).save(dest / f'{stem}_clean.png')
            view = next(v for v in kp['views'] if v['view_id'] == view_id)
            points = view['keypoints']
            uv = np.asarray([p['uv'] for p in points], float)
            keep = np.isfinite(uv).all(1)
            h, w = raw.shape[:2]
            x0, y0 = np.maximum(0, np.floor(uv[keep].min(0) - 42)).astype(int)
            x1, y1 = np.minimum([w, h], np.ceil(uv[keep].max(0) + 42)).astype(int)
            fig = plt.figure(figsize=(18, 8), layout='constrained')
            grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1.05])
            a, b, legend = [fig.add_subplot(grid[0, i]) for i in range(3)]
            a.imshow(raw); a.set_title('Original synthetic RGB (no labels)'); a.axis('off')
            b.imshow(raw); b.set_xlim(x0, x1); b.set_ylim(y1, y0); b.axis('off')
            b.set_title('Actual rig GT projection (zoom)\nFilled: support visible | Hollow: support hidden')
            # Labels occupy two external rails, sorted by point height.
            # Separate 6/23 callouts deliberately show their coincident coordinates.
            middle = (x0 + x1) / 2
            sides = [[i for i in range(23) if keep[i] and uv[i, 0] < middle],
                     [i for i in range(23) if keep[i] and uv[i, 0] >= middle]]
            for side, indices in enumerate(sides):
                indices.sort(key=lambda i: (uv[i, 1], i))
                slots = np.linspace(y0 + 8, y1 - 8, len(indices))
                for i, yy in zip(indices, slots):
                    color = '#138859' if i in PRIMARY else '#ad4cbb'
                    x, y = uv[i]
                    b.scatter([x], [y], s=24, facecolors=color if points[i]['visible'] else 'none',
                              edgecolors=color, linewidths=1.3, zorder=5)
                    b.annotate(str(i + 1), xy=(x, y), xytext=(x0 + 5 if side == 0 else x1 - 5, yy),
                               color=color, fontsize=10, weight='bold', ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=.12', fc='white', ec=color, alpha=.92),
                               arrowprops=dict(arrowstyle='-', color=color, lw=.7), zorder=6)
            legend.axis('off')
            legend.text(0, .98, 'Core23 display number / legacy bone', va='top', weight='bold', fontsize=11)
            for i, (name, bone) in enumerate(zip(kp['names'], kp['bones'])):
                legend.text(0, .93 - i * .035, f'{i+1:02d}  {name}  [{bone}]', va='top',
                            fontsize=8.5, color='#138859' if i in PRIMARY else '#963fa4')
            legend.text(0, .09, 'Green: candidate first-pass observations\nPurple: definition/derivation deferred\n6 and 23 share EXACTLY the same joint.\nVisibility is synthetic support, not human certainty.',
                        fontsize=9, va='top')
            fig.suptitle(f'{action} | frame 31 | {role} view {view_id} | GT REFERENCE, NOT REAL PRELABEL', fontsize=15)
            fig.savefig(dest / f'{stem}_numbered.png', dpi=140)
            plt.close(fig)
            record = dict(dataset=dataset, frame=31, view_id=view_id, camera=view['camera'], role=role,
                          clean_image=f'{stem}_clean.png', numbered_image=f'{stem}_numbered.png',
                          width=w, height=h, keypoints=points,
                          source_image=str(src.relative_to(ROOT)), source_image_sha256=sha(src),
                          source_keypoints=str(kpfile.relative_to(ROOT)), source_keypoints_sha256=sha(kpfile),
                          coordinate_space='original synthetic RGB pixels; numbered zoom is reference only',
                          crop_xyxy=[int(x0), int(y0), int(x1), int(y1)], reference_only=True)
            write_json(dest / f'{stem}_gt.json', record)
            sources.append({k: v for k, v in record.items() if k != 'keypoints'})
    write_json(dest / 'index.json', sources)
    return canonical, sources


def export_frames(out, tasks):
    groups = defaultdict(list)
    for task in tasks:
        groups[task['video']].append(task)
    records = []
    for n, (video, group) in enumerate(sorted(groups.items()), 1):
        src = (ROOT / video).resolve()
        assert src.is_relative_to((ROOT / 'dataset/PyMouseStudioRaw').resolve())
        digest = sha(src)
        selected = {t['frame']: t for t in group}
        assert len(selected) == len(group)
        cap = cv2.VideoCapture(str(src))
        assert cap.isOpened(), src
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 52) < 1e-5
        frame = -1
        try:
            while frame < max(selected):
                ok, rgb = cap.read()
                frame += 1
                assert ok, (src, frame)
                if frame not in selected:
                    continue
                t = selected[frame]
                assert rgb.shape == (1200, 1080, 3), (src, rgb.shape)
                session = t['sequence_id'].split('/')[-1]
                task_id = f'{session}__frame{frame:06d}__cam{t["camera"]}'
                rel = Path('real_images') / t['split'] / f'{task_id}.png'
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(out / rel), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                reloaded = cv2.imread(str(out / rel))
                assert np.array_equal(rgb, reloaded), rel
                records.append(dict(task_id=task_id, image=rel.as_posix(), session=session,
                                    sequence_id=t['sequence_id'], source_group=t['source_group'], split=t['split'],
                                    frame_index_zero_based=frame, timestamp_seconds=frame / 52,
                                    camera=t['camera'], view_role='bottom' if t['camera'] == 'A' else 'side',
                                    width=1080, height=1200, source_video=video, source_video_sha256=digest,
                                    image_sha256=sha(out / rel), lens_transform_applied=False,
                                    source_lens_history='unconfirmed', train_allowed=False,
                                    model_predictions_shown=False, annotation_status='pending'))
        finally:
            cap.release()
        print(f'Exported {n}/{len(groups)} videos; {len(records)} images', flush=True)
    return sorted(records, key=lambda r: (r['session'], r['frame_index_zero_based'], r['camera']))


def build(out):
    out = out.resolve()
    assert out.is_relative_to((ROOT / 'output').resolve()), out
    assert not out.exists() and not out.with_suffix('.zip').exists(), 'Refusing overwrite'
    taskfile = ROOT / 'output/pymousestudio_raw_delivery_v2/annotation_tasks.jsonl'
    tasks = [json.loads(l) for l in taskfile.read_text().splitlines() if l]
    assert len(tasks) == 288
    out.mkdir(parents=True)
    shutil.copy2(ROOT / 'docs/pymousestudio_core23_annotation_20260912.md', out / 'README_标注说明.md')
    rows = export_frames(out, tasks)
    kp, examples = synthetic_examples(out)
    write_jsonl(out / 'manifest.jsonl', rows)
    write_csv(out / 'manifest.csv', rows)
    blank = [dict(task_id=r['task_id'], image=r['image'], width=r['width'], height=r['height'],
                  status='pending', annotator=None, reviewer=None, instances=[], notes='',
                  empty_instances_means='UNANNOTATED; not verified absence') for r in rows]
    write_jsonl(out / 'annotation_template.jsonl', blank)
    instance = dict(instance_id=None, identity_verified=False, bbox_body_xyxy=None,
                    bbox_full_animal_xyxy=None, bbox_extent='visible',
                    keypoints=[dict(id=i, name=name, x=None, y=None,
                                    state='pending' if i in PRIMARY else 'deferred_definition')
                               for i, name in enumerate(kp['names'])],
                    extra_observations=[], tail_visible_polyline_segments=[], notes='')
    write_json(out / 'instance_template.json', instance)
    definition_rows = [dict(id=i, display_number=i+1, name=name, chinese=CHINESE[i],
                            synthetic_bone=kp['bones'][i], synthetic_endpoint='head',
                            manual_priority='first_pass_candidate' if i in PRIMARY else 'deferred_definition',
                            manual_exact_rig_equivalence_confirmed=False)
                       for i, name in enumerate(kp['names'])]
    write_csv(out / 'core23_definitions.csv', definition_rows)
    candidates = {4:'nose', 6:'left_forepaw_center', 10:'right_forepaw_center',
                  13:'left_hindpaw_center', 16:'right_hindpaw_center', 24:'tail_base', 30:'tail_tip'}
    mapping = [dict(source_id=i, source_name=src, source_model_name=MODEL31[i],
                    core23_candidate=candidates.get(i, ''),
                    mapping_status='candidate_requires_review' if i in candidates else 'no_direct_mapping',
                    automatic_copy_allowed=False) for i, src in enumerate(SOURCE31)]
    write_csv(out / 'source31_vs_core23.csv', mapping)
    write_json(out / 'schema.json', dict(format='mousegeoforge_manual_observations_v1',
                native_annotation_platform_format=False, schema_kind='field contract, not JSON Schema',
                image_coordinates='zero-based x-right y-down original PNG pixels',
                display_number='keypoint id + 1', states_with_xy=['visible', 'occluded_localizable'],
                states_without_xy=['pending', 'unlabelable', 'out_of_frame', 'deferred_definition'],
                completed_task_status='annotated', pending_task_is_ground_truth=False,
                bbox='[xmin,ymin,xmax,ymax], same original continuous pixel coordinate system',
                anatomical_left_right=True, keypoints=definition_rows,
                extra_observation_fields=['name','x','y','state'],
                extra_observation_names=[f'observed_{part}_{side}' for part in
                                        ['ear_tip','ear_attachment','eye','wrist','elbow','ankle','knee']
                                        for side in ['l','r']],
                tail_visible_polyline_segments='list of segments, each [[x,y],...], root-to-tip order; no occlusion bridges',
                first_pass_review_required=True))
    pilot = [r['task_id'] for r in rows if r['session'] in ['AZL3','CR2','FR2']
             and r['frame_index_zero_based'] in [0, 890]]
    assert len(pilot) == 18
    write_json(out / 'pilot_ids.json', pilot)
    # A portable offline gallery opens clean PNGs; no network, no predictions.
    html = ['<!doctype html><meta charset="utf-8"><title>Core23 annotation package</title>',
            '<style>body{font:16px sans-serif;max-width:1300px;margin:30px auto}img{max-width:100%}li{margin:8px}</style>',
            '<h1>PyMouseStudioRaw：288 张待标图片</h1><p>请先读 <a href="README_标注说明.md">标注说明</a>。',
            '绿色编号为首批候选部位；紫色为定义待复核。旧 GT 投影不等于真实图的可见解剖标志。</p>',
            '<h2>Synthetic GT 示例</h2>']
    for ex in examples:
        name = ex['numbered_image']
        html.append(f'<p>{name}</p><a href="synthetic_examples/{name}"><img loading="lazy" src="synthetic_examples/{name}"></a>')
    html.append('<h2>真实图片：点击原图标注，以下不内嵌 holdout 缩略图</h2><ul>')
    for r in rows:
        html.append(f'<li>{r["split"]} | <a href="{r["image"]}">{r["task_id"]}</a></li>')
    html.append('</ul>')
    (out / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    # Clean development-only contact sheet for visual QA, never holdout model selection.
    contacts = [r for r in rows if r['session']=='AZL3' and r['frame_index_zero_based'] in [0,890]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), layout='constrained')
    for ax, r in zip(axes.flat, contacts):
        ax.imshow(Image.open(out / r['image'])); ax.set_title(r['task_id'], fontsize=8); ax.axis('off')
    fig.suptitle('Real development images only | no labels, no predictions | originals 1080 x 1200')
    fig.savefig(out / 'real_development_preview.png', dpi=140); plt.close(fig)
    counts = dict(Counter(r['split'] for r in rows))
    assert counts == {'real_development':216, 'real_locked_holdout':72}
    assert len({r['task_id'] for r in rows}) == len(rows) == 288
    assert len({r['image_sha256'] for r in rows}) == 288, 'Duplicate frames require review'
    for r in rows:
        with Image.open(out / r['image']) as im:
            assert im.size == (1080,1200) and im.mode == 'RGB'; im.verify()
    groups = defaultdict(set)
    for r in rows: groups[r['split']].add(r['source_group'])
    assert not groups['real_development'] & groups['real_locked_holdout']
    report = dict(passed=True, count=len(rows), split_counts=counts,
                  view_counts=dict(Counter(r['view_role'] for r in rows)),
                  synchronized_index_groups=len({(r['session'],r['frame_index_zero_based']) for r in rows}),
                  synchronization_hardware_verified=False, exact_unique_pngs=288, synthetic_examples=len(examples),
                  pilot_images=len(pilot), decoding='sequential from frame 0 for each video',
                  png_roundtrip_pixel_equality_checked=True, width=1080,height=1200,
                  lens_transform_applied=False, model_inference_run=False,
                  holdout_export_purpose='independent human annotation, not model selection',
                  task_list_source=str(taskfile.relative_to(ROOT)), task_list_sha256=sha(taskfile),
                  generator_sha256=sha(Path(__file__)), source_images_unchanged=True)
    write_json(out / 'verification.json', report)
    hashes = {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'checksums.json', hashes)
    zip_path = out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, (Path(out.name) / p.relative_to(out)).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(hashes)+1
    receipt = dict(zip=str(zip_path.relative_to(ROOT)), bytes=zip_path.stat().st_size,
                   sha256=sha(zip_path), zip_crc_passed=True, report=report)
    write_json(out.parent / (out.name + '.receipt.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT)
    build(parser.parse_args().out)
