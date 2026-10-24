"""WSL controller: exclusive GPU lock -> real smoke -> 768-scene render -> file QA.

Finite local rendering only. No training, Slurm, uploads, or model evaluation.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

import v5_render_contract_20261024 as c

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / c.OUT_REL
BLENDER = '/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe'
STOPPED = False


def utc():
    return dt.datetime.now(dt.timezone.utc)


def status(phase, **kw):
    c.write(OUT / 'controller_status.json', dict(utc=utc().isoformat(), pid=os.getpid(), phase=phase, **kw))
    print('V5_CONTROLLER', phase, kw, flush=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    old = c.read(ROOT / c.SOURCE_DESIGN)
    qa = c.read(ROOT / c.REVIEW_REL / 'scene_review_qa.json')
    assert qa['passed'] and c.sha(ROOT / c.REVIEW_BLEND) == qa['blend_sha256']
    design = c.make_design(old)
    c.write(OUT / 'render_design.json', design)
    code_pins = {rel: c.sha(ROOT / rel) for rel in c.PIN_FILES}
    source_pins = {s['id']: c.sha(ROOT / s['path']) for s in design['sources']}
    assert source_pins['climb'] == qa['source_sha256']
    token = str(uuid.uuid4())
    deadline = (utc() + dt.timedelta(hours=48)).isoformat()
    base = dict(schema=c.SCHEMA, design=f'{c.OUT_REL}/render_design.json',
                design_sha256=c.sha(OUT / 'render_design.json'), code_pins=code_pins,
                template_blend=c.REVIEW_BLEND, template_sha256=qa['blend_sha256'],
                template_provenance_pins={f'{c.REVIEW_REL}/{name}': c.sha(ROOT / c.REVIEW_REL / name)
                                          for name in ('scene_review_manifest.json', 'scene_review_qa.json')},
                source_sha256=source_pins, output=c.OUT_REL, stop=f'{c.OUT_REL}/STOP',
                deadline_utc=deadline, max_payload_bytes=20 * 2**30, resolution=384,
                cycles_samples=32, job_token=token, engineering_smoke=False, scenes=design['scenes'])
    c.write(OUT / 'launch_config.json', base)
    for shard in (0, 1):
        c.write(OUT / f'render_config_shard{shard}.json', dict(base, shard=shard,
                scenes=[r for r in design['scenes'] if r['shard'] == shard]))
    smoke_rows = [r for r in design['scenes'] if r['rig'] == 'train_rig00' and r['environment'] == 'train_env00' and r['frame'] == 49]
    assert len(smoke_rows) == 2
    smoke = dict(base, output=c.OUT_REL + '/engineering_smoke', engineering_smoke=True, scenes=smoke_rows)
    c.write(OUT / 'engineering_smoke/launch_config.json', smoke)
    c.write(OUT / 'engineering_smoke/render_config_shard0.json', smoke)
    c.write(OUT / 'protocol.json', dict(status='authorized_render_only', created_utc=utc().isoformat(),
            user_request=design['user_authorization'], scope='768 scenes / 2304 RGB and truth views; existing TRAIN/DEV/sealed rigs; sealed geometry only',
            counts=dict(train=576, val=144, sealed=48), environment_counts=dict(train=24, dev=6, sealed=6),
            design_sha256=base['design_sha256'], template_sha256=base['template_sha256'], code_pins=code_pins,
            deadline_utc=deadline, smoke_must_pass=True, one_GPU_renderer_at_a_time=True,
            GPU_lock='output/teacher_temporal_cycle_20261001/gpu_job.lock',
            original_sources_never_saved=True, uploads=False, training=False, model_calls=0,
            defaults='Approved V5 scene style; original v3 rigs/motions/frames; 384px RGB8, Cycles32 samples, denoising; 24/6/6 shared layout defaults.',
            stop='Create STOP in this output directory; controller and renderer check it at each stage.',
            directory='render_shard{0|1}/{rig}/{environment}/{climb|moseq}/frame_XXXXXX/{record.json,rgb_0..2.png,truth_0..2.npz}',
            release_rule='No final completion or ready manifest until every expected scene, payload hash, split and camera-Z format passes independent CPU QA.'))
    (OUT / 'README_数据布局.md').write_text('''# V5 complex synthetic render

768 scenes / 2304 views: TRAIN576, DEV144 (primary72 + attribution72), sealed48.
48 TRAIN / 12 DEV / 8 sealed camera rigs are byte-for-byte the v3 design.
24 TRAIN / 6 DEV / 6 sealed environments. Two motions, frames49/97/145.

```
render_design.json
render_config_shard0.json                 # TRAIN
render_config_shard1.json                 # DEV + sealed, rendering/geometry only
render_shard0|1/<rig>/<environment>/<motion>/frame_XXXXXX/
    record.json                          # written last, commits all 3 views
    rgb_0.png rgb_1.png rgb_2.png          # RGB8, 384x384; 0 bottom, 1/2 side
    truth_0.npz truth_1.npz truth_2.npz    # depth,body,valid,K,E
render_shard0|1/completion.json           # only after every assigned record exists
environments/<environment>/              # actual layout, per-source inventories, visibility receipt
lighting/<environment>/<rig>/<motion>/frame_XXXXXX.json
rejected_configurations/<environment>/attempt_XX.json
visibility_prescreen.json                # complete only after all 36 layouts pass
controller_status.json progress.json     # live status
manifest.json data_file_pins.json completion.json  # only after final independent QA
engineering_smoke/                       # discarded engineering input, outside formal manifest
```

All record paths are repository-relative POSIX paths. Preserve this output prefix when copying,
or use an explicit path-remapping loader; do not flatten/rename the image folders.
`split` retains v3 spelling train/val/sealed; `dev_condition` distinguishes DEV attribution.
K uses integer array-index pixel centres (principal point side/2-0.5); E is 4x4 world-to-camera
OpenCV convention. Depth is camera Z in source mm coordinates, zero when invalid. `body` is
visible opaque skin, not the fur silhouette; glass and hair are excluded from geometric truth.
Do not treat environment or glass depth as mouse foreground. New opaque cables are meshes.

The immutable design, source/template/code hashes, accepted layout parameters and all lighting
parameters accompany the render. No model calls, sealed scoring, training or uploads run here.
''', encoding='utf-8')
    status('prepared', expected_scenes=768, expected_views=2304)


def stopped():
    return STOPPED or (OUT / 'STOP').exists()


def run_blender(config_rel, name, plan):
    if stopped():
        raise InterruptedError('STOP before launching child')
    windows_root = subprocess.check_output(['wslpath', '-w', str(ROOT)], text=True).strip()
    script = windows_root + '\\BlenderScripts\\render_v5_complex_20261024.py'
    argv = [BLENDER, '--background', '--disable-autoexec', '--python-exit-code', '1', '--python', script,
            '--', '--config', config_rel, '--job-token', plan['job_token']]
    config = c.read(ROOT / config_rel)
    progress_path = ROOT / config['output'] / 'progress.json'
    stdout_path = OUT / f'{name}_stdout.log'
    stderr_path = OUT / f'{name}_stderr.log'
    # A normal retry appends logs; immutable committed records are hash-checked by the worker.
    with stdout_path.open('ab') as stdout, stderr_path.open('ab') as stderr:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr)
        c.write(OUT / 'active_child.json', dict(linux_launcher_pid=child.pid, command=argv, name=name,
                 started_utc=utc().isoformat(), job_token=plan['job_token']))
        last_stage = None
        while child.poll() is None:
            if stopped() or utc() >= dt.datetime.fromisoformat(plan['deadline_utc']):
                # Prefer a cooperative stop; terminate only this owned invocation if stuck.
                (OUT / 'STOP').touch(exist_ok=True)
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    terminate_owned_windows(plan['job_token'], progress_path)
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise InterruptedError('STOP/deadline; owned rendering invocation stopped')
            if progress_path.exists():
                progress = c.read(progress_path)
                if progress.get('state') == 'failed':
                    # Let Blender unwind its exception; never start another queue item.
                    child.wait(timeout=120)
                    raise RuntimeError(progress.get('error'))
                stage = (progress.get('stage'), progress.get('scenes_completed'))
                if stage != last_stage:
                    status(name, child=progress, gpu_lock_owned=True)
                    last_stage = stage
                if time.time() - progress_path.stat().st_mtime > 1800:
                    terminate_owned_windows(plan['job_token'], progress_path)
                    child.terminate()
                    raise RuntimeError('No renderer progress for 30 minutes')
            time.sleep(3)
        if child.returncode != 0:
            raise RuntimeError(f'{name} Blender exited {child.returncode}; see {stdout_path.name} / {stderr_path.name}')
    progress = c.read(progress_path)
    assert progress['state'] == 'completed', progress


def terminate_owned_windows(token, progress_path):
    if not progress_path.exists():
        return
    pid = int(c.read(progress_path)['windows_pid'])
    # PID and unique job token must both match the native command line.
    script = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
              f"if($null -ne $p -and $p.Name -eq 'blender.exe' -and $p.CommandLine -like '*{token}*' "
              "-and $p.CommandLine -like '*render_v5_complex_20261024.py*'){Stop-Process -Id $p.ProcessId -Force}")
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], check=True)


def validate_job(config_rel, final=False):
    config = c.read(ROOT / config_rel)
    out = ROOT / config['output']
    records = []
    for shard in sorted({r['shard'] for r in config['scenes']}):
        completion = c.read(out / f'render_shard{shard}/completion.json')
        assert not completion['rejected']
        assert completion['script_sha256'] == config['code_pins']['BlenderScripts/render_v5_complex_20261024.py']
        assert completion['config_sha256'] == c.sha(out / f'render_config_shard{shard}.json')
        records.extend(completion['records'])
    expected = {(r['rig'], r['environment'], r['source'], r['frame']) for r in config['scenes']}
    found = {(r['rig'], r['environment_id'], r['motion_id'], r['frame']) for r in records}
    assert found == expected and len(records) == len(expected)
    pins = {}
    for record in records:
        c.validate_record(ROOT, record)
        record_file = ROOT / record['scene'] / 'record.json'
        assert c.read(record_file) == record
        pins[record_file.relative_to(ROOT).as_posix()] = c.sha(record_file)
        assert record['design_sha256'] == config['design_sha256']
        for key, pin in [('environment_parameters', 'environment_parameters_sha256'), ('visibility_receipt', 'visibility_receipt_sha256'), ('lighting_parameters', 'lighting_sha256')]:
            path = ROOT / record[key]
            assert c.sha(path) == record[pin]
            pins[record[key]] = record[pin]
        for view in record['views']:
            for key in ('image', 'truth'):
                pins[view[key]] = view['file_sha256'][Path(view[key]).name]
    receipt = dict(passed=True, scenes=len(records), views=len(records) * 3,
                   split_counts=dict(collections.Counter(r['split'] for r in records)),
                   camera_z_pixel_center_contract=True, sha256_verified=True, sealed_model_calls=0,
                   engineering_smoke=config['engineering_smoke'], utc=utc().isoformat())
    c.write(out / 'data_qa.json', receipt)
    if final:
        assert len(records) == 768 and receipt['split_counts'] == dict(train=576, val=144, sealed=48)
        visibility = c.read(out / 'visibility_prescreen.json')
        assert visibility['passed'] and visibility['complete'] and len(visibility['accepted_environments']) == 36
        design = c.read(ROOT / config['design'])
        env_split = {e['id']: e['split'] for e in design['environments']}
        for record in records:
            assert env_split[record['environment_id']] == record['environment_split']
            if record['split'] == 'train':
                assert record['environment_split'] == 'train'
            elif record['split'] == 'sealed':
                assert record['environment_split'] == 'sealed'
            else:
                assert record['environment_split'] == ('dev' if record['dev_condition'] == 'dev_primary' else 'train')
        for file in ['render_design.json', 'launch_config.json', 'protocol.json', 'render_config_shard0.json',
                     'render_config_shard1.json', 'visibility_prescreen.json', 'render_shard0/completion.json',
                     'render_shard1/completion.json', 'data_qa.json', 'analytic_fixture.json']:
            pins[(out / file).relative_to(ROOT).as_posix()] = c.sha(out / file)
        c.write(out / 'data_file_pins.json', dict(schema=c.SCHEMA, files=pins))
        c.write(out / 'manifest.json', dict(schema=c.SCHEMA, records=records,
                 rig_splits={s: sorted({r['rig'] for r in records if r['split'] == s}) for s in ('train', 'val', 'sealed')},
                 environment_splits={s: sorted(e['id'] for e in design['environments'] if e['split'] == s) for s in ('train', 'dev', 'sealed')},
                 code_pins=config['code_pins'], truth_policy=design['truth_policy'], data_qa=receipt))
        c.write(out / 'completion.json', dict(**receipt, state='completed', records=records,
                 design_sha256=config['design_sha256'], manifest_sha256=c.sha(out / 'manifest.json'),
                 data_file_pins_sha256=c.sha(out / 'data_file_pins.json'), model_calls=0,
                 training_started=False, uploads_started=False))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        print(validate_job(f'{c.OUT_REL}/launch_config.json', final=True))
        return
    if not args.prepared:
        prepare()
    if args.prepare_only:
        return
    config = c.read(OUT / 'launch_config.json')
    assert Path(BLENDER).exists()
    lockpath = ROOT / 'output/teacher_temporal_cycle_20261001/gpu_job.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            c.write(OUT / 'controller_process.json', dict(pid=os.getpid(), process_group=os.getpgrp(),
                    started_utc=utc().isoformat(), job_token=config['job_token'], command=str(Path(__file__))))
            if not (OUT / 'engineering_smoke/data_qa.json').exists():
                status('engineering_smoke', gpu_lock_owned=True)
                run_blender(f'{c.OUT_REL}/engineering_smoke/launch_config.json', 'engineering_smoke', config)
                smoke = validate_job(f'{c.OUT_REL}/engineering_smoke/launch_config.json')
                status('engineering_smoke_passed', receipt=smoke)
            else:
                validate_job(f'{c.OUT_REL}/engineering_smoke/launch_config.json')
            status('production_render', expected_scenes=768, gpu_lock_owned=True)
            run_blender(f'{c.OUT_REL}/launch_config.json', 'production_render', config)
            status('final_data_qa', gpu_lock_owned=True)
            receipt = validate_job(f'{c.OUT_REL}/launch_config.json', final=True)
            status('completed', receipt=receipt, training_started=False, uploads_started=False)
        except BaseException as exc:
            status('stopped' if isinstance(exc, InterruptedError) else 'failed', error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    def stop(*_):
        global STOPPED
        STOPPED = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
