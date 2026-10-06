"""Fixed-parameter sparse manual LOO audit and continuous ON/OFF videos. Diagnostic only."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from weighted_framewise_cycle_20261002 import Geometry, selftest, BASE
from audit_teacher_dlc_predictions_20261001 import stats
from run_teacher_cycle_dlc_20261001 import ROOT, OUT, write_state

DEST = OUT / 'context_clip_geometry_review_v1'


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    deadline = min(time.time() + 1200, dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():
        assert time.time() < deadline and not (OUT / 'STOP').exists()
    check(); tests = selftest(); cv2.setNumThreads(1)
    DEST.mkdir(exist_ok=False)
    clips = json.loads((OUT / 'context_ablation_clips_v1/manifest.json').read_text())
    m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
    ids = m['ids_1based']; A = np.array(m['affine_source_to_input'])
    with np.load(OUT / 'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        manual = {(m['entries'][i]['session'], m['entries'][i]['camera'], m['entries'][i]['frame']): z['gt'][pos].copy() for pos, i in enumerate(z['manifest_indices'])}
    config = dict(floor=4., loss='soft_l1', gate=False)
    settings = dict(geometry='Existing custom scipyGeometry.solve, NOT officialAnipose/EKS', config=config, f_scale=3., max_nfev=80,
        uncertainty='Uniform4px peraxis. Single teachers have no ensemble variance; do NOT pretend native confidence is calibrated variance.',
        gates='No confidence gate/no source spread gate/no new threshold selection. Positive depth and solver success checks only.',
        held_view_removed_before_fit=True, teacher_context='Only samecamera +/-2frames; no crossview teacher inputs.',
        tests=tests, temporal_smoothing=False, dense_groundtruth_available=False, train_allowed=False,
        video='Direct raw2Dpredictions, OFFleft/ONright,camAupper/camFlower; source52fps playback26fps=half-speed. Not geometric reprojections.')
    (DEST / 'settings.json').write_text(json.dumps(settings, indent=2))
    write_state(stage='context_clip_geometry_review', pid=os.getpid(), active_run=DEST.name, mode='CPU',
        job_deadline_utc=dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat())
    rows = []; dense = []; files = []
    for clip in clips:
        check(); session = clip['session']; cohort = clip['cohort']
        assert session in plan['development_validation_sessions']
        with np.load(clip['file']) as z:
            data = {k: z[k].copy() for k in z.files}
        cams = data['cameras'].tolist(); frames = data['source_frames']; C = len(cams)
        cal = json.loads((BASE / 'calibration' / session / 'rectified_calibration.json').read_text())
        for c in cams:
            # Projection uses already rectified K and rigid world-to-camera; zero lens distortion.
            assert not np.any(np.array(cal[c].get('distortion', [])))
        geo = Geometry(cal, cams)
        uniform_var = np.full((C, 2), 16.); fake_conf = np.ones(C); no_spread = np.zeros(C)
        for vi, c in enumerate(cams):
            other = [i for i in range(C) if i != vi]
            assert vi not in other
            for fi, f in enumerate(frames):
                key = (session, c, int(f))
                if key not in manual:
                    continue
                gt = manual[key]
                for j, kp in enumerate(ids):
                    if not np.isfinite(gt[j]).all():
                        continue
                    check(); row = dict(cohort=cohort, session=session, camera=c, frame=int(f), id=kp)
                    for mode in ['off', 'on']:
                        uv = data[mode + '_uv'][:, fi, j]
                        x, meta = geo.solve(uv, uniform_var, fake_conf, no_spread, other, config)
                        projected = geo.project(x, np.array([vi]))[0][0] if np.isfinite(x).all() else np.full(2, np.nan)
                        row[mode + '_raw2d'] = float(np.linalg.norm(uv[vi] - gt[j]))
                        row[mode + '_loo'] = float(np.linalg.norm(projected - gt[j])) if np.isfinite(projected).all() else None
                        row[mode + '_meta'] = meta
                    rows.append(row)
        # Motion statistics diagnose discontinuities, not real motion or lag accuracy.
        for mode in ['off', 'on']:
            points = data[mode + '_uv']
            step = np.linalg.norm(np.diff(points, axis=1), axis=-1)
            accel = np.linalg.norm(np.diff(points, n=2, axis=1), axis=-1)
            dense.append(dict(cohort=cohort, session=session, mode=mode, frame_step_px=stats(step.ravel()),
                acceleration_px_per_frame2=stats(accel.ravel()), paw_jumps_gt50px=int(np.sum(step[:, :, 3:] > 50)),
                paw_transitions=int(step[:, :, 3:].size), warning='Large steps may be truefastmotion OR identityswitches. No smoothness=accuracy claim.'))
        selected_cams = ['A', 'F']; caps = {}
        for c in selected_cams:
            path = list(Path(clip['video_folder']).glob(f'{session}_cam{c}_*.avi')); assert len(path) == 1
            caps[c] = cv2.VideoCapture(str(path[0])); assert caps[c].isOpened()
            for _ in range(2):
                assert caps[c].read()[0]
        filename = f'{cohort}_{session}_ON_OFF_2D_half_speed.mp4'
        W, H = 1056, 1248
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
            '-r', '26', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(DEST / filename)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for fi, frame in enumerate(frames):
                check(); canvas = np.full((H, W, 3), 25, np.uint8)
                cv2.putText(canvas, f'{cohort} {session} frame{frame} | OFF left, ON right | 52fps played at26fps', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(canvas, 'Green squares: available manual labels; magenta: predictions (unknown points still shown)', (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
                for row, c in enumerate(selected_cams):
                    ok, source = caps[c].read(); assert ok; vi = cams.index(c)
                    for col, mode in enumerate(['off', 'on']):
                        im = cv2.resize(source, (528,576), interpolation=cv2.INTER_AREA)
                        for j, kp in enumerate(ids):
                            xy = (data[mode + '_uv'][vi, fi, j] * np.diag(A)[:2] + A[:2,2]) * .75
                            u, v = np.round(xy).astype(int)
                            cv2.drawMarker(im, (u,v), (220,80,255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                            cv2.putText(im, str(kp), (u+4,v+10), cv2.FONT_HERSHEY_SIMPLEX, .4, (220,80,255), 1, cv2.LINE_AA)
                            key = (session,c,int(frame))
                            if key in manual and np.isfinite(manual[key][j]).all():
                                x,y = np.round((manual[key][j]*np.diag(A)[:2]+A[:2,2])*.75).astype(int)
                                cv2.rectangle(im, (x-4,y-4),(x+4,y+4),(70,255,80),1)
                        cy, cx = 80+row*584, col*528
                        canvas[cy:cy+576,cx:cx+528] = im
                        cv2.putText(canvas, f'cam{c} {mode.upper()}', (cx+8,cy+20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255),2,cv2.LINE_AA)
                proc.stdin.write(canvas.tobytes())
                if any((session,c,int(frame)) in manual for c in selected_cams):
                    cv2.imwrite(str(DEST / f'{cohort}_{session}_frame{int(frame):06d}.jpg'), canvas)
            proc.stdin.close(); stderr = proc.stderr.read(); assert proc.wait(timeout=30) == 0, stderr
        finally:
            for cap in caps.values(): cap.release()
            if proc.poll() is None: proc.kill(); proc.wait()
        files.append(filename)
        (DEST / 'observations_progress.json').write_text(json.dumps(rows, indent=2))
        print('REVIEW_CLIP_DONE',cohort,session,flush=True)
    def summarize(subset):
        common = [r for r in subset if r['off_loo'] is not None and r['on_loo'] is not None]
        return dict(n_manual=len(subset), common_loo_n=len(common),
            raw2d={mode: stats([r[mode+'_raw2d'] for r in subset]) for mode in ['off','on']},
            common_loo={mode: stats([r[mode+'_loo'] for r in common]) for mode in ['off','on']},
            coverage={mode: sum(r[mode+'_loo'] is not None for r in subset)/len(subset) if subset else 0 for mode in ['off','on']},
            common_loo_severe_gt100px={mode: sum(r[mode+'_loo']>100 for r in common) for mode in ['off','on']})
    report = dict(all=summarize(rows), paws=summarize([r for r in rows if r['id'] in [15,16,17,18]]),
        cohorts={c:summarize([r for r in rows if r['cohort']==c]) for c in ['pilot','stress1']},
        dense_motion_diagnostics=dense, held_view_removed_before_fit=True, settings=settings,
        parity_checks=sum(len(c['sparse_png_parities']) for c in clips),
        parity_max_px=max(p['max_abs_source_px'] for c in clips for p in c['sparse_png_parities']),
        warning='Same 4frozen developmentclips, sparsemanual2D not true3D. No dense action amplitude/lag GT. No pseudo-GT release.',
        videos=files)
    (DEST / 'observations.json').write_text(json.dumps(rows,indent=2)); (DEST / 'summary.json').write_text(json.dumps(report,indent=2))
    html = '<!doctype html><meta charset="utf-8"><h1>Continuous context ON/OFF</h1><p>OFFleft / ONright, raw2D predictions. Half-speed. Green: sparse manual. Pink: model including unannotated points. Diagnostic only.</p>'
    html += ''.join(f'<h2>{f}</h2><video controls width="900" src="{f}"></video>' for f in files)
    (DEST / 'index.html').write_text(html)
    export = Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-TeacherTemporal-20261001/context_clip_geometry_review_v1')
    assert not export.exists(); shutil.copytree(DEST, export)
    write_state(stage='context_clip_geometry_reviewed',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(DEST/'summary.json'),next_action='Visually inspect paired continuous snapshots; summarize limitations and exact ON/OFF LOO results. No new training.')
    print('CONTEXT_GEOMETRY_RESULT',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
