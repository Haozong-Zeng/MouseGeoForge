"""Bottom-view foundation adaptation, with immutable source splits and matched control.

Run prepare first, preflight next, then train. No held-out test payloads are read.
Only camera/depth and a five-point 2D probe are optimized; no 3D keypoint loss.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vggt-omega'), str(ROOT / 'tools')]
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from training.finetune_blendedmvs import load_scene, read_pfm, median_camera_baseline
from vggt_omega.models import VGGTOmega
from vggt_omega.models.heads.keypoint_head import KeypointHead
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding, encoding_to_camera
from audit_bottom_view_coverage import orientation_from_extrinsic

OUT = ROOT / 'output/bottom_domain_adaptation_20261018_v1'
FIVE = ['nose', 'left_forepaw_center', 'right_forepaw_center',
        'left_hindpaw_center', 'right_hindpaw_center']
REAL_TRAIN = {'AZL3', 'AZR2', 'CR2', 'EL2', 'EL2L2', 'ER3L2'}
REAL_VAL = {'FR3', 'AZR2L2', 'FR2'}
PROTECTED = {'GR2', 'UR2', 'UR3'}
STOP_REQUESTED = False


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def image_record(frame, role):
    return dict(id=frame.view_id, image=rel(frame.image_path),
                K=frame.intrinsic.tolist(), E=frame.extrinsic.tolist(), physical_role=role,
                depth=rel(frame.depth_path) if frame.depth_path else None,
                mask=rel(frame.body_mask_path) if frame.body_mask_path else None)


def prepare():
    assert not (OUT/'manifest.json').exists(), 'Immutable registration exists; do not overwrite it.'
    OUT.mkdir(parents=True,exist_ok=True)
    grouping_path = ROOT / 'dataset/motion_source_groups.json'
    grouping = read(grouping_path)['groups']
    membership = {m: (g, v['split']) for g, v in grouping.items() for m in v['members']}
    dual_plan = read(ROOT / 'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json')
    dual_sources = {k['scene_frame']: Path(k['source_image']).parent.name for k in dual_plan['keyframes']}
    records, excluded, rigs, pins = [], [], {}, {}
    base = ROOT / 'dataset/BlenderPyMouse'
    for dataset in sorted(p for p in base.iterdir() if p.is_dir()):
        paths = sorted(dataset.glob('frame_*/blended_images'))
        if not paths:
            continue
        for image_dir in paths:
            scene_path = image_dir.parent
            group_split = membership.get('BlenderPyMouse/' + dataset.name)
            if group_split is None:
                if 'KeypointMoseq-Core23-Sample10' in dataset.name:
                    group_split = ('moseq_sample10_all_correlated_variants', 'train')
                elif 'DiversePose400' in dataset.name:
                    session = dual_sources[int(scene_path.name.split('_')[-1])]
                    group_split = ('crim13/' + session, 'train' if session == '021709_A25_Block22_BCfe1_t' else 'val')
                else:
                    excluded.append(dict(path=rel(scene_path), reason='No registered motion-source split; no automatic admission'))
                    continue
            group, split = group_split
            if split == 'test':
                excluded.append(dict(path=rel(scene_path), reason='Protected motion-source test split'))
                continue
            scene = load_scene(scene_path)
            views = []
            for f in scene.frames.values():
                assert f.extrinsic is not None and f.intrinsic is not None, str(scene_path)
                role = orientation_from_extrinsic(f.extrinsic)[1]
                views.append(image_record(f, role))
                pins[rel(f.cam_path)] = sha(f.cam_path)
            if not views:
                raise ValueError(f'Empty scene: {scene_path}')
            kp = scene_path / 'keypoints.npz'
            records.append(dict(kind='synthetic', split=split, group=group, scene=rel(scene_path),
                                views=views, keypoints=rel(kp) if kp.exists() else None))
    # Read only metadata for protected acquisitions; never open their labels or images.
    mp = ROOT / 'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'
    manifest = read(mp); pins[rel(mp)] = sha(mp)
    provenance = collections.defaultdict(list)
    for r in manifest['provenance']:
        if r['session'] in REAL_TRAIN | REAL_VAL:
            provenance[r['session'], r['frame']].append(r)
    for g in manifest['frame_groups']:
        session = g['session']
        if session not in REAL_TRAIN | REAL_VAL:
            continue
        if not g['eligible_for_development'] or g['excluded']:
            excluded.append(dict(session=session, frame=g['frame'], reason='Existing quality exclusion'))
            continue
        cal = ROOT / 'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration' / session / 'rectified_calibration.json'
        calibration = read(cal); pins[rel(cal)] = sha(cal)
        views = []
        for r in provenance[session, g['frame']]:
            c = calibration[r['camera']]
            assert not np.any(c['distortion']) and c['size_wh'] == [1080, 1200]
            views.append(dict(id=ord(r['camera']) - 65, image=r['source_image'], K=c['K'],
                              E=c['world_to_camera_mm'], physical_role='bottom' if r['camera'] == 'A' else 'other',
                              depth=None, mask=None))
        label = g['effective_normalized_file']; pins[label] = sha(ROOT / label)
        records.append(dict(kind='real_labeled', split='train' if session in REAL_TRAIN else 'val',
                            group=session, frame=g['frame'], views=views, keypoints=label))
    sequences = read(ROOT / 'dataset/PyMouseStudio/sequences.json')
    # All available frames of admitted training captures enter the camera pool.
    # The lossy rectified video stream has no sparse point/depth supervision.
    for s in sequences:
        if s['session'] not in REAL_TRAIN:
            continue
        session = s['session']; root = ROOT / 'dataset/PyMouseStudio/sessions' / session
        calibration = read(root / 'calibration.json'); pins[rel(root / 'calibration.json')] = sha(root / 'calibration.json')
        views = []
        for cam in s['cameras']:
            p = root / 'videos' / f'cam{cam}.mp4'
            if p.exists():
                c = calibration[cam]
                assert not np.any(c['distortion'])
                cap = cv2.VideoCapture(str(p)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
                assert count >= s['frame_count'], (p, count, s['frame_count'])
                views.append(dict(id=ord(cam)-65, video=rel(p), K=c['K'], E=c['world_to_camera_mm'],
                                  physical_role='bottom' if cam == 'A' else 'other', depth=None, mask=None))
        assert any(v['id'] == 0 for v in views)
        rig_key = 'video/' + session; rigs[rig_key] = views
        bad = {g['frame'] for g in manifest['frame_groups'] if g['session'] == session and g['excluded']}
        for frame in range(s['frame_count']):
            if frame not in bad:
                records.append(dict(kind='real_video', split='train', group=session, frame=frame, rig=rig_key))
    # DANNCE camera labels add a second real rig. Its pseudo depth is NOT GT.
    # Mouse2 is a protected TEST group, not an additional development set.
    for number in (1,):
        root = ROOT / f'dataset/DANNCE-MarkerlessMouse{number}-Core23-RealMultiview/sessions/markerless_mouse_{number}'
        meta = read(root / 'metadata.json'); pins[rel(root / 'metadata.json')] = sha(root / 'metadata.json')
        pins[rel(root / 'frames.jsonl')] = sha(root / 'frames.jsonl')
        group, split = membership[rel(root).removeprefix('dataset/')]
        assert split == 'train'
        rig_key = f'dannce/{number}'
        rigs[rig_key] = [dict(id=c['view_id'], K=c['intrinsic'], E=c['extrinsic'],
                             physical_role='unknown_real_orientation', depth=None, mask=None) for c in meta['cameras']]
        for line in (root / 'frames.jsonl').open():
            row = json.loads(line)
            records.append(dict(kind='real_external', split=split, group=group, frame=row['local_index'], rig=rig_key,
                                rgb=[rel(root / p) for p in row['rgb']]))
    # Explicitly retain dataset exclusions, rather than interpreting missing labels as negatives.
    excluded.extend([
        dict(path='dataset/DANNCE-MarkerlessMouse2-Core23-RealMultiview', reason='Protected motion-source TEST; no image/label payload read'),
        dict(path='dataset/Orbbec-Openfield', reason='Single top-view; camera/world supervision invalid; not admitted to geometric objectives'),
        dict(path='dataset/SBeA-Mouse16-Core23-RealMultiviewTemporal', reason='Camera-to-pose alignment unavailable; no certified bottom-view/camera targets'),
        dict(path='dataset/sDANNCE-Mouse-Core23-PoseSequences', reason='No RGB or cameras'),
    ])
    train_groups = {r['group'] for r in records if r['split'] == 'train'}
    val_groups = {r['group'] for r in records if r['split'] == 'val'}
    assert not train_groups & val_groups and not PROTECTED & (train_groups | val_groups)
    assert not any(any(s in str(r) for s in PROTECTED) for r in records)
    counts = collections.Counter((r['split'], r['kind'], r['group']) for r in records)
    write(OUT / 'manifest.json', dict(utc=utc(), records=records, rigs=rigs, exclusions=excluded,
                                    counts=[dict(split=k[0], kind=k[1], group=k[2], groups=v) for k,v in counts.items()]))
    write(OUT / 'protocol.json', dict(
        utc=utc(), authorization='2026-10-18 user: start bottom domain adaptation with all trainable data, independent validation and other-view replay',
        evidence='User directly inspected released demo on real synchronized multiview video containing bottom views and observed failure. This is the initiating empirical observation.',
        manifest_sha256=sha(OUT / 'manifest.json'), source_pins=pins,
        code_sha256=sha(__file__), checkpoint='vggt-omega/vggt_omega_1b_512.pt',
        checkpoint_sha256=sha(ROOT / 'vggt-omega/vggt_omega_1b_512.pt'),
        image_size=512, seeds=[17,29,43], first_run_seed=17,
        arms=['adapt_last2', 'frozen_encoder_control'],
        initialization='Fresh release aggregator/camera/depth; same seed random five-point 2D head in both arms; no old adapters',
        optimization='AdamW lr1e-5 heads, 1e-6 last2 frame/inter-frame blocks; confidence and camera/register input tokens frozen; clip1; BF16 encoder, FP32 heads',
        objectives='Canonical first-camera and median full-rig baseline gauge; camera translation/quaternion/FoV + relative rotation/direction; synthetic camera-Z log Huber and image gradients; five-point 2D Gaussian CE only where supported. No real/pseudo depth, no 3D keypoint objective.',
        sampling='Coverage-first schedule: every synthetic and labeled-real camera pack, every admitted video frame with rotating bottom-containing context, every DANNCE1 frame with rotating camera context; shuffle/interleave all sources. After coverage, repeat shuffled source-balanced packs. Side/top replay included explicitly.',
        validation='Full labeled FR3/AZR2L2/FR2 and source-group held-out synthetic pools registered; fixed at most8 scenes per synthetic group for interim evaluation. FR3 is reused development; AZR2L2/FR2 acquisition-separated validation, not unseen animal certification. GR2/UR2/UR3, DANNCE2 and test motion groups stay protected.',
        phases='Bounded seed17 feasibility/learning curve first; no automatic deployment or 3D keypoint training. Then matched frozen-encoder control, then seeds29/43 if first paired comparison supports it.',
        initial_budget=dict(max_steps=24000, max_hours=12, evaluation_steps='0,250,then every1000,final', checkpoint_every=500),
        gates=dict(geometry='bottom relative rotation median and P95 improve>=30%; rotation P95<=10deg, translation direction P95<=15deg; synthetic co-visible predicted-depth+predicted-camera transfer coverage32 improves>=5pp',
                   localization='bottom reference coverage32 +3pp OR P95 -30%, no increase >100px or contralateral proxy; whole-view deployment coverage32>=90% and P95<=32 separate',
                   retention='other-view coverage32 drop<=1pp; other-view geometry/depth errors increase<=5%; group/seed results all disclosed',
                   promotion='All gates on acquisition/motion-source held-out data and same-budget frozen encoder control; no training-loss-only promotion. Only after paired confirmation start 3D keypoints'),
        limitations=['Real calibration is inherited rather than independently remeasured; camera metrics are against supplied calibration.',
                     'Real video and labeled PNGs differ by lossy video encoding; report separately.',
                     'DANNCE world is not ground aligned; do not infer bottom from its raw Z axis.',
                     'Synthetic legacy rendered depth/body masks are separate from certified exact visible skin; report provenance.',
                     'Admission is complete pools, not a claim that the bounded first run has visited every frame. Track actual unique exposure.',
                     'Global scale is fixed by training camera baseline; no per-frame GT point/scale alignment at evaluation.']))
    print(json.dumps(read(OUT/'manifest.json')['counts'], ensure_ascii=False), flush=True)


def views_for(record, manifest):
    views = [dict(v) for v in record.get('views', manifest['rigs'].get(record.get('rig'), []))]
    if record['kind'] == 'real_external':
        for v, path in zip(views, record['rgb']):
            v['image'] = path
    return sorted(views, key=lambda v:v['id'])


def packs(views):
    ids = [v['id'] for v in views]
    bottom = [v['id'] for v in views if v['physical_role'] == 'bottom']
    if len(ids) <= 3:
        return [ids]
    if bottom:
        b=bottom[0]; rest=[v for v in ids if v!=b]
        return [[b] + rest[j:j+2] for j in range(0,len(rest),2)] + [rest[:3]]
    return [ids[j:j+3] for j in range(0,len(ids),3) if len(ids[j:j+3])>=2] + [ids[-3:]]


@lru_cache(maxsize=8)
def video_capture(path):
    cap = cv2.VideoCapture(str(ROOT / path))
    if not cap.isOpened():
        raise RuntimeError(path)
    return cap


def resized(im, side, nearest=False):
    w,h=im.size; scale=min(side/w, side/h)
    nw,nh=round(w*scale),round(h*scale); x,y=(side-nw)//2,(side-nh)//2
    out=Image.new(im.mode,(side,side),0)
    out.paste(im.resize((nw,nh),Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC),(x,y))
    affine=np.array([[nw/w,0,x],[0,nh/h,y],[0,0,1]],np.float32)
    return np.asarray(out).copy(), affine


def load_item(record, manifest, pack, side=512):
    allviews=views_for(record,manifest); selected=[next(v for v in allviews if v['id']==i) for i in pack]
    es=np.array([v['E'] for v in allviews],np.float32)
    scale=median_camera_baseline(es) if len(es)>1 else 1.
    e0=np.array(selected[0]['E'],np.float32); e0inv=np.linalg.inv(e0)
    images=[]; extrinsics=[]; intrinsics=[]; depths=[]; masks=[]; bodies=[]; affines=[]
    for v in selected:
        if 'video' in v:
            cap=video_capture(v['video']); cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame'])
            okay,bgr=cap.read()
            if not okay:raise RuntimeError((v['video'],record['frame']))
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        else:
            with Image.open(ROOT/v['image']) as source: im=source.convert('RGB')
        rgb,a=resized(im,side); affines.append(a)
        images.append(torch.from_numpy(rgb).permute(2,0,1).float()/255)
        intrinsics.append(a@np.array(v['K'],np.float32))
        e=np.array(v['E'],np.float32)@e0inv; e[:3,3]/=scale; extrinsics.append(e[:3])
        if record['kind']=='synthetic' and v.get('depth'):
            z=read_pfm(ROOT/v['depth'],flip_y=True)
            z,a_depth=resized(Image.fromarray(z.astype(np.float32)),side,True)
            # Raster/image dimensions must be identical before the same affine.
            np.testing.assert_allclose(a_depth,a,atol=1e-6)
            z=z.astype(np.float32)/scale
            valid=np.isfinite(z)&(z>1e-6)
        else:z=np.zeros((side,side),np.float32);valid=np.zeros((side,side),bool)
        body=np.zeros((side,side),bool)
        if v.get('mask'):
            with Image.open(ROOT/v['mask']) as m: body=resized(m.convert('L'),side,True)[0]>0
        depths.append(np.nan_to_num(z)); masks.append(valid); bodies.append(body)
    xy=np.zeros((len(selected),5,2),np.float32); known=np.zeros((len(selected),5),bool)
    if record.get('keypoints'):
        with np.load(ROOT/record['keypoints'],allow_pickle=False) as z:
            names=z['names'].tolist()
            if all(n in names for n in FIVE):
                indices=[names.index(n) for n in FIVE]
                uv=z['uv']; uv=uv[:,0] if uv.ndim==4 and uv.shape[1]==1 else uv
                if uv.ndim==3:  # Multi-instance scenes supervise geometry, not an ambiguous merged 2D probe.
                    vids=([ord(str(c)[-1])-65 for c in z['camera_names']] if record['kind']=='real_labeled' else z['view_ids'].tolist())
                    valid=z['xy_observed'] if record['kind']=='real_labeled' else z['xy_label_valid']
                    if valid.ndim==3:valid=valid[:,0]
                    if record['kind']=='real_labeled' and record['split']=='train':
                        loss_valid=z['loss_eligible']; valid=valid&(loss_valid[:,0] if loss_valid.ndim==3 else loss_valid)
                    for j,v in enumerate(selected):
                        k=vids.index(v['id']); p=uv[k,indices]; xy[j]=p@affines[j][:2,:2].T+affines[j][:2,2]
                        known[j]=valid[k,indices]&np.isfinite(p).all(-1)&(xy[j]>=0).all(-1)&(xy[j]<side).all(-1)
    return dict(images=torch.stack(images)[None], E=torch.tensor(np.array(extrinsics))[None],
                K=torch.tensor(np.array(intrinsics))[None], depth=torch.tensor(np.array(depths))[None],
                valid=torch.tensor(np.array(masks))[None], body=torch.tensor(np.array(bodies))[None],
                xy=torch.tensor(np.nan_to_num(xy))[None], known=torch.tensor(known)[None],
                roles=[v['physical_role'] for v in selected], ids=pack, record=record)


def camera_terms(pose, batch):
    target=extri_intri_to_pose_encoding(batch['E'],batch['K'],batch['images'].shape[-2:])
    q=F.normalize(pose[...,3:7],dim=-1); qt=F.normalize(target[...,3:7],dim=-1)
    # A single sign must apply to the entire quaternion, not separately per coordinate.
    quat=torch.minimum((q-qt).square().sum(-1),(q+qt).square().sum(-1)).mean()
    pe,_=encoding_to_camera(pose,batch['images'].shape[-2:])
    pr=pe[:,1:,:3,:3]@pe[:,:1,:3,:3].transpose(-1,-2)
    tr=batch['E'][:,1:,:3,:3]@batch['E'][:,:1,:3,:3].transpose(-1,-2)
    pt=pe[:,1:,:3,3]-(pr@pe[:,:1,:3,3,None])[...,0]
    tt=batch['E'][:,1:,:3,3]-(tr@batch['E'][:,:1,:3,3,None])[...,0]
    relative=(pr-tr).square().mean()+(F.normalize(pt,dim=-1)-F.normalize(tt,dim=-1)).square().mean() if pose.shape[1]>1 else pose.sum()*0
    return F.smooth_l1_loss(pose[...,:3],target[...,:3])+quat+F.smooth_l1_loss(pose[...,7:],target[...,7:])+relative


def loss_terms(pred,b):
    cam=camera_terms(pred['pose_enc'],b)
    z=pred['depth'][...,0].clamp_min(1e-6); valid=b['valid']
    residual=z.log()-b['depth'].clamp_min(1e-6).log()
    body=valid&b['body']; other=valid&~b['body']
    terms=[F.smooth_l1_loss(residual[m],torch.zeros_like(residual[m])) for m in (body,other) if m.any()]
    depth=torch.stack(terms).mean() if terms else z.sum()*0
    grads=[]
    for axis in (-2,-1):
        r=residual.diff(dim=axis); good=valid.narrow(axis,0,valid.shape[axis]-1)&valid.narrow(axis,1,valid.shape[axis]-1)
        if good.any():grads.append(r[good].abs().mean())
    gradient=torch.stack(grads).mean() if grads else z.sum()*0
    logits=pred['keypoint_heatmap_logits']; h,w=logits.shape[-2:]; side=b['images'].shape[-1]
    yy,xx=torch.meshgrid(torch.arange(h,device=z.device),torch.arange(w,device=z.device),indexing='ij')
    xy=b['xy']/side*torch.tensor([w,h],device=z.device)-.5
    gaussian=-((xx-xy[...,0,None,None]).square()+(yy-xy[...,1,None,None]).square())/(2*(8*w/side)**2)
    target=gaussian.flatten(-2).softmax(-1)
    ce=-(target*logits.flatten(-2).log_softmax(-1)).sum(-1)
    loc=ce[b['known']].mean() if b['known'].any() else logits.sum()*0
    return cam+.2*depth+.02*gradient+loc, dict(camera=cam,depth=depth,gradient=gradient,localization=loc)


def make_model(arm,seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model=VGGTOmega(enable_camera=True,enable_depth=True)
    state=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    result=model.load_state_dict({k:v for k,v in state.items() if k.startswith(('aggregator.','camera_head.','dense_head.'))},strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    del state
    model.keypoint_head=KeypointHead(num_keypoints=5)
    model.requires_grad_(False)
    for module in (model.camera_head,model.dense_head,model.keypoint_head):module.requires_grad_(True)
    model.dense_head.proj_conf.requires_grad_(False)
    if arm=='adapt_last2':
        for module in list(model.aggregator.frame_blocks[-2:])+list(model.aggregator.inter_frame_blocks[-2:]):module.requires_grad_(True)
    model=model.cuda().train()
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('aggregator.')],lr=1e-5)]
    encoder=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('aggregator.')]
    if encoder:groups.append(dict(params=encoder,lr=1e-6))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
    return model,optimizer


def on_device(b):
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in b.items()}


def metric_rows(pred,b):
    pose=pred['pose_enc']; pe,_=encoding_to_camera(pose,b['images'].shape[-2:]); rows=[]
    for j,role in enumerate(b['roles']):
        row=dict(group=b['record']['group'],kind=b['record']['kind'],role=role,view=b['ids'][j], frame=b['record'].get('frame'),scene=b['record'].get('scene'))
        if j:
            pr=pe[0,j,:3,:3]@pe[0,0,:3,:3].T; tr=b['E'][0,j,:3,:3]@b['E'][0,0,:3,:3].T
            angle=torch.acos(((torch.trace(pr@tr.T)-1)/2).clamp(-1,1))*180/math.pi
            pt=pe[0,j,:3,3]-pr@pe[0,0,:3,3];tt=b['E'][0,j,:3,3]-tr@b['E'][0,0,:3,3]
            direction=torch.acos((F.normalize(pt,dim=0)*F.normalize(tt,dim=0)).sum().clamp(-1,1))*180/math.pi
            row['rotation_deg']=float(angle);row['direction_deg']=float(direction)
            row['baseline_relative_error']=float((pt.norm()-tt.norm()).abs()/tt.norm().clamp_min(1e-8))
            # Pair contains bottom if either endpoint is bottom, including a bottom reference.
            row['pair_role']='bottom' if 'bottom' in (b['roles'][j],b['roles'][0]) else 'other'
        gt_fov=extri_intri_to_pose_encoding(b['E'],b['K'],b['images'].shape[-2:])[0,j,7:]
        row['focal_relative_error']=float((torch.tan(gt_fov/2)/torch.tan(pose[0,j,7:]/2)-1).abs().mean())
        if b['known'][0,j].any():
            heat=pred['keypoint_heatmap_logits'][0,j];h,w=heat.shape[-2:];arg=heat.flatten(-2).argmax(-1)
            xy=torch.stack([(arg%w+.5)/w,(arg//w+.5)/h],-1)*b['images'].shape[-1]
            delta=(xy-b['xy'][0,j])*torch.tensor([704,768],device=xy.device)/b['images'].shape[-1]
            errors=delta.norm(dim=-1);known=b['known'][0,j]
            row['point_errors_ref']=errors[known].tolist();row['point_ids']=torch.arange(5,device=known.device)[known].tolist()
            proxy=0
            for a,c in [(1,2),(2,1),(3,4),(4,3)]:
                if known[a] and known[c] and errors[a]>32:
                    proxy+=int((((xy[a]-b['xy'][0,j,c])*torch.tensor([704,768],device=xy.device)/b['images'].shape[-1]).norm()<=32))
            row['contralateral_proxy']=proxy
        valid=b['valid'][0,j]&b['body'][0,j]
        if valid.any():
            z=pred['depth'][0,j,...,0];gt=b['depth'][0,j]
            row['body_absrel']=float(((z[valid]-gt[valid]).abs()/gt[valid]).mean())
            p=z[valid]-z[valid].mean();t=gt[valid]-gt[valid].mean()
            row['centered_shape_rmse_baseline']=float((p-t).square().mean().sqrt())
        rows.append(row)
    for source in range(len(rows)):
        rows[source]['transfer'] = transfer_metrics(pred,b,source)
    return rows


def transfer_metrics(pred,b,source):
    """Evaluate predicted depth AND its own predicted cameras, without GT scale fitting.

    Support is fixed by synthetic GT co-visibility, never by predicted confidence.
    Nonfinite/behind-camera predictions remain failures in that fixed denominator.
    """
    if b['record']['kind']!='synthetic' or len(b['roles'])<2:
        return []
    device=b['images'].device;side=b['images'].shape[-1]
    yy,xx=torch.meshgrid(torch.arange(8,side,16,device=device),torch.arange(8,side,16,device=device),indexing='ij')
    x,y=xx.flatten(),yy.flatten();support=b['valid'][0,source,y,x]&b['body'][0,source,y,x]
    if not support.any():return []
    x,y=x[support],y[support];uv=torch.stack((x,y,torch.ones_like(x)),0).float()
    gt_z=b['depth'][0,source,y,x];ray=torch.linalg.solve(b['K'][0,source],uv)
    gt_cam=ray*gt_z[None];se=b['E'][0,source]
    gt_world=torch.linalg.solve(se[:3,:3],gt_cam-se[:3,3,None])
    pe,pk=encoding_to_camera(pred['pose_enc'],(side,side));ps=pe[0,source]
    pred_ray=torch.linalg.solve(pk[0,source],uv)
    pred_cam=pred_ray*pred['depth'][0,source,y,x,0][None]
    pred_world=torch.linalg.solve(ps[:3,:3],pred_cam-ps[:3,3,None]);result=[]
    for target in range(len(b['roles'])):
        if target==source:continue
        te=b['E'][0,target];gt_target=te[:3,:3]@gt_world+te[:3,3,None]
        homogeneous=b['K'][0,target]@gt_target;expected=homogeneous[:2]/homogeneous[2:]
        inside=torch.isfinite(expected).all(0)&(gt_target[2]>0)&(expected>=0).all(0)&(expected<side-1).all(0)
        u,v=expected.nan_to_num().round().long().clamp(0,side-1)
        target_z=b['depth'][0,target,v,u]
        good=inside&b['valid'][0,target,v,u]&((target_z-gt_target[2]).abs()<(.01*target_z+.001))
        count=int(good.sum())
        if not count:continue
        pt=pe[0,target,:3,:3]@pred_world+pe[0,target,:3,3,None]
        projection=pk[0,target]@pt;actual=projection[:2]/projection[2:]
        error=((actual-expected)*torch.tensor([[704/side],[768/side]],device=device)).norm(dim=0)
        finite=torch.isfinite(error)&(pt[2]>0)
        success=good&finite&(error<=32)
        result.append(dict(target=b['ids'][target],pair_role='bottom' if 'bottom' in (b['roles'][source],b['roles'][target]) else 'other',
                           support=count,within32=int(success.sum()),invalid=int((good&~finite).sum())))
    return result


def summarize(rows):
    result={}
    for role in ('bottom','other','all'):
        rr=[r for r in rows if role=='all' or (r['role']=='bottom')==(role=='bottom')]
        pairs=[r for r in rows if 'rotation_deg' in r and (role=='all' or r['pair_role']==role)]
        errors=[e for r in rr for e in r.get('point_errors_ref',[])]
        data=dict(views=len(rr),camera_pairs=len(pairs),known_points=len(errors))
        for name,values in [('rotation_deg',[r['rotation_deg'] for r in pairs]),('direction_deg',[r['direction_deg'] for r in pairs]),('localization_ref',errors),('body_absrel',[r['body_absrel'] for r in rr if 'body_absrel' in r])]:
            if values:data[name]=dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(sorted(values)[math.ceil(.95*len(values))-1]))
        if errors:data.update(coverage8=float(np.mean(np.array(errors)<=8)),coverage16=float(np.mean(np.array(errors)<=16)),coverage32=float(np.mean(np.array(errors)<=32)),p99=float(sorted(errors)[math.ceil(.99*len(errors))-1]),over100=int(sum(e>100 for e in errors)),contralateral_proxy=sum(r.get('contralateral_proxy',0) for r in rr))
        transfer=[t for r in rows for t in r.get('transfer',[]) if role=='all' or t['pair_role']==role]
        support=sum(t['support'] for t in transfer)
        data['predicted_camera_depth_transfer']=dict(support=support,coverage32=sum(t['within32'] for t in transfer)/support if support else None,invalid=sum(t['invalid'] for t in transfer))
        result[role]=data
    # Macro source statistics kept alongside pooled values, never silently pooled as independent frames.
    result['per_group']={g:summarize_group([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    return result


def summarize_group(rows):
    result={}
    for role in ('bottom','other','all'):
        rr=[r for r in rows if role=='all' or (r['role']=='bottom')==(role=='bottom')]
        pairs=[r for r in rows if 'rotation_deg' in r and (role=='all' or r['pair_role']==role)]
        errors=[e for r in rr for e in r.get('point_errors_ref',[])]
        result[role]=dict(views=len(rr),camera_pairs=len(pairs),known_points=len(errors),
            rotation_mean=float(np.mean([r['rotation_deg'] for r in pairs])) if pairs else None,
            coverage32=float(np.mean(np.array(errors)<=32)) if errors else None,
            p95=sorted(errors)[math.ceil(.95*len(errors))-1] if errors else None,
            over100=sum(e>100 for e in errors),focal_relative_error_mean=float(np.mean([r['focal_relative_error'] for r in rr])) if rr else None)
    return result


def eval_indices(manifest):
    grouped=collections.defaultdict(list)
    for i,r in enumerate(manifest['records']):
        if r['split']=='val':grouped[r['kind'],r['group']].append(i)
    result=[]
    for (kind,group),indices in sorted(grouped.items()):
        n=len(indices) if kind=='real_labeled' else min(len(indices),24 if kind=='real_external' else 8)
        result.extend(indices[j] for j in sorted(set(np.linspace(0,len(indices)-1,n).round().astype(int))))
    return result


@torch.no_grad()
def evaluate(model,manifest,path,deadline):
    model.eval();rows=[]
    for i in eval_indices(manifest):
        if STOP_REQUESTED or time.monotonic()>deadline:raise InterruptedError('Budget/stop during evaluation')
        r=manifest['records'][i]; vv=views_for(r,manifest)
        # Fixed packs rotate reference to a side camera so bottom camera error is directly observed.
        for pack in packs(vv):
            if len(pack)>1 and next(v for v in vv if v['id']==pack[0])['physical_role']=='bottom':pack=pack[1:]+pack[:1]
            b=on_device(load_item(r,manifest,pack)); pred=model(b['images']);rows.extend(metric_rows(pred,b))
    write(path,dict(utc=utc(),summary=summarize(rows),rows=rows,selection='Fixed registered interim validation; not the protected test'))
    model.train();return summarize(rows)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def save_checkpoint(path,model,optimizer,step,exposures,protocol):
    state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=state,optimizer=optimizer.state_dict(),step=step,rng=rng_state(),exposures=dict(exposures),
                    protocol_sha256=sha(OUT/'protocol.json'),base_sha256=protocol['checkpoint_sha256']),tmp)
    tmp.replace(path)


def schedule(manifest,steps,seed):
    pools=collections.defaultdict(list); mandatory=[]
    for i,r in enumerate(manifest['records']):
        if r['split']!='train':continue
        choices=packs(views_for(r,manifest));pools[r['kind']].append(i)
        if r['kind'] in ('synthetic','real_labeled'):
            selected=choices
        elif r['kind']=='real_video':
            bottom_choices=[p for p in choices if any(v['physical_role']=='bottom' and v['id'] in p for v in views_for(r,manifest))]
            selected=[bottom_choices[r['frame']%len(bottom_choices)]]
        else:selected=[choices[r['frame']%len(choices)]]
        mandatory.extend(dict(index=i,views=p) for p in selected)
    rng=random.Random(seed);rng.shuffle(mandatory)
    # Repeated passes after the compulsory coverage pass remain source-balanced.
    orders={};positions=collections.Counter();visits=collections.Counter()
    pattern=['synthetic']*4+['real_labeled']*4+['real_video','real_external'];out=[]
    for step in range(steps):
        if step<len(mandatory):
            item=dict(mandatory[step]);pack=item['views']
            if len(pack)>1 and step%2:pack=pack[1:]+pack[:1]
            out.append(dict(index=item['index'],views=pack));continue
        kind=pattern[step%10]
        if positions[kind]%len(pools[kind])==0:
            orders[kind]=pools[kind].copy();rng.shuffle(orders[kind]);positions[kind]=0
        i=orders[kind][positions[kind]];positions[kind]+=1
        choices=packs(views_for(manifest['records'][i],manifest));pack=choices[visits[i]%len(choices)];visits[i]+=1
        # Alternate reference position; otherwise a bottom reference has no direct relative-pair metric.
        if len(pack)>1 and step%2:pack=pack[1:]+pack[:1]
        out.append(dict(index=i,views=pack))
    return out


def fixtures():
    e=torch.eye(4).repeat(1,3,1,1)[:,:,:3];e[0,1,0,3]=1;e[0,2,1,3]=1
    k=torch.tensor([[400.,0,256],[0,400.,256],[0,0,1]]).repeat(1,3,1,1)
    b=dict(E=e,K=k,images=torch.zeros(1,3,3,512,512))
    p=extri_intri_to_pose_encoding(e,k,(512,512));p.requires_grad_()
    assert float(camera_terms(p,b))<1e-10
    changed=p.detach().clone();changed[:,1,0]+=.5;changed.requires_grad_();loss=camera_terms(changed,b);loss.backward()
    assert loss>0 and torch.isfinite(changed.grad).all() and changed.grad.abs().sum()>0
    neg=p.detach().clone();neg[...,3:7]*=-1;assert float(camera_terms(neg,b))<1e-10
    assert packs([dict(id=i,physical_role='bottom' if i==6 else 'side') for i in range(8)])==[[6,0,1],[6,2,3],[6,4,5],[6,7],[0,1,2]]
    _,a=resized(Image.new('RGB',(1080,1200)),512)
    np.testing.assert_allclose(a@np.array([540,600,1]),[256,256,1],atol=.6)
    return dict(camera_identity=True,quaternion_global_sign=True,nonzero_finite_gradient=True,physical_bottom_sampling=True,letterbox_camera_transform=True)


def run(arm,seed,steps,hours,preflight=False):
    global STOP_REQUESTED
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=read(OUT/'protocol.json');manifest=read(OUT/'manifest.json')
    assert sha(OUT/'manifest.json')==protocol['manifest_sha256']
    assert sha(__file__)==protocol['code_sha256'], 'Code changed after registration'
    # Preparation pins all admitted label/camera files. Do not silently use a newer revision.
    for p,digest in protocol['source_pins'].items():assert sha(ROOT/p)==digest,p
    name=('preflight_' if preflight else '')+f'{arm}_seed{seed}'
    dest=OUT/name;dest.mkdir(exist_ok=False)
    began=time.monotonic();deadline=began+hours*3600
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=dest/'status.json';step=0;exposures=collections.Counter();unique=set()
    model=optimizer=None
    def status(state,**kwargs):
        write(status_path,dict(utc=utc(),state=state,pid=os.getpid(),arm=arm,seed=seed,step=step,elapsed_s=time.monotonic()-began,
                               max_steps=steps,max_hours=hours,unique_record_packs=len(unique),exposures=dict(exposures),**kwargs))
    try:
        status('loading');write(dest/'fixtures.json',fixtures())
        assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
        model,optimizer=make_model(arm,seed)
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        write(dest/'trainable.json',dict(names=trainable,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        work=schedule(manifest,steps,seed);write(dest/'schedule.json',work)
        if preflight:
            # Every admitted scene/view path/camera is checked at prepare; decode one representative
            # per kind and real held-out acquisition through the actual consumer before training.
            examples=[]
            for kind in ('synthetic','real_labeled','real_video','real_external'):
                i=next(i for i,r in enumerate(manifest['records']) if r['kind']==kind and r['split']=='train');examples.append(i)
            examples += [next(i for i,r in enumerate(manifest['records']) if r['group']==g) for g in sorted(REAL_VAL)]
            work=[dict(index=i,views=packs(views_for(manifest['records'][i],manifest))[0]) for i in examples[:4]]
            write(dest/'schedule.json',work)
            for i in examples:
                r=manifest['records'][i];b=load_item(r,manifest,packs(views_for(r,manifest))[0])
                assert torch.isfinite(b['images']).all() and torch.isfinite(b['E']).all() and torch.isfinite(b['K']).all()
                assert not b['valid'].any() if r['kind']!='synthetic' else b['valid'].any()
            for item in work:
                r=manifest['records'][item['index']];b=on_device(load_item(r,manifest,item['views']))
                optimizer.zero_grad(set_to_none=True);pred=model(b['images']);loss,terms=loss_terms(pred,b)
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(norm) and norm>0
                if arm=='adapt_last2':assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith('aggregator.') and p.grad is not None)>0
                optimizer.step();step+=1;status('preflight_updates',loss=float(loss),gradient_norm=float(norm))
                metrics=metric_rows(pred,b)
                assert all(math.isfinite(r['rotation_deg']) for r in metrics if 'rotation_deg' in r)
            status('passed',peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
            write(dest/'passed.json',dict(passed=True,steps=step,utc=utc(),protocol_sha256=sha(OUT/'protocol.json')))
            return
        assert read(OUT/f'preflight_{arm}_seed{seed}/passed.json')['passed']
        status('baseline_validation')
        evaluate(model,manifest,dest/'validation_step000000.json',deadline)
        with (dest/'train.jsonl').open('x',buffering=1) as log:
            for item in work:
                if STOP_REQUESTED or (OUT/'STOP').exists() or time.monotonic()>deadline:break
                r=manifest['records'][item['index']];b=on_device(load_item(r,manifest,item['views']))
                optimizer.zero_grad(set_to_none=True);pred=model(b['images']);loss,terms=loss_terms(pred,b)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                optimizer.step();step+=1
                exposures[r['kind']]+=1;unique.add((item['index'],tuple(item['views'])))
                row=dict(utc=utc(),step=step,kind=r['kind'],group=r['group'],record_index=item['index'],views=item['views'],loss=float(loss),gradient_norm=float(norm),**{k:float(v) for k,v in terms.items()})
                log.write(json.dumps(row)+'\n')
                if step<=3 or step%10==0:status('training',last=row,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3);print(json.dumps(row),flush=True)
                if step%500==0:
                    save_checkpoint(dest/'latest.pt',model,optimizer,step,exposures,protocol)
                if step==250 or step%1000==0:
                    status('validation');summary=evaluate(model,manifest,dest/f'validation_step{step:06d}.json',deadline);status('training',validation=summary)
        status('saving');save_checkpoint(dest/'latest.pt',model,optimizer,step,exposures,protocol)
        if not STOP_REQUESTED and not (OUT/'STOP').exists() and time.monotonic()<deadline:
            evaluate(model,manifest,dest/f'validation_final_step{step:06d}.json',deadline)
        status('completed_budget' if step<steps else 'completed',promotion='Not promoted; matched control and geometry transfer gate still required')
    except BaseException as exc:
        if model is not None and optimizer is not None and step and not preflight:
            save_checkpoint(dest/'interrupted.pt',model,optimizer,step,exposures,protocol)
        status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','train'])
    parser.add_argument('--arm',choices=['adapt_last2','frozen_encoder_control'],default='adapt_last2')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--steps',type=int,default=24000);parser.add_argument('--hours',type=float,default=12)
    args=parser.parse_args()
    def stopped(*_):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    if args.stage=='prepare':prepare()
    else:run(args.arm,args.seed,min(args.steps,4) if args.stage=='preflight' else args.steps,args.hours,args.stage=='preflight')


if __name__=='__main__':main()
