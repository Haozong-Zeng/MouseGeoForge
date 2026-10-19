"""Fresh-release pure geometry adaptation; no keypoint head or keypoint gradients.

PyMouseStudio is one known rig. Its manual 2D correspondences constrain depth
through supplied cameras; predicted cameras there are diagnostic, not a gate.
Unseen-rig gates use independently rendered camera configurations.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import bottom_domain_adaptation_20261018 as old
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera,extri_intri_to_pose_encoding

OUT=ROOT/'output/pure_geometry_bottom_20261018_v2'
STOP=False
TERMS=('camera','depth','shape','transfer','real_sparse')


def write(path,obj):old.write(path,obj)
def read(path):return old.read(path)
def sha(path):return old.sha(path)
def utc():return old.utc()


def prepare():
    assert not (OUT/'manifest.json').exists()
    render_cfg=read(OUT/'render_config.json');render_dir=ROOT/render_cfg['output']
    rendered=read(render_dir/'completion.json')
    source=read(old.OUT/'manifest.json');records=[]
    for row in source['records']:
        if row['kind']=='real_video':continue
        if row['kind']=='real_external' and row['split']!='train':continue
        row=dict(row)
        row['evaluation_role']='known_rig_diagnostic' if row['kind']=='real_labeled' else 'motion_holdout_not_unseen_rig'
        records.append(row)
    for row in rendered['records']:
        row=dict(row,evaluation_role='unseen_rig_gate' if row['split']=='val' else 'randomized_rig_train')
        row['scene']=f"{render_cfg['output']}/{row['rig']}/frame_{row['frame']:06d}"
        records.append(row)
    train_rigs={r['rig'] for r in records if r.get('rig','').startswith('train_rig')}
    val_rigs={r['rig'] for r in records if r.get('rig','').startswith('val_rig')}
    assert len(train_rigs)==6 and len(val_rigs)==2 and not train_rigs&val_rigs
    # Configuration identity is invariant to world gauge and global length scale.
    def rig_fingerprint(views):
        E=np.array([v['E'] for v in views]);scale=old.median_camera_baseline(E)
        normalized=E@np.linalg.inv(E[0]);normalized[:,:3,3]/=scale
        return hashlib.sha256(np.round(normalized,5).tobytes()).hexdigest()
    train_sig={rig_fingerprint(r['views']) for r in rendered['records'] if r['split']=='train'}
    val_sig={rig_fingerprint(r['views']) for r in rendered['records'] if r['split']=='val'}
    assert not train_sig&val_sig
    counts=collections.Counter((r['split'],r['kind'],r['evaluation_role']) for r in records)
    write(OUT/'manifest.json',dict(records=records,rigs=source['rigs'],counts=[dict(split=k[0],kind=k[1],role=k[2],count=v) for k,v in counts.items()],
                                 train_rigs=sorted(train_rigs),heldout_rigs=sorted(val_rigs),train_rig_fingerprints=sorted(train_sig),val_rig_fingerprints=sorted(val_sig)))
    base=read(old.OUT/'protocol.json')
    pins=dict(base['source_pins']);pins.update(read(OUT/'rig_data_qa.json')['source_pins'])
    write(OUT/'protocol.json',dict(utc=utc(),authorization='User: save/stop joint training; restart pure geometry from release, rig-separated validation, gradient accumulation and task-gradient measurement, warmup/cosine and EMA.',
        manifest_sha256=sha(OUT/'manifest.json'),code_sha256=sha(__file__),loader_sha256=sha(old.__file__),
        source_manifest_sha256=sha(old.OUT/'manifest.json'),render_completion_sha256=sha(render_dir/'completion.json'),source_pins=pins,
        checkpoint=base['checkpoint'],checkpoint_sha256=base['checkpoint_sha256'],seed=17,
        initialization='Fresh exact released aggregator/camera/depth; no KeypointHead instantiated; no old joint weights/optimizer.',
        deployment='Known PyMouseStudio rig: use supplied K/E directly; camera prediction diagnostic only. Unseen-rig camera gates apply exclusively to held-out rendered physical rigs.',
        real_supervision='No PyMouseStudio camera-only video updates (0%). Manual known 2D correspondences constrain dense depth through supplied K/E; weak calibrated reprojection, not real depth/3D GT, not keypoint prediction loss. DANNCE1 supplies a separate known camera rig, with no pseudo depth.',
        gauge=dict(pymousestudio_full_calibration_baseline_mm=known_rig_scale(),rule='One fixed full-calibration rig baseline across all real captures and missing-camera subsets; both canonical camera translation and predicted depth use that same gauge. Synthetic/DANNCE use each full rig; single-view replay uses rendered medianZ.'),
        sampling='Eight micros per update:2 new random rig,1 existing bottom,1 non-water synthetic replay,2 corrected real labeled groups,1 DANNCE1 camera every second update (otherwise replay),1 retention (water every fourth update, otherwise non-water). Real labels25% from update1; PyMouseStudio video0%; all camera-only real/DANNCE micros6.25%; water3.125%. Group-balanced shuffled frame pools.',
        optimizer=dict(type='AdamW',head_lr=1e-5,encoder_lr=1e-6,warmup_steps=500,schedule='cosine to10% peak',clip_norm=10.,accumulation=8,weight_decay=.01),
        balance='Measure each active task gradient norm on all shared trainable encoder parameters at release initialization. Freeze reciprocal median-norm weights clipped [.01,100]. Log per-term raw/weighted norms and pairwise gradient cosine every100 updates; do not infer influence from loss values.',
        camera_objective='atan2 geodesic relative rotation/radians plus0.05 chordal stabilization; translation direction, canonical translation, baseline and FoV; no PyMouseStudio predicted camera supervision.',
        geometry_objective='Synthetic camera-Z log-Huber, body-centered shape, and joint predicted-camera+depth transfer on fixed GT co-visible support; real sparse known-camera reprojection. No learned keypoint objective.',
        ema=dict(decay=.999,warm_start='min(.999,(step+1)/(step+10))',validation='EMA primary; paired raw evaluation at final; no best-checkpoint selection'),
        budget=dict(updates=4000,microbatches=32000,max_hours=8,evaluation_every=500,checkpoint_every=250),
        validation='All6 unseen-rig scenes, plus fixed8 per old synthetic held-out source and all71 real labeled held-out groups. Real rig memorization not a promotion criterion; old source holdout not called unseen rig.',
        gates=dict(unseen_rig='rotation median/P95 improves>=30%, absolute P95<=10deg and direction P95<=15deg; predicted-camera+depth transfer coverage32 improves>=5pp',
                   known_rig='supplied-camera depth transfer and sparse annotated-ray reprojection improve; no metric3D claim',
                   retention='other-view transfer/shape and per-source errors must not regress; no promotion with side-view collapse',
                   keypoints='Deferred: frozen-encoder detached2D probe warmup with lr1e-4..3e-4 to convergence; subsequent adaptation comparison starts from same converged head. No keypoint backprop into geometry during this stage.'),
        limitations=['Two held-out rigs and same mouse assets are a feasibility evaluation, not broad unseen-rig/animal certification.',
                     'Geodesic rotation has no globally unique derivative at exactly180deg; test numerical behavior near the antipode and keep a small chordal term.',
                     'Global clipping does not make AdamW updates equal. Depth improvement attribution requires gradient/control evidence, not loss magnitudes.',
                     'Current synthetic RGB includes transparent glass, while geometric truth explicitly ignores glass/hair; retain this provenance.']))
    print(json.dumps(read(OUT/'manifest.json')['counts']),flush=True)


@lru_cache(maxsize=1)
def known_rig_scale():
    cal=read(ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/AZL3/rectified_calibration.json')
    E=np.array([v['world_to_camera_mm'] for v in cal.values() if isinstance(v,dict) and 'world_to_camera_mm'in v])
    assert len(E)==7
    return float(old.median_camera_baseline(E))


def load_item(rec,manifest,pack):
    b=old.load_item(rec,manifest,pack)
    views=old.views_for(rec,manifest)
    if rec['kind']=='real_labeled':
        observed_scale=old.median_camera_baseline(np.array([v['E'] for v in views]))
        b['E'][...,:3,3]*=observed_scale/known_rig_scale()
    if any(v.get('truth') for v in views):
        scale=old.median_camera_baseline(np.array([v['E'] for v in views]))
        for j,vid in enumerate(pack):
            v=next(v for v in views if v['id']==vid)
            with np.load(ROOT/v['truth']) as a:
                z=old.resized(Image.fromarray(a['depth'].astype(np.float32)),512,True)[0]/scale
                mask=old.resized(Image.fromarray(a['body'].astype(np.uint8)*255),512,True)[0]>0
                valid=old.resized(Image.fromarray(a['valid'].astype(np.uint8)*255),512,True)[0]>0
                # New truth uses pixel-array coordinates explicitly. PIL resizing
                # maps center x to (x+.5)*scale-.5, unlike an edge-coordinate K.
                h,w=a['depth'].shape
                b['K'][0,j,0,2]+=(512/w-1)/2
                b['K'][0,j,1,2]+=(512/h-1)/2
            b['depth'][0,j]=torch.tensor(z);b['body'][0,j]=torch.tensor(mask);b['valid'][0,j]=torch.tensor(valid)
    # Single-view replay has no baseline. Use rendered median-Z as the shared
    # depth gauge; it cannot evaluate multiview camera generalization.
    if len(views)==1 and b['valid'].any():
        s=b['depth'][b['valid']].median().clamp_min(1e-6);b['depth']/=s
    return old.on_device(b)


def rotation_angle(error):
    vee=torch.stack((error[...,2,1]-error[...,1,2],error[...,0,2]-error[...,2,0],error[...,1,0]-error[...,0,1]),-1)
    sine=.5*torch.linalg.vector_norm(vee,dim=-1)
    cosine=((error.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    return torch.atan2(sine,cosine)


def camera_loss(pred,b):
    target=extri_intri_to_pose_encoding(b['E'],b['K'],(512,512));pe,_=encoding_to_camera(pred['pose_enc'],(512,512))
    pR=pe[:,1:,:3,:3]@pe[:,:1,:3,:3].transpose(-1,-2);tR=b['E'][:,1:,:3,:3]@b['E'][:,:1,:3,:3].transpose(-1,-2)
    pt=pe[:,1:,:3,3]-(pR@pe[:,:1,:3,3,None])[...,0];tt=b['E'][:,1:,:3,3]-(tR@b['E'][:,:1,:3,3,None])[...,0]
    absolute=F.smooth_l1_loss(pred['pose_enc'][...,:3],target[...,:3])
    fov=F.smooth_l1_loss(pred['pose_enc'][...,7:],target[...,7:])
    anchor=rotation_angle(pe[:,:1,:3,:3]@b['E'][:,:1,:3,:3].transpose(-1,-2)).mean()
    if pe.shape[1]<2:return absolute+.1*fov+.1*anchor
    angle=rotation_angle(pR@tR.transpose(-1,-2)).mean()
    direction=(F.normalize(pt,dim=-1)-F.normalize(tt,dim=-1)).square().sum(-1).mean()
    scale=F.smooth_l1_loss(pt.norm(dim=-1).clamp_min(1e-5).log(),tt.norm(dim=-1).clamp_min(1e-5).log())
    return angle+.05*(pR-tR).square().mean()+.25*direction+.25*scale+.25*absolute+.1*fov+.1*anchor


def transfer_data(pred,b,source,known_cameras=False):
    side=512;device=b['images'].device
    yy,xx=torch.meshgrid(torch.arange(8,side,16,device=device),torch.arange(8,side,16,device=device),indexing='ij')
    x,y=xx.flatten(),yy.flatten();support=b['valid'][0,source,y,x]&b['body'][0,source,y,x]
    if not support.any():return []
    x,y=x[support],y[support];uv=torch.stack((x,y,torch.ones_like(x)),0).float()
    gtcam=torch.linalg.solve(b['K'][0,source],uv)*b['depth'][0,source,y,x][None];E=b['E'][0,source]
    world=torch.linalg.solve(E[:3,:3],gtcam-E[:3,3,None])
    pe,pk=(b['E'],b['K']) if known_cameras else encoding_to_camera(pred['pose_enc'],(512,512))
    ray=torch.linalg.solve(pk[0,source],uv);pcam=ray*pred['depth'][0,source,y,x,0][None];E=pe[0,source]
    pworld=torch.linalg.solve(E[:3,:3],pcam-E[:3,3,None]);pairs=[]
    for target in range(len(b['roles'])):
        if target==source:continue
        te=b['E'][0,target];tc=te[:3,:3]@world+te[:3,3,None];hom=b['K'][0,target]@tc;expected=hom[:2]/hom[2:]
        inside=torch.isfinite(expected).all(0)&(tc[2]>0)&(expected>=0).all(0)&(expected<511).all(0)
        u,v=expected.nan_to_num().round().long().clamp(0,511);tz=b['depth'][0,target,v,u]
        good=inside&b['valid'][0,target,v,u]&((tz-tc[2]).abs()<.01*tz+.001)
        if not good.any():continue
        actualcam=pe[0,target,:3,:3]@pworld+pe[0,target,:3,3,None];ph=pk[0,target]@actualcam
        actual=ph[:2]/ph[2:].clamp_min(.01)
        pairs.append(dict(delta=(actual[:,good]-expected[:,good]).T/512,z=actualcam[2,good],
                          role='bottom' if 'bottom'in(b['roles'][source],b['roles'][target]) else 'other'))
    return pairs


def sparse_data(pred,b):
    result=[];depth=pred['depth'][...,0]
    for source in range(len(b['roles'])):
        xy=b['xy'][0,source];uv=torch.cat((xy,torch.ones_like(xy[:,:1])),-1).T
        sampled=F.grid_sample(depth[0,source][None,None],((xy+.5)/512*2-1)[None,None],align_corners=False,padding_mode='border').reshape(-1)
        camera=torch.linalg.solve(b['K'][0,source],uv)*sampled[None];E=b['E'][0,source]
        world=torch.linalg.solve(E[:3,:3],camera-E[:3,3,None])
        for target in range(len(b['roles'])):
            if source==target:continue
            good=b['known'][0,source]&b['known'][0,target]
            if not good.any():continue
            E=b['E'][0,target];camera=E[:3,:3]@world+E[:3,3,None];hom=b['K'][0,target]@camera
            uvpred=(hom[:2]/hom[2:].clamp_min(.01)).T
            result.append(dict(delta=(uvpred[good]-b['xy'][0,target,good])/512,z=camera[2,good],
                               role='bottom' if 'bottom'in(b['roles'][source],b['roles'][target]) else 'other'))
    return result


def projection_loss(pairs):
    return torch.stack([F.smooth_l1_loss(p['delta'],torch.zeros_like(p['delta']),beta=.03)+F.relu(.01-p['z']).mean() for p in pairs]).mean() if pairs else None


def terms(pred,b):
    result={};rec=b['record']
    if rec['kind']!='real_labeled':result['camera']=camera_loss(pred,b)
    if b['valid'].any():
        z=pred['depth'][...,0].clamp_min(1e-6);target=b['depth'].clamp_min(1e-6);residual=z.log()-target.log()
        masks=[b['valid']&b['body'],b['valid']&~b['body']]
        result['depth']=torch.stack([F.smooth_l1_loss(residual[m],torch.zeros_like(residual[m])) for m in masks if m.any()]).mean()
        shapes=[]
        for v in range(z.shape[1]):
            m=b['valid'][0,v]&b['body'][0,v]
            if m.sum()>16:
                p=z[0,v][m];t=target[0,v][m];shapes.append(F.smooth_l1_loss(p-p.mean(),t-t.mean(),beta=.01))
        if shapes:result['shape']=torch.stack(shapes).mean()
        pairs=[p for s in range(len(b['roles'])) for p in transfer_data(pred,b,s)]
        loss=projection_loss(pairs)
        if loss is not None:result['transfer']=loss
    if rec['kind']=='real_labeled':
        loss=projection_loss(sparse_data(pred,b))
        if loss is None:raise ValueError('Real pack has no valid cross-view manual correspondences')
        result['real_sparse']=loss
    assert result
    return result


def make_model():
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    m=VGGTOmega(enable_camera=True,enable_depth=True)
    state=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    m.load_state_dict({k:v for k,v in state.items() if k.startswith(('aggregator.','camera_head.','dense_head.'))},strict=True);del state
    assert m.keypoint_head is None;m.requires_grad_(False)
    for module in (m.camera_head,m.dense_head):module.requires_grad_(True)
    m.dense_head.proj_conf.requires_grad_(False)
    for module in list(m.aggregator.frame_blocks[-2:])+list(m.aggregator.inter_frame_blocks[-2:]):module.requires_grad_(True)
    m=m.cuda().train()
    shared=[p for n,p in m.named_parameters() if n.startswith('aggregator.') and p.requires_grad]
    heads=[p for n,p in m.named_parameters() if not n.startswith('aggregator.') and p.requires_grad]
    opt=torch.optim.AdamW([dict(params=heads,lr=1e-5,base_lr=1e-5),dict(params=shared,lr=1e-6,base_lr=1e-6)],weight_decay=.01,foreach=False)
    return m,opt,shared


def schedule(manifest,updates):
    pools=collections.defaultdict(list)
    for i,r in enumerate(manifest['records']):
        if r['split']!='train':continue
        if r.get('evaluation_role')=='randomized_rig_train':kind='new_rig'
        elif r['kind']=='real_labeled':kind='real'
        elif r['kind']=='real_external':kind='dannce'
        elif 'water_reaching' in r['group']:kind='water'
        elif any(v['physical_role']=='bottom' for v in old.views_for(r,manifest)):kind='bottom'
        else:kind='replay'
        pools[kind].append(i)
    assert set(pools)=={'new_rig','real','dannce','water','bottom','replay'}
    rng=random.Random(20261018);groups={};orders={};position=collections.Counter();visits=collections.Counter();group_pos=collections.Counter()
    for kind,ii in pools.items():
        g=collections.defaultdict(list)
        for i in ii:g[manifest['records'][i]['group']].append(i)
        groups[kind]=g
    result=[]
    for update in range(updates):
        kinds=['new_rig','new_rig','bottom','replay','real','real','dannce' if update%2==0 else 'replay','water' if update%4==3 else 'replay'];micros=[]
        for kind in kinds:
            gg=sorted(groups[kind]);group=gg[group_pos[kind]%len(gg)];group_pos[kind]+=1;key=(kind,group)
            pool=groups[kind][group]
            if position[key]%len(pool)==0:orders[key]=pool.copy();rng.shuffle(orders[key]);position[key]=0
            i=orders[key][position[key]];position[key]+=1;rec=manifest['records'][i]
            choices=old.packs(old.views_for(rec,manifest));pack=choices[visits[i]%len(choices)];visits[i]+=1
            if len(pack)>1 and (update+len(micros))%2:pack=pack[1:]+pack[:1]
            micros.append(dict(index=i,views=pack,sampling_role=kind))
        result.append(micros)
    return result


def task_gradients(losses,shared):
    vectors={};norms={}
    for key,value in losses.items():
        grads=torch.autograd.grad(value,shared,retain_graph=True,allow_unused=True)
        vec=[g.detach().float() if g is not None else None for g in grads];vectors[key]=vec
        norms[key]=float(torch.sqrt(sum(g.square().sum() for g in vec if g is not None)))
    cosines={}
    for i,a in enumerate(vectors):
        for b in list(vectors)[i+1:]:
            dot=sum((x*y).sum() for x,y in zip(vectors[a],vectors[b]) if x is not None and y is not None)
            cosines[a+'/'+b]=float(dot)/(max(norms[a]*norms[b],1e-20))
    return norms,cosines


def gradient_balance(model,manifest,work,shared):
    samples=collections.defaultdict(list);rows=[]
    for item in work[0]+work[1]:
        b=load_item(manifest['records'][item['index']],manifest,item['views']);pred=model(b['images']);losses=terms(pred,b)
        norms,cosines=task_gradients(losses,shared)
        for k,v in norms.items():
            if v>1e-12 and math.isfinite(v):samples[k].append(v)
        rows.append(dict(kind=b['record']['kind'],group=b['record']['group'],norms=norms,cosines=cosines))
        del pred,losses,b
    assert set(samples)==set(TERMS),set(samples)
    med={k:float(np.median(v)) for k,v in samples.items()};weights={k:float(np.clip(1/v,.01,100)) for k,v in med.items()}
    return dict(weights=weights,median_encoder_gradient_norm=med,samples=rows,method='Reciprocal initial median shared-encoder gradient norm; target1; clipped[.01,100], frozen after calibration; no validation tuning')


def pair_stats(pairs):
    if not pairs:return dict(support=0,coverage32=None,p95=None,invalid=0,errors_ref=[])
    errors=[];invalid=0
    for p in pairs:
        e=(p['delta']*torch.tensor([704,768],device=p['delta'].device)).norm(dim=-1)
        valid=torch.isfinite(e)&(p['z']>0);invalid+=int((~valid).sum());errors.extend(e[valid].tolist())
    n=len(errors)+invalid;ordered=sorted(errors);rank=math.ceil(.95*n)-1
    return dict(support=n,coverage32=sum(e<=32 for e in errors)/n,p95=ordered[rank] if rank<len(ordered) else None,invalid=invalid,errors_ref=errors+[None]*invalid)


@torch.no_grad()
def evaluate(model,manifest,dest):
    model.eval();rows=[]
    groups=collections.defaultdict(list)
    for i,r in enumerate(manifest['records']):
        if r['split']=='val':groups[r['evaluation_role'],r['group']].append(i)
    indices=[]
    for (role,group),ii in groups.items():
        n=len(ii) if role in ('known_rig_diagnostic','unseen_rig_gate') else min(8,len(ii))
        indices.extend(ii[j] for j in sorted(set(np.linspace(0,len(ii)-1,n).round().astype(int))))
    for count,i in enumerate(indices):
        if STOP or (OUT/'STOP').exists():raise InterruptedError('Stopped during evaluation')
        rec=manifest['records'][i]
        for pack in old.packs(old.views_for(rec,manifest)):
            if len(pack)>1:pack=pack[1:]+pack[:1]
            b=load_item(rec,manifest,pack);pred=model(b['images']);pe,pk=encoding_to_camera(pred['pose_enc'],(512,512));row=dict(group=rec['group'],rig=rec.get('rig'),evaluation_role=rec['evaluation_role'],frame=rec.get('frame'),scene=rec.get('scene'),context=pack,cameras=[])
            for v,role in enumerate(b['roles']):
                item=dict(role=role,view=pack[v])
                if v:
                    pr=pe[0,v,:3,:3]@pe[0,0,:3,:3].T;tr=b['E'][0,v,:3,:3]@b['E'][0,0,:3,:3].T
                    pt=pe[0,v,:3,3]-pr@pe[0,0,:3,3];tt=b['E'][0,v,:3,3]-tr@b['E'][0,0,:3,3]
                    item.update(rotation_deg=float(rotation_angle(pr@tr.T)*180/math.pi),direction_deg=float(torch.acos((F.normalize(pt,dim=0)*F.normalize(tt,dim=0)).sum().clamp(-1,1))*180/math.pi),pair_role='bottom' if 'bottom'in(role,b['roles'][0]) else 'other')
                m=b['valid'][0,v]&b['body'][0,v]
                if m.any():
                    z=pred['depth'][0,v,...,0][m];target=b['depth'][0,v][m]
                    item.update(body_absrel=float(((z-target).abs()/target.clamp_min(1e-6)).mean()),centered_shape_rmse=float(((z-z.mean())-(target-target.mean())).square().mean().sqrt()))
                row['cameras'].append(item)
            for label,known in [('predicted_camera_depth',False),('known_camera_depth',True)]:
                pp=[p for s in range(len(pack)) for p in transfer_data(pred,b,s,known)] if b['valid'].any() else []
                row[label]={role:pair_stats([p for p in pp if p['role']==role]) for role in ['bottom','other']}
            if rec['kind']=='real_labeled':
                pp=sparse_data(pred,b);row['known_camera_sparse']={role:pair_stats([p for p in pp if p['role']==role]) for role in ['bottom','other']}
            rows.append(row)
        if count%20==0:print('EVAL',count,len(indices),flush=True)
    summary={}
    for erole in sorted({r['evaluation_role'] for r in rows}):
        selected=[r for r in rows if r['evaluation_role']==erole];s={}
        for role in ['bottom','other']:
            cams=[c for r in selected for c in r['cameras'] if c.get('pair_role')==role]
            def stats(values):
                return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=sorted(values)[math.ceil(.95*len(values))-1]) if values else None
            metrics=dict(rotation_deg=stats([c['rotation_deg'] for c in cams]),direction_deg=stats([c['direction_deg'] for c in cams]))
            for key in ['body_absrel','centered_shape_rmse']:
                metrics[key]=stats([c[key] for r in selected for c in r['cameras'] if (c['role']=='bottom')==(role=='bottom') and key in c])
            for key in ['predicted_camera_depth','known_camera_depth','known_camera_sparse']:
                data=[r[key][role] for r in selected if key in r];n=sum(d['support'] for d in data)
                ordered=sorted(e for d in data for e in d['errors_ref'] if e is not None);rank=math.ceil(.95*n)-1
                metrics[key]=dict(support=n,coverage32=sum(d['coverage32']*d['support'] for d in data if d['support'])/n if n else None,invalid=sum(d['invalid'] for d in data),p95=ordered[rank] if n and rank<len(ordered) else None)
            s[role]=metrics
        summary[erole]=s
    per_source={}
    for name in sorted({r['rig'] or r['group'] for r in rows}):
        rr=[r for r in rows if (r['rig'] or r['group'])==name];per_source[name]={}
        for role in ['bottom','other']:
            angles=[c['rotation_deg'] for r in rr for c in r['cameras'] if c.get('pair_role')==role]
            data=[r['predicted_camera_depth'][role] for r in rr];n=sum(d['support'] for d in data)
            per_source[name][role]=dict(camera_pairs=len(angles),rotation_mean=float(np.mean(angles)) if angles else None,
                rotation_p95=sorted(angles)[math.ceil(.95*len(angles))-1] if angles else None,
                transfer_support=n,transfer_coverage32=sum(d['coverage32']*d['support'] for d in data if d['support'])/n if n else None)
    summary['per_source']=per_source
    write(dest,dict(utc=utc(),summary=summary,rows=rows));model.train();return summary


def checkpoint(path,model,opt,ema,step,weights,exposures):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},ema={n:p.cpu() for n,p in ema.items()},optimizer=opt.state_dict(),step=step,loss_weights=weights,exposures=dict(exposures),rng=old.rng_state(),protocol_sha256=sha(OUT/'protocol.json'),base_sha256=read(OUT/'protocol.json')['checkpoint_sha256']),tmp)
    tmp.replace(path)


def fixtures():
    results=[]
    for degrees in [0,90,179,179.9,180]:
        angle=torch.tensor(degrees*math.pi/180,requires_grad=True);c,s=angle.cos(),angle.sin();zero=c*0;one=zero+1
        R=torch.stack((c,-s,zero,s,c,zero,zero,zero,one)).reshape(3,3);loss=rotation_angle(R);loss.backward()
        assert abs(float(loss.detach())-degrees*math.pi/180)<2e-6 and torch.isfinite(angle.grad)
        if degrees==179.9:assert abs(float(angle.grad))>.5
        results.append(dict(degrees=degrees,gradient=float(angle.grad)))
    return dict(passed=True,geodesic=results)


def train(preflight=False):
    torch.set_num_threads(4);old.cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=read(OUT/'protocol.json');manifest=read(OUT/'manifest.json')
    assert sha(__file__)==protocol['code_sha256'] and sha(old.__file__)==protocol['loader_sha256']
    assert sha(OUT/'manifest.json')==protocol['manifest_sha256']
    for path,digest in protocol['source_pins'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    dest=OUT/('preflight' if preflight else 'geometry_seed17');dest.mkdir(exist_ok=False)
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();step=0;model=opt=ema=None;weights={};exposures=collections.Counter()
    def status(state,**extra):write(dest/'status.json',dict(utc=utc(),state=state,pid=os.getpid(),step=step,elapsed_s=time.monotonic()-began,exposures=dict(exposures),**extra))
    try:
        write(dest/'fixtures.json',fixtures());status('loading_release')
        model,opt,shared=make_model();work=schedule(manifest,2 if preflight else protocol['budget']['updates']);write(dest/'schedule.json',work)
        write(dest/'trainable.json',dict(names=[n for n,p in model.named_parameters() if p.requires_grad],keypoint_head_absent=model.keypoint_head is None))
        status('gradient_calibration');balance=gradient_balance(model,manifest,work,shared);write(dest/'gradient_calibration.json',balance);weights=balance['weights']
        # Each run measures the same release-initialization calibration, never the preflight's updated state.
        if not preflight:
            assert read(OUT/'preflight/passed.json')['passed'];status('release_baseline_validation');evaluate(model,manifest,dest/'release_baseline.json')
        ema={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        params=dict(model.named_parameters())
        def validate_ema(name):
            backup={n:params[n].detach().clone() for n in ema}
            with torch.no_grad():
                for n,v in ema.items():params[n].copy_(v)
            result=evaluate(model,manifest,dest/name)
            with torch.no_grad():
                for n,v in backup.items():params[n].copy_(v)
            return result
        clipped=0
        with (dest/'train.jsonl').open('x',buffering=1) as log:
            for micros in work:
                if STOP or (OUT/'STOP').exists() or time.monotonic()-began>protocol['budget']['max_hours']*3600:break
                opt.zero_grad(set_to_none=True);totals=collections.Counter();details=[]
                warm=min(1.,(step+1)/500);progress=max(0.,(step-500)/max(1,protocol['budget']['updates']-500));factor=warm*(.1+.9*(1+math.cos(math.pi*min(progress,1)))/2)
                for pg in opt.param_groups:pg['lr']=pg['base_lr']*factor
                for item in micros:
                    b=load_item(manifest['records'][item['index']],manifest,item['views']);pred=model(b['images']);losses=terms(pred,b)
                    if step%100==0:
                        norms,cosines=task_gradients(losses,shared);details.append(dict(role=item['sampling_role'],raw_gradient_norms=norms,weighted_gradient_norms={k:v*weights[k] for k,v in norms.items()},cosines=cosines))
                    loss=sum(weights[k]*v for k,v in losses.items())/len(micros)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mixed objective')
                    loss.backward();exposures[item['sampling_role']]+=1
                    for k,v in losses.items():totals[k]+=float(v.detach())/len(micros)
                    del pred,losses,loss,b
                norm=torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],10.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite accumulated gradient')
                clipped+=int(norm>10);opt.step();step+=1
                decay=min(.999,(step+1)/(step+10))
                with torch.no_grad():
                    for n,value in ema.items():value.lerp_(params[n].detach(),1-decay)
                row=dict(utc=utc(),step=step,losses=dict(totals),gradient_norm=float(norm),clip_fraction=clipped/step,lr_factor=factor,ema_decay=decay,gradient_details=details)
                log.write(json.dumps(row)+'\n')
                if step<=3 or step%10==0:status('training',last=row,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3);print(json.dumps(row),flush=True)
                if not preflight and step%250==0:checkpoint(dest/'latest.pt',model,opt,ema,step,weights,exposures)
                if not preflight and step%500==0:status('ema_validation');validate_ema(f'ema_step{step:06d}.json')
        if preflight:
            assert step==2;write(dest/'passed.json',dict(passed=True,step=step,keypoint_head_absent=True,gradient_accumulation=8,protocol_sha256=sha(OUT/'protocol.json')));status('passed')
        else:
            status('saving');checkpoint(dest/'latest.pt',model,opt,ema,step,weights,exposures)
            if not STOP and not (OUT/'STOP').exists():
                validate_ema(f'ema_final_step{step:06d}.json');evaluate(model,manifest,dest/f'raw_final_step{step:06d}.json')
            status('completed' if step==protocol['budget']['updates'] else 'stopped_or_budget',promotion='None; inspect unseen-rig and known-camera geometry/retention before detached probe or3D work')
    except BaseException as exc:
        if model is not None and ema is not None and step and not preflight:checkpoint(dest/'interrupted.pt',model,opt,ema,step,weights,exposures)
        status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    global STOP
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','train']);args=p.parse_args()
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage=='prepare':prepare()
    else:train(args.stage=='preflight')


if __name__=='__main__':main()
