"""Free-moving-only manifest, exact ordered sampling, and CPU worker loading."""
from __future__ import annotations
import collections
import hashlib
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import pure_geometry_bottom_20261018 as geo
import geometry_prefetch_20261018 as io
import bottom_geometry_v3_losses_20261018 as losses

OUT=geo.ROOT/'output/bottom_geometry_v3_local12h_20261018'
FREE_GROUPS={'moseq_sample10_all_correlated_variants','mouse_climb_wall_91','mouse_wash_face_91',
    'crim13/021709_A25_Block22_BCfe1_t','crim13/021309_A29_Block16_BCma1_t'}


def admit_dannce(records):
    session=geo.ROOT/'dataset/DANNCE-MarkerlessMouse1-Core23-RealMultiview/sessions/markerless_mouse_1'
    metadata=geo.read(session/'metadata.json');cfg=geo.read(session/'pseudo_depth_full_v1/config.json')
    assert metadata['recommended_split']=='train' and metadata['depth_complete']
    assert cfg['depth_unit']=='mm' and cfg['depth_semantics']=='camera_z' and cfg['pfm_flip_y']
    assert cfg['minimum_views']==6
    rig=geo.read(geo.OUT/'manifest.json')['rigs']['dannce/1']
    assert len(rig)==len(metadata['cameras'])==6
    for v,camera in zip(rig,metadata['cameras']):
        np.testing.assert_allclose(v['K'],camera['intrinsic'],atol=1e-4)
        np.testing.assert_allclose(v['E'],camera['extrinsic'],atol=1e-4)
    import json
    frames={x['source_frame_id']:x for x in map(json.loads,(session/'frames.jsonl').read_text().splitlines())}
    assert len(frames)==4500
    count=0
    for r in records:
        if r['kind']!='real_external':continue
        assert r['split']=='train' and r['group']=='dannce_mouse_1'
        frame=int(geo.Path(r['rgb'][0]).stem);row=frames[frame]
        assert row['split']=='train' and row['pseudo_depth_training_valid'] and not row['pseudo_quality_warnings']
        assert row['depth_is_pseudo_label'] and row['depth_unit']=='mm'
        for a,b in zip(r['rgb'],row['rgb']):assert a==(session/b).relative_to(geo.ROOT).as_posix()
        r['pseudo_depth']=[(session/p).relative_to(geo.ROOT).as_posix() for p in row['depth']]
        r['pseudo_mask']=[(session/p).relative_to(geo.ROOT).as_posix() for p in row['mask']]
        r['pseudo_confidence']=[(session/p).relative_to(geo.ROOT).as_posix() for p in row['confidence']]
        r['depth_is_pseudo_label']=True;r['depth_method']=cfg['method'];r['source_frame_id']=frame
        count+=1
    assert count==4500
    geo.write(OUT/'extra_data_decision.json',dict(utc=geo.utc(),state='DANNCE1 foreground pseudo depth admitted per explicit user clarification',
        frames=count,views=count*6,method=cfg['method'],minimum_confidence=212,
        confidence_semantics=cfg['confidence_semantics'],depth_unit='mm',depth_semantics='camera_z',
        proxy_geometry_weight=.2,background_invalid=True,mask_boundary_erosion_px=1,
        DANNCE2_opened=False,PyMouseStudio_visual_hull_used=False,
        GS_lineage_note='Full published payload is SAM2 plus calibrated6of6 visual hull postprocessing; not raw Gaussian expected-depth or independent GT.',
        source_index_sha256=geo.sha(session/'frames.jsonl'),source_config_sha256=geo.sha(session/'pseudo_depth_full_v1/config.json')))
    return records


def prepare():
    assert not (OUT/'manifest.json').exists()
    original=geo.read(geo.OUT/'manifest.json');records=[];excluded=collections.Counter()
    for rec in original['records']:
        allowed=rec['kind'] in ('real_labeled','real_external') or rec['group'] in FREE_GROUPS
        if not allowed:excluded[rec['group']]+=1;continue
        if rec.get('evaluation_role')=='randomized_rig_train':continue
        rec=dict(rec)
        rec['evaluation_role']='known_rig_diagnostic' if rec['kind']=='real_labeled' else 'free_motion_replay'
        records.append(rec)
    new=[]
    for shard in (0,1):
        completion=geo.read(OUT/f'render_shard{shard}/completion.json')
        assert not completion['rejected']
        assert completion['config_sha256']==geo.sha(OUT/f'render_config_shard{shard}.json')
        assert completion['script_sha256']==geo.sha(geo.ROOT/'BlenderScripts/render_bottom_geometry_v3_cycle_20261018.py')
        for r in completion['records']:
            assert r['group'] in FREE_GROUPS and not r.get('apparatus_ablation',False)
            assert all(v['passed'] and v['foreground_pixels']>=100 and v['projection_closure_max_px']<.003 for v in r['views'])
            rr=dict(r,evaluation_role='randomized_rig_train' if r['split']=='train' else 'unseen_rig_gate' if r['split']=='val' else 'sealed_confirmation')
            new.append(rr)
    assert len(new)==408
    rigsets={s:{x['rig'] for x in new if x['split']==s} for s in ('train','val','sealed')}
    assert [len(rigsets[s]) for s in ('train','val','sealed')]==[48,12,8]
    assert all(not rigsets[a]&rigsets[b] for a,b in [('train','val'),('train','sealed'),('val','sealed')])
    def signature(rec):
        e=np.array([v['E'] for v in rec['views']]);normalized=e@np.linalg.inv(e[0])
        normalized[:,:3,3]/=geo.old.median_camera_baseline(e)
        # Do not let world origin or global units create fake rig identities.
        return hashlib.sha256(np.round(normalized,4).tobytes()).hexdigest()
    sigs={s:{signature(x) for x in new if x['split']==s} for s in rigsets}
    assert all(not sigs[a]&sigs[b] for a,b in [('train','val'),('train','sealed'),('val','sealed')])
    records.extend(sorted(new,key=lambda x:(x['rig'],x['motion_id'],x['frame'])))
    admit_dannce(records)
    assert all('water_reaching' not in x['group'] and not x['group'].startswith('anipose') for x in records)
    manifest=dict(records=records,rigs=original['rigs'],excluded_groups=dict(excluded),
        rig_splits={k:sorted(v) for k,v in rigsets.items()},rig_fingerprints={k:sorted(v) for k,v in sigs.items()},
        limitations=['Two crossed admitted motion groups, not three; CRIM13 is free-motion replay only.',
            'PyMouseStudio real captures share one calibration; not unseen-rig camera evidence.'])
    geo.write(OUT/'manifest.json',manifest)
    geo.write(OUT/'data_qa.json',dict(passed=True,rendered_scenes=len(new),rendered_views=len(new)*3,
        train_rigs=48,development_rigs=12,sealed_rigs=8,water_reaching_records=0,
        rejected_records=0,independent_normalized_rig_signatures=True,
        counts=[dict(split=k[0],kind=k[1],role=k[2],count=n) for k,n in
                collections.Counter((r['split'],r['kind'],r['evaluation_role']) for r in records).items()]))
    return manifest


def sampling_role(rec):
    if rec['evaluation_role']=='randomized_rig_train':return 'new_rig'
    if rec['kind']=='real_labeled':return 'real'
    if rec['kind']=='real_external':return 'dannce'
    if any(v['physical_role']=='bottom' for v in rec.get('views',[])):return 'bottom'
    return 'replay'


def packs_for(rec,manifest):
    views=geo.old.views_for(rec,manifest)
    if rec['kind']=='real_external':
        ids=[v['id'] for v in views];assert len(ids)==6
        return ([[ids[(j+k)%6] for k in range(3)] for j in range(6)]+[ids[::2],ids[1::2]]+
            [[ids[j],ids[(j+3)%6],ids[(j+1)%6]] for j in range(6)])
    return [list(p) for p in dict.fromkeys(tuple(p) for p in geo.old.packs(views))]


def schedule(manifest,updates=1500):
    pools=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,r in enumerate(manifest['records']):
        if r['split']=='train':
            role=sampling_role(r)
            key=(r['rig'],r['motion_id']) if role=='new_rig' else r['group']
            pools[role][key].append(i)
    assert set(pools)=={'new_rig','bottom','replay','real','dannce'}
    rng=random.Random(2026101803);orders={};cursor=collections.Counter();gc=collections.Counter();visits=collections.Counter();work=[]
    for u in range(updates):
        roles=['new_rig','new_rig','bottom','replay','real','real','dannce','replay']
        micros=[]
        for role in roles:
            gs=sorted(pools[role]);g=gs[gc[role]%len(gs)];gc[role]+=1;key=(role,g);pool=pools[role][g]
            if cursor[key]%len(pool)==0:orders[key]=pool.copy();rng.shuffle(orders[key]);cursor[key]=0
            i=orders[key][cursor[key]];cursor[key]+=1;r=manifest['records'][i]
            packs=packs_for(r,manifest)
            index=(visits[i]+gc[role]-1)%len(packs) if role=='dannce' else visits[i]%len(packs)
            pack=packs[index];visits[i]+=1
            if role=='dannce':
                shift=((gc[role]-1)//len(packs))%3;pack=pack[shift:]+pack[:shift]
            elif len(pack)>1 and (u+len(micros))%2:pack=pack[1:]+pack[:1]
            micros.append(dict(index=i,views=pack,sampling_role=role))
        work.append(micros)
    return work


def calibration_items(manifest):
    # 48 distinct training rigs plus eight real groups/frames, four old bottom
    # and four free-motion replay samples; DANNCE shares scale-free camera term.
    pools=collections.defaultdict(list)
    for i,r in enumerate(manifest['records']):
        if r['split']=='train':pools[sampling_role(r)].append(i)
    chosen=[];rigs=collections.defaultdict(list)
    for i in pools['new_rig']:
        r=manifest['records'][i]
        rigs[r['rig']].append(i)
    for n,rig in enumerate(sorted(rigs)):
        ii=rigs[rig];motions=sorted({manifest['records'][i]['motion_id'] for i in ii});motion=motions[n%len(motions)]
        mm=[i for i in ii if manifest['records'][i]['motion_id']==motion];chosen.append(mm[n%len(mm)])
    for role,n in [('real',8),('bottom',4),('replay',4),('dannce',4)]:
        ii=pools[role]
        groups=sorted({manifest['records'][i]['group'] for i in ii});assert len(groups)<=n
        for j in range(n):
            group=groups[j%len(groups)];gg=[i for i in ii if manifest['records'][i]['group']==group]
            chosen.append(gg[min(len(gg)-1,(j//len(groups))*len(gg)//2)])
        if role=='dannce':chosen[-n:]=[ii[int(x)] for x in np.linspace(0,len(ii)-1,n).round()]
    assert len(chosen)==68
    result=[];counts=collections.Counter()
    for i in chosen:
        r=manifest['records'][i];role=sampling_role(r);packs=packs_for(r,manifest)
        p=packs[(counts[role]*len(packs)//4)%len(packs)] if role=='dannce' else packs[0];counts[role]+=1
        result.append(dict(index=i,views=p,sampling_role=role))
    return result


def load_cpu(rec,manifest,pack):
    assert 'water_reaching' not in rec['group'] and not rec['group'].startswith('anipose')
    b=io.load_cpu(rec,manifest,pack)
    if rec.get('depth_is_pseudo_label'):
        views=geo.old.views_for(rec,manifest);scale=geo.old.median_camera_baseline(np.array([v['E'] for v in views]))
        for j,vid in enumerate(pack):
            index=[v['id'] for v in views].index(vid)
            depth=geo.old.read_pfm(geo.ROOT/rec['pseudo_depth'][index],flip_y=True)
            with Image.open(geo.ROOT/rec['pseudo_mask'][index]) as im:mask=np.array(im.convert('L'))
            with Image.open(geo.ROOT/rec['pseudo_confidence'][index]) as im:confidence=np.array(im.convert('L'))
            assert depth.shape==mask.shape==confidence.shape and np.isfinite(depth).all() and (depth>=0).all()
            # No invented background target or confidence-as-probability.
            interior=geo.old.cv2.erode(mask,np.ones((3,3),np.uint8))>127
            valid=(depth>0)&interior&(confidence>=212)
            z=geo.old.resized(Image.fromarray(depth),512,True)[0]/scale
            valid=geo.old.resized(Image.fromarray(valid.astype(np.uint8)*255),512,True)[0]>0
            assert valid.sum()>=64,(rec['source_frame_id'],vid,int(valid.sum()))
            b['depth'][0,j]=torch.from_numpy(z.astype(np.float32));b['body'][0,j]=torch.from_numpy(valid);b['valid'][0,j]=torch.from_numpy(valid)
    b['point_distance_scale']=losses.point_distance_scale(b)
    return b


class Samples(Dataset):
    def __init__(self,manifest,items):self.manifest=manifest;self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        item=self.items[i]
        return dict(sequence=i,item=item,batch=load_cpu(self.manifest['records'][item['index']],self.manifest,item['views']))


class Feed:
    def __init__(self,manifest,items,workers=4):
        self.items=items;self.position=0
        opts=dict(dataset=Samples(manifest,items),batch_size=None,shuffle=False,num_workers=workers,
            pin_memory=True,generator=torch.Generator().manual_seed(20261018),worker_init_fn=io.worker_init)
        if workers:opts.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2,timeout=180)
        self.dl=DataLoader(**opts);self.it=iter(self.dl)
    def next(self,item):
        assert self.items[self.position]==item
        result=next(self.it);assert result['sequence']==self.position and result['item']==item
        self.position+=1
        return io.to_device(result['batch'])
    def close(self):self.it=None;self.dl=None


if __name__=='__main__':prepare()
