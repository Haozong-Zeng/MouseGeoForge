"""Deterministic TRAIN-only diversity expansion; validation source/frame sets frozen."""
import json,hashlib,datetime as dt,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/surface_curve_cycle_20261003'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    source=BASE/'training_v1/protocol.json';cfg=json.loads(source.read_text());out=BASE/'diverse_curve_expansion_protocol.json';assert not out.exists()
    by=collections.defaultdict(list)
    for r in cfg['synthetic_train']:by[r['dataset']].append(r)
    sourceplan=json.loads((ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json').read_text());train=[]
    for name,original in sorted(by.items()):
        sources={r['source_group'] for r in original};assert len(sources)==1;group=next(iter(sources))
        dirs=sorted((ROOT/'dataset/BlenderPyMouse'/name).glob('frame_*'))
        if group.startswith('crim13/'):
            session=group.split('/',1)[1];allowed={int(k['scene_frame']) for k in sourceplan['keyframes'] if Path(k['source_image'].replace('\\','/')).parent.name==session}
            assert len(allowed)==200;dirs=[d for d in dirs if int(d.name.split('_')[-1]) in allowed]
        # All other chosen datasets already belong entirely to their inherited TRAIN group.
        else:assert group in ('mouse_climb_wall_91','moseq_sample10'),group
        chosen={dirs[i] for i in np.linspace(0,len(dirs)-1,min(20,len(dirs))).round().astype(int)}|{ROOT/r['scene'] for r in original}
        for folder in sorted(chosen):
            with np.load(folder/'keypoints.npz') as a:frame=int(a['frame']);instances=1 if a['joint_world_xyz'].ndim==2 else len(a['joint_world_xyz'])
            train.append(dict(scene=folder.relative_to(ROOT).as_posix(),dataset=name,frame=frame,expected_instances=instances,source_group=group,role='train',
                annotation_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [folder/'keypoints.npz',folder/'scene_meta.json'] if p.exists()}))
    validation=cfg['synthetic_development'];assert not {r['scene'] for r in train}&{r['scene'] for r in validation}
    assert {r['scene'] for r in cfg['synthetic_train']}<={r['scene'] for r in train}
    payload=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_protocol_sha256=sha(source),
        purpose='More diverse TRAIN geometry, not more updates on same tiny3pose set. Existing validation remains unchanged. All source geometry must pass strict support checks before training.',
        selection='20evenly spaced scenes per authorized training source plus all original4selections, before looking at model outcomes. CRIM13 scene membership explicitly resolved by training video session.',
        synthetic_train=train,synthetic_development=validation,new_synthetic_train=cfg['new_synthetic_train'],
        real_data_changed=False,validation_selection_changed=False,source_group_policy='Inherited original experiment groups; no GR2/UR2/UR3 real data touched')
    out.write_text(json.dumps(payload,indent=2));print(json.dumps(dict(train_candidates=len(train),validation=len(validation),train_sources=dict(collections.Counter(r['dataset'] for r in train))),indent=2))
if __name__=='__main__':main()
