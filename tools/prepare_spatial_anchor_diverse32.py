"""Deterministically expand train-only source poses, preserving original16."""
import copy,hashlib,json
import numpy as np
from run_depth_pose_ablation import ROOT,write_json,fingerprint
from train_cylinder_view_balance import scene_digest
from audit_bottom_view_coverage import cameras

OUT=ROOT/'output/spatial_anchor_diverse32_plan_v1.json'


def main():
    if OUT.exists():raise FileExistsError('Preserve existing sampling plan')
    old_path=ROOT/'output/spatial_anchor_pilot_v1/protocol.json'
    old=json.loads(old_path.read_text())
    parent=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
    manifest_path=ROOT/parent['config']['manifest'];manifest=json.loads(manifest_path.read_text())
    assert fingerprint(manifest_path)==parent['manifest_sha256']
    keys=list(old['train_keys']);records={k:old['records'][k] for k in keys}
    used={r['scene'] for r in records.values()}
    groups=[('PyMouseStudio-BrownFur-ClimbWall-Core23',8),
            ('Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1',4),
            ('PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200',1),
            ('Homecage-BlackFur-KeypointMoseq-Core23-Sample10-Frames200-399',3)]
    additions=[]
    for dataset,n in groups:
        pool=sorted([r for r in manifest['records'] if r['split']=='train' and r['dataset']==dataset and r['scene'] not in used],key=lambda r:r['frame'])
        assert len(pool)>=n
        chosen=[pool[i] for i in np.linspace(0,len(pool)-1,n).round().astype(int)]
        templates=[r for r in records.values() if r['dataset']==dataset]
        for i,source in enumerate(chosen):
            template=templates[i%len(templates)];r=copy.deepcopy(source)
            order=template['views'];meta=json.loads((ROOT/r['scene']/'scene_meta.json').read_text())
            refmeta=json.loads((ROOT/template['scene']/'scene_meta.json').read_text())
            byid={v['view_id']:v for v in meta['views']};refid={v['view_id']:v for v in refmeta['views']}
            for view in order:
                np.testing.assert_allclose(byid[view]['extrinsic'],refid[view]['extrinsic'],atol=1e-6,rtol=1e-6)
            roles={v['view_id']:v['orientation'] for v in cameras(ROOT/r['scene'])}
            assert [roles[v] for v in order]==template['physical_roles']
            r.update(views=order,diagnostic_view_order=order,physical_roles=[roles[v] for v in order],
                normalization_scale=template['normalization_scale'],source_revision=scene_digest(ROOT/r['scene']))
            key=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
            assert r['scene'] not in used
            records[key]=r;keys.append(key);used.add(r['scene']);additions.append(dict(key=key,template_scene=template['scene']))
    assert len(keys)==32 and len(used)==32
    assert not {r['source_group'] for r in records.values()}&{old['records'][e['key']]['source_group'] for e in old['evaluation']}
    write_json(OUT,dict(train_keys=keys,records=records,evaluation=old['evaluation'],additions=additions,
        parent_protocol_sha256=fingerprint(old_path),manifest_sha256=fingerprint(manifest_path),
        code_sha256=fingerprint(__import__('pathlib').Path(__file__)),
        scope='Same synthetic train split: original16 retained +16 new scenes;16 climb/8 dual/4 brown MoSeq/4 black MoSeq. No augmentation/loss/architecture change. Fixed800 means half updates per scene vs original16.',
        source_selection='Uniform frame positions among eligible new train scenes, no prediction/GT difficulty scoring',
        normalization='New scenes share checked static camera extrinsics and role order with existing same-dataset template.'))
    print(json.dumps(dict(records=32,datasets={name:sum(r['dataset']==name for r in records.values()) for name,_ in groups})))


if __name__=='__main__':main()
