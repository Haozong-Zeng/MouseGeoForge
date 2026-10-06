"""Append only previously unused eligible TRAIN groups; freeze all validation."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import prepare_surface_curve_real_20261003 as derive
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'expanded_real_point_train_v1'


def main():
    assert not OUT.exists();OUT.mkdir()
    old=json.loads((CYCLE/'training_v1/protocol.json').read_text());manifest_path=ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json';m=json.loads(manifest_path.read_text())
    used={(r['session'],r['frame']) for r in old['real']};train=[g for g in m['frame_groups'] if g['eligible_for_development'] and g['role']=='train']
    additions=sorted([g for g in train if (g['session'],g['frame']) not in used],key=lambda g:(g['session'],g['frame']))
    assert len(train)==156 and len(additions)==128
    provenance=collections.defaultdict(list)
    for r in m['provenance']:provenance[(r['session'],r['frame'])].append(r)
    real=[]
    for g in additions:
        assert g['session'] not in ('GR2','UR2','UR3','AZR2L2','FR2') and not g['excluded']
        views=provenance[(g['session'],g['frame'])]
        real.append(dict(session=g['session'],frame=g['frame'],role='train',normalized=g['effective_normalized_file'],normalized_sha256=fingerprint(ROOT/g['effective_normalized_file']),views=[{k:r[k] for k in ['camera','annotation','source_image','mask','mask_valid','annotation_sha256','mask_sha256']} for r in views]))
    write_json(OUT/'protocol.json',dict(real=real,real_affine=old['real_affine'],real_supervised_ids_1based=old['real_supervised_ids_1based'],
        source_manifest_sha256=fingerprint(manifest_path),old_protocol_sha256=fingerprint(CYCLE/'training_v1/protocol.json'),
        policy='128additionaleligibleTRAINgroups only. Original28TRAIN+8VALgroups/236features will be preserved as exactprefix for data-scalecontrols. DoNOTadd/changevalidation or touchlockedholdout.',
        expected_groups=128,expected_images=sum(len(r['views']) for r in real)))
    derive.BASE=OUT;derive.main()


if __name__=='__main__':main()
