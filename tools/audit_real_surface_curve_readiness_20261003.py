"""Read-only annotation evidence audit; no reconstruction or locked-label access."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image
from mousegeoforge.surface_curve_supervision import back_tangents
from run_depth_pose_ablation import fingerprint,write_json
VERSION=ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1'
OUT=ROOT/'output/surface_curve_cycle_20261003/real_curve_readiness_v1'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    manifest=json.loads((VERSION/'manifest.json').read_text());rows=[];counts=collections.defaultdict(collections.Counter)
    for group in manifest['frame_groups']:
        # Filter by manifest BEFORE loading any actual observation coordinates.
        if group['role'] not in ('train','validation') or group['excluded'] or not group['eligible_for_development']:continue
        assert group['session'] not in ('GR2','UR2','UR3')
        path=ROOT/group['effective_normalized_file']
        with np.load(path) as z:
            uv=z['uv'].copy();valid=z['xy_observed'].astype(bool)&z['loss_eligible'].astype(bool)&np.isfinite(uv).all(-1)
            assert not bool(z['excluded_group']);cameras=z['camera_names'].tolist()
        _,tangent_valid=back_tangents(torch.tensor(uv),torch.tensor(valid))
        b=valid[:,0,5:7];four=valid[:,0,4:8].all(-1);near=tangent_valid.numpy()[:,0,5:7]
        c=counts[group['role']];c['groups']+=1;c['images']+=len(uv)
        c['observed06']+=int(b[:,0].sum());c['observed07']+=int(b[:,1].sum())
        c['images_both_back_points']+=int(b.all(-1).sum());c['images_legacy05to08_observed']+=int(four.sum())
        c['06_with_neighbor_tangent']+=int(near[:,0].sum());c['07_with_neighbor_tangent']+=int(near[:,1].sum())
        c['groups_both_back_points_in_at_least3views']+=int(b.all(-1).sum()>=3)
        rows.append(dict(session=group['session'],frame=group['frame'],role=group['role'],
            views=len(uv),cameras=cameras,back_observations_by_view=b.tolist(),tangent_available_by_view=near.tolist(),
            legacy_fourpoint_views=int(four.sum()),source=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path)))
    rejected=manifest['rejected_return2_mask'];path=ROOT/rejected['mask']
    assert fingerprint(path)==rejected['mask_sha256']
    with Image.open(path) as im:area=int((np.asarray(im)>0).sum())
    assert area==146504
    write_json(OUT/'report.json',dict(counts={k:dict(v) for k,v in counts.items()},groups=rows,
        current_mask_regression=dict(session='FR2',frame=584,camera='B',area_px=area,sha256=fingerprint(path),old_sent_mask_preserved=True),
        manifest_sha256=fingerprint(VERSION/'manifest.json'),locked_observations_loaded=0,
        explicit_curve_groundtruth_available=False,full_surface_endpoint_groundtruth_available=False,
        conclusions=['Existing06/07support sparse weak supervision, not a full neck-to-tail-root curve.',
                     'Availability of legacy05/08does not make them the new synthetic virtual surface endpoints.',
                     'Visibility unknown: absence is not proof of occlusion; no guessed visibility targets.',
                     'Do not turn any two-point projection or silhouette into strong3Dsurface groundtruth.',
                     'A future small multi-view curve reference set should record visible shared-surface polyline vs silhouette-only, ordered partial segments, uncertainty and observer ID.'],
        no_labels_changed=True,no_pseudolabels_created=True))
    print(json.dumps({k:dict(v) for k,v in counts.items()},indent=2))


if __name__=='__main__':main()
