"""Read-only calibration/provenance audit restricted to already admitted real images.

No real depth, triangulation, old reconstruction, normalized keypoint observations,
excluded images, hardware-sync certification, or calibration fitting is performed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, datetime as dt, hashlib, itertools, json, sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/calibrated_mouse_10h_20261025'
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def camera_checks(camera):
    k=np.asarray(camera['K'],float);e=np.asarray(camera['world_to_camera_mm'],float)
    assert k.shape==(3,3) and e.shape==(4,4)
    assert np.isfinite(k).all() and np.isfinite(e).all()
    assert k[0,0]>0 and k[1,1]>0 and np.allclose(k[2],[0,0,1],atol=1e-12)
    assert np.allclose(e[3],[0,0,0,1],atol=1e-12)
    r=e[:3,:3];orth=float(np.max(np.abs(r@r.T-np.eye(3))));det=float(np.linalg.det(r))
    assert orth<1e-8 and abs(det-1)<1e-8
    assert np.array_equal(camera['distortion'],np.zeros(5))
    assert camera['lens_correction_applied'] is True
    assert camera['input_assumed_distorted'] is True
    return k,e,dict(rotation_orthogonality_max=orth,rotation_determinant=det,
        camera_center_world_mm=(-r.T@e[:3,3]).tolist())

def ray_box(xy,k,e,bounds):
    """Native pixel rays parameterized by positive CAMERA-Z, matching trained bounds."""
    k=np.asarray(k,float);e=np.asarray(e,float);bounds=np.asarray(bounds,float)
    origin=-e[:3,:3].T@e[:3,3]
    direction=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(k).T@e[:3,:3]
    parallel=np.abs(direction)<1e-8
    safe=np.where(parallel,1,direction)
    first,second=(bounds[0]-origin)/safe,(bounds[1]-origin)/safe
    lower=np.where(parallel,-np.inf,np.minimum(first,second))
    upper=np.where(parallel,np.inf,np.maximum(first,second))
    near=np.maximum(1.,lower.max(-1));far=np.minimum(2000.,upper.min(-1))
    outside=(parallel&((origin<bounds[0])|(origin>bounds[1]))).any(-1)
    return (far>near)&~outside,near,far

def halfpixel_resize(k,xy,scale):
    a=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
    transformed=np.c_[xy,np.ones(len(xy))]@a.T
    before=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(k).T
    after=transformed@np.linalg.inv(a@k).T
    return float(np.max(np.abs(before-after)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT/'admitted_real_calibration_audit_v2.json');args=ap.parse_args()
    assert not args.output.exists(),'Immutable output already exists'
    manifest=OLD/'semseg_items.json';rows=[r for r in json.loads(manifest.read_text())['items'] if r['domain']=='real']
    assert len(rows)==1336 and Counter(r['split'] for r in rows)=={'train':889,'dev':447}
    admitted={r['image']:r for r in rows};assert len(admitted)==len(rows)
    groups=sorted({r['group'] for r in rows})
    assert groups==['AZL3','AZR2','AZR2L2','CR2','EL2','EL2L2','ER3L2','FR2','FR3']
    ingestion_path=BASE/'ingestion_manifest.json';ingestion=json.loads(ingestion_path.read_text())
    entries={v['session']:v for v in ingestion['calibration'] if v['session'] in groups}
    seq_path=ROOT/'dataset/PyMouseStudio/sequences.json'
    sequences={v['session']:v for v in json.loads(seq_path.read_text()) if v['session'] in groups}
    input_paths=[manifest,ingestion_path,seq_path,ROOT/'dataset/PyMouseStudio/processing_config.json',ROOT/'dataset/PyMouseStudio/README.md',BASE/'README.md',OLD/'data_manifest.json']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in input_paths}
    bounds=json.loads((OLD/'data_manifest.json').read_text())['capture_aabb_mm']
    reports={};frame_groups=defaultdict(set);contexts=Counter();external_contexts=set();row_receipts=[]
    for group in groups:
        path=BASE/entries[group]['path'];assert path==BASE/'calibration'/group/'rectified_calibration.json'
        digest=sha(path);assert digest==entries[group]['sha256'];source_hashes[str(path.relative_to(ROOT))]=digest
        calibration=json.loads(path.read_text());selected=[r for r in rows if r['group']==group]
        cameras={};seq=sequences[group]
        for name in sorted({Path(r['image']).stem[3:] for r in selected}):
            k,e,checks=camera_checks(calibration[name]);w,h=calibration[name]['size_wh']
            y,x=np.meshgrid(np.linspace(0,h-1,33),np.linspace(0,w-1,33),indexing='ij');xy=np.c_[x.ravel(),y.ravel()]
            intersects,near,far=ray_box(xy,k,e,bounds)
            affine_error=max(halfpixel_resize(k,xy,side/max(w,h)) for side in [384,512])
            assert affine_error<1e-12
            cameras[name]=dict(**checks,K=k.tolist(),world_to_camera_mm=e.tolist(),size_wh=[w,h],
                physical_role=calibration[name]['physical_role'],units_evidence=calibration[name]['units_evidence'],
                lens_correction_applied=True,raw_was_distorted_is_user_assumption=True,
                halfpixel_isotropic_resize_ray_max_error=affine_error,
                fixed_synthetic_AABB_diagnostic=dict(grid='33x33 including image borders',rays=len(xy),intersecting_positive_cameraZ_rays=int(intersects.sum()),fraction=float(intersects.mean())))
        for row in selected:
            ip=ROOT/row['image'];lp=ROOT/row['label'];cam=ip.stem[3:];frame=int(ip.parent.name.split('_')[1])
            assert ip.parent.parent.name==group and cam in seq['cameras'] and frame in seq['annotation_frame_indices']
            assert sha(ip)==row['image_sha256'] and sha(lp)==row['label_sha256']
            with Image.open(ip) as im:shape=im.size
            with Image.open(lp) as im:assert im.size==shape
            assert list(shape)==calibration[cam]['size_wh'] and list(shape[::-1])==row['native_hw']
            frame_groups[(row['split'],group,frame)].add(cam)
            context=row['context'];same_frame=all((ROOT/s).parent==ip.parent for s in context)
            assert same_frame,'Cross-frame context unexpectedly present'
            missing=[s for s in context if s not in admitted];contexts['same_frame_index']+=1
            if missing:contexts['rows_with_nonadmitted_context_image']+=1;external_contexts.update(missing)
            else:contexts['all_context_images_admitted']+=1
            row_receipts.append(dict(id=row['id'],split=row['split'],group=group,frame=frame,camera=cam,
                image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],same_frame_context=True,
                all_context_images_admitted=not missing))
        reports[group]=dict(split=selected[0]['split'],images=len(selected),frames=len({int(Path(r['image']).parent.name.split('_')[1]) for r in selected}),
            calibration_sha256=digest,fps=seq['fps'],provided_camera_names=seq['cameras'],missing_camera_videos=seq['missing_camera_videos'],
            admitted_camera_counts=dict(Counter(Path(r['image']).stem[3:] for r in selected)),cameras=cameras)
    # Source-index grouping is recorded without reading nonadmitted contexts or hardware clocks.
    frame_rows=[dict(split=s,group=g,source_frame=f,admitted_cameras=sorted(c)) for (s,g,f),c in sorted(frame_groups.items())]
    result=dict(schema='admitted-real-calibration-provenance-audit-v1',completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Already pinned real TRAIN/DEV image and manual-mask hashes plus corresponding calibration/provenance metadata only',
        source_sha256=source_hashes,audit_source_sha256=sha(__file__),images=len(rows),split_counts=dict(Counter(r['split'] for r in rows)),
        groups=reports,frame_groups=frame_rows,context_counts=dict(contexts),nonadmitted_context_paths_count=len(external_contexts),
        nonadmitted_context_images_opened=0,excluded_images_opened=0,old_reconstructions_or_pseudo_3D_opened=0,
        image_label_hashes_verified=True,camera_K_E_numerical_checks_passed=True,
        unique_calibration_hashes=sorted({r['calibration_sha256'] for r in reports.values()}),
        synthetic_capture_AABB_mm=bounds,ray_diagnostic_uses_no_mouse_labels=True,
        synchronization=dict(evidence='Equal source frame indices and nominal52fps metadata only',hardware_verified=False,
            independent_clock_or_trigger_receipt_found=False,scope='This metadata package; absence here is not proof no hardware synchronization existed'),
        units=dict(metadata='world_to_camera_mm; original calibration mm provenance',independent_scale_reference_verified=False,
            original_world_frame_not_ground_aligned=True,synthetic_to_real_gauge_registration_verified=False),
        limitations=['Do not undistort already rectified images again; original distortion status was a user working assumption.',
            'Glass refraction is not removed by lens correction.',
            'Same source index alone does not verify hardware synchronization.',
            'Current synthetic capture AABB and world-normal prior cannot be assumed compatible with real calibration gauge.',
            'No real depth inference, geometric GT scoring, fitting, source admission or model promotion performed.'],
        rows=row_receipts)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    compact={k:v for k,v in result.items() if k not in ['rows','frame_groups','groups']}
    compact['frame_groups']=len(frame_rows)
    compact['per_group']={g:{k:v for k,v in r.items() if k!='cameras'} for g,r in reports.items()}
    compact['per_camera_synthetic_AABB_intersection']={g:{c:r['fixed_synthetic_AABB_diagnostic'] for c,r in v['cameras'].items()} for g,v in reports.items()}
    compact['full_audit_sha256']=sha(args.output)
    target=ROOT/'docs/experiments/calibrated_mouse_admitted_real_calibration_20261025.json'
    target.write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps({k:compact[k] for k in ['images','split_counts','frame_groups','context_counts','unique_calibration_hashes','per_camera_synthetic_AABB_intersection']}))
if __name__=='__main__':main()
