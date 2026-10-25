"""Initial zero-model real-reference feasibility audit; no new training admission."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import numpy as np
from scipy.io import whosmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=OUT/'real_reference_feasibility_v1.json'
    if dest.exists():raise FileExistsError(dest)
    adm_path=ROOT/'output/panomouse_stage01_20261023/stage0/admission.json';adm=json.loads(adm_path.read_text())
    pano=ROOT/'dataset/PanoMouse';inputs={str(adm_path.relative_to(ROOT)):sha(adm_path)}
    for f,h in adm['file_hashes'].items():assert sha(pano/f)==h;inputs[str((pano/f).relative_to(ROOT))]=h
    assert adm['roles']['dev']==['S03','S11'] and set(adm['roles']['sealed'])=={'S04','S05'}
    a2=json.loads((pano/'raw/annotations/annotations_2d.json').read_text());a3=json.loads((pano/'raw/annotations/annotations_3d.json').read_text());cal=json.loads((pano/'raw/annotations/calib_data.json').read_text())
    groups=[g for g in a2['groups'] if g['scene_name'].split('_')[0] in adm['roles']['dev']]
    anns={a['image_id']:a for a in a2['annotations']};images={i['id']:i for i in a2['images']}
    from mousegeoforge.models.calibrated_mouse_depth import ray_bounds
    import torch
    bounds=torch.tensor([[-110.,-125.,289.],[110.,131.,466.]],dtype=torch.float64)
    rows=[];all_z=[];all_inside=[]
    for g in groups:
        session=g['scene_name'].split('_')[0];assert session in ['S03','S11']
        raw=np.asarray(a3[g['scene_name']],dtype=np.float64)[:24];x=raw[:,:3]*1000;valid3=raw[:,3]>0
        inside=valid3&(x>=bounds[0].numpy()).all(-1)&(x<=bounds[1].numpy()).all(-1);all_inside.extend(inside[valid3].tolist())
        for cam,iid in zip(g['cams'],g['image_id']):
            c=cal[session][cam];k=np.asarray(c['K'],dtype=np.float64);R=np.asarray(c['R'],dtype=np.float64);T=np.asarray(c['T'],dtype=np.float64).reshape(3)*1000;e=np.eye(4);e[:3,:3]=R;e[:3,3]=T
            kp=np.asarray(anns[iid]['keypoints'],dtype=np.float64).reshape(-1,3)[:24];ok=valid3&(kp[:,2]==2);indices=np.flatnonzero(ok)
            xy=kp[ok,:2];z=(x@R.T+T)[:,2];all_z.extend(z[ok].tolist())
            _,_,near,far,rayvalid=ray_bounds(torch.from_numpy(xy),torch.from_numpy(k),torch.from_numpy(e),bounds)
            uv=(x@R.T+T)@k.T;uv=uv[:,:2]/uv[:,2:];res=np.linalg.norm(uv[ok]-xy,axis=-1)
            path=pano/'unpacked/images'/images[iid]['file_name'];assert path.is_file()
            rows.append(dict(scene=g['scene_name'],session=session,camera=cam,image=str(path.relative_to(ROOT)),image_sha256=sha(path),visible_labelled_body_points=len(indices),indices=indices.tolist(),released_anatomical_cameraZ_mm=z[ok].tolist(),projection_residual_px=res.tolist(),rays_intersect_original_synthetic_volume=int(rayvalid.sum()),released_point_inside_original_synthetic_volume=int(inside[ok].sum()),released_point_within_ray_interval=int(((torch.from_numpy(z[ok])>=near)&(torch.from_numpy(z[ok])<=far)&rayvalid).sum())))
    dannce=[];download=ROOT/'dataset/DANNCE/download_manifest.json';dm=json.loads(download.read_text());inputs[str(download.relative_to(ROOT))]=sha(download)
    for f in dm['repository_files']:
        if not f['path'].endswith(('label3d_dannce.mat','DANNCE_LICENSE','dannce_mouse_config.yaml','mouse22_skeleton.mat')):continue
        p=ROOT/'dataset/DANNCE'/f['path'];assert sha(p)==f['sha256'];inputs[str(p.relative_to(ROOT))]=f['sha256']
        rec=dict(path=str(p.relative_to(ROOT)),sha256=f['sha256'])
        if p.suffix=='.mat':rec['mat_variables']=[dict(name=n,shape=list(s),matclass=c) for n,s,c in whosmat(p)]
        dannce.append(rec)
    real=ROOT/'output/calibrated_mouse_resume8h_20261025/admitted_real_calibration_audit_v2.json';inputs[str(real.relative_to(ROOT))]=sha(real)
    result=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),model_calls=0,training_updates=0,protected_session_model_calls=0,panomouse=dict(dev_sessions=['S03','S11'],timepoints=len(groups),views=len(rows),visible_body_reference_observations=sum(r['visible_labelled_body_points'] for r in rows),original_synthetic_volume_ray_intersection_fraction=sum(r['rays_intersect_original_synthetic_volume'] for r in rows)/max(1,sum(r['visible_labelled_body_points'] for r in rows)),released_body_point_inside_synthetic_volume_fraction=float(np.mean(all_inside)),released_cameraZ_mm_quantiles=np.percentile(all_z,[0,5,50,95,100]).tolist(),rows=rows),dannce=dict(original_files=dannce,training_or_scoring_new_admission=False,pseudo_depth_used=False,next='Resolve existing split/provenance/semantics and allowed evaluation roles before model calls. Original label provenance does not make existing visual-hull pseudo_depth independent GT.'),pymousestudio='Existing real calibration audit retained; hardware synchronization and independentmm reference not established. Do not re-undistort rectified inputs.',interpretation=['Anatomical3D cameraZ is a correlated sparse reference, not visible skin depth GT. Surface-vs-landmark residual cannot be called dense depth accuracy.','Original synthetic AABB/world-normalized prior must not be silently fitted/aligned to real reference points. Report invalid rays and coverage as failures; no GT gauge fitting.','DANNCE-style direct volumetric landmarks are task-aligned candidate; not established more reliable without matched evaluation.','PanoMouse existing evaluation-only S03/S11 remains; no training data admission or protected S04/S05 scoring.'],input_sha256=inputs)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result['panomouse'].items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
