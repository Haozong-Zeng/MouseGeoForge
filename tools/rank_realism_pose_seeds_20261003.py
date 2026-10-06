"""Rank source poses as rough, floor-preserving initializations, never annotation GT."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/pymousestudio_realism_cycle_20261002'
def main():
    source=json.loads((OUT/'source_pose_range.json').read_text());cal=json.loads((OUT/'v002_real_cameras/manifest.json').read_text())['global_registration'];Q=np.array(cal['calibration_to_blender_rotation']);T=np.array(cal['translation']);weights=np.array([1.,.3,.5]);weights/=weights.sum();results=[]
    for session,frame in [('FR2',1104),('AZR2L2',129)]:
        path=ROOT/f'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/geometry_diagnostics_NOT_GT/real_development/{session}/frame_{frame:06d}.npz';raw=np.load(path)['xyz_all_ls_mm'];target=np.array([raw[0],raw[14:16].mean(0),raw[16:18].mean(0)])@Q.T+T;rows=[]
        for row in source['sampled_frames']:
            points=np.array([row['bones']['HeadEnd_M'],row['fore_mean'],row['hind_mean']]);a0=(points*weights[:,None]).sum(0);b0=(target*weights[:,None]).sum(0);a=points-a0;b=target-b0
            cross=np.sum(weights*(a[:,0]*b[:,1]-a[:,1]*b[:,0]));dot=np.sum(weights*(a[:,0]*b[:,0]+a[:,1]*b[:,1]));yaw=math.atan2(cross,dot);R=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
            rotated=a@R.T;scale=float(np.clip(np.sum(weights*np.sum(rotated*b,axis=1))/np.sum(weights*np.sum(a*a,axis=1)),.85,1.35))
            if globals().get('FIXED_SCALE') is not None:scale=float(FIXED_SCALE)
            offset=b0-scale*(R@a0);offset[2]=340.5-scale*points[2,2];fitted=scale*points@R.T+offset;errors=np.linalg.norm(fitted-target,axis=1);score=float(np.sum(weights*errors))
            matrix=np.eye(4);matrix[:3,:3]=scale*R;matrix[:3,3]=offset
            rows.append(dict(source_frame=row['frame'],rough_score_mm=score,residuals_nose_foremean_hindmean_mm=errors.tolist(),scale=scale,yaw_degrees=math.degrees(yaw),global_matrix=matrix.tolist(),source_nose_elevation_degrees=row['nose_axis_elevation_degrees'],fore_hind_height_after_mm=scale*row['fore_minus_hind_height_mm']))
        rows.sort(key=lambda r:r['rough_score_mm']);results.append(dict(session=session,real_frame=frame,rough_targets=target.tolist(),top_candidates=rows[:globals().get('TOP_CANDIDATES',8)]))
    result=dict(source='source_pose_range.json',results=results,limitations=['Initialization ranking only. Real least-squares3D is NOT GT; pooled paw identities and nose proxy are uncertain.','Yaw-only, uniform scale0.85–1.35 and assumed hind bone centerZ340.5; actual skin contact/cylinder/tail collision NOT solved here.','Uses FR2 global scene registration for rough coordinate comparison. AZR2L2 camera/frame alignment needs independent verification.','Do not directly publish these matrices as reconstructed labels or animation.'])
    result['fixed_scale']=globals().get('FIXED_SCALE');target=OUT/globals().get('OUTPUT_NAME','pose_seed_ranking.json');assert not target.exists();target.write_text(json.dumps(result,indent=2));print(json.dumps([dict(session=r['session'],real_frame=r['real_frame'],top=[{k:v for k,v in x.items() if k in ['source_frame','rough_score_mm','scale','fore_hind_height_after_mm']} for x in r['top_candidates'][:4]]) for r in results],indent=2))
if __name__=='__main__':main()
