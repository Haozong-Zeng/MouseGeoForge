"""Read-only prior DANNCE4DGS artifact audit; no training/admission/scoring."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dest=OUT/'existing_dannce4dgs_review_v1.json'
 if dest.exists():raise FileExistsError(dest)
 base=ROOT/'output/DANNCE-4DGS-Pilot-mm1-f0000-0079';vis=base/'visualizations_v3_foreground';summary=json.loads((vis/'visualization_summary.json').read_text());sanity=json.loads((vis/'depth_keypoint_check.json').read_text())
 assert summary['iteration']==10000 and summary['camera_count']==6 and summary['frames_per_camera']==80
 sources=[base/'4dgs_foreground_v3_review.md',base/'4dgs_visual_review.md',vis/'visualization_summary.json',vis/'depth_keypoint_check.json',ROOT/'tools/export_dannce_4dgs_visualizations.py',ROOT/'tools/evaluate_dannce_4dgs_depth.py']
 depths=[]
 for p in sorted((vis/'selected').glob('*_depth.npy')):
  a=np.load(p,allow_pickle=False);assert a.shape==(512,512) and np.isfinite(a).all();sources.append(p);depths.append(dict(path=str(p.relative_to(ROOT)),shape=list(a.shape),stored_positive_pixels=int((a>0).sum()),stored_units='sceneunit=100mm'))
 assert len(depths)==30
 source_dataset=Path(summary['source_path']);manifest=source_dataset/'pilot_manifest.json'
 if manifest.exists():sources.append(manifest)
 gs=base/'model_v3_foreground/point_cloud/iteration_10000/point_cloud.ply';assert gs.exists();sources.append(gs)
 for f in base.joinpath('model_v3_foreground').rglob('deformation*.pth'):sources.append(f)
 result=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),model_calls=0,training_updates=0,new_data_admission=False,reused_existing_model=True,source_clip=dict(session='DANNCE markerless_mouse_1',raw_frames=[0,79],synchronized_views=6,images=480,coarse_updates=1000,fine_updates=10000,gaussians=38963),stored_observed_view_appearance=dict(psnr_db=summary['mean_psnr_db'],mean_mae=summary['mean_mae']),stored_landmark_sanity_only={k:sanity[k] for k in ['sample_count','median_absolute_error_mm','mean_absolute_error_mm','p90_absolute_error_mm','median_bias_mm']},depth_arrays=depths,depth_semantics='Exporter divides sum(alpha*transmittance*cameraZ) by accumulatedalpha: expected cameraZ, not guaranteed firstopaque surface. Multiplystoreddepth100for mm.',dependency_disclosure=['SAM2 masks were prompted from projected released3D atframes27and72 then propagated.','Visualhull initialization uses SAM2 silhouette masks and thetwo promptedposes.','World centering uses mouseCOM and virtualcameras; must audit derivation before any heldout-GT-independent claim.','Stored view render scores and same labelled prompt-point comparisons do not establish heldout surface accuracy.','Anatomical/internal/occluded joints differ from visible skin;2.43mmmedian is not dense depth error.'],recommended_role='Existing weakdepth/appearance/temporal consistency baseline and hypothesis source, not sensor/GT supervision. No duplicatepilot training; first establish disjoint heldout views/times with independent reference and full failuredenominators.',not_restarted=True,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ['completed_utc','source_clip','stored_landmark_sanity_only','recommended_role']}))
if __name__=='__main__':main()
