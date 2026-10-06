"""Publish only QA-passed static-pose dataset; preserve all failed diagnostics."""
import datetime as dt,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/surface_curve_cycle_20261003';SRC=CYCLE/'render_staging'
OUT=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'
DELIVERY=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SurfaceCurve-Cycle-20261003')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not OUT.exists() and not DELIVERY.exists()
    qa=load(SRC/'independent_geometry_qa.json');assert qa['gates']['depth_passed'];assert load(SRC/'loader_qa.json')['passed']
    m=load(SRC/'manifest.json');assert len(m['poses'])==3 and all(len(p['views'])==7 for p in m['poses'])
    OUT.mkdir();DELIVERY.mkdir();inventory=[]
    for p in m['poses']:
        name=f"frame_{p['frame_index']:06d}";src=SRC/name;dst=OUT/name;dst.mkdir()
        for sub in ['blended_images','rendered_depth_maps','masks','cams','annotations']:
            shutil.copytree(src/sub,dst/sub)
        for file in ['keypoints.npz','surface_curve.npz','scene_meta.json','geometry.npz','visibility_support.npz']:
            shutil.copy2(src/file,dst/file)
        for path in sorted(dst.rglob('*')):
            if path.is_file():
                rel=path.relative_to(dst);assert sha(path)==sha(src/rel);inventory.append(dict(path=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    metadata=dict(format='blenderpymouse_core23_surface_curve_pilot_v1',status='validated_static_pilot',frames=3,views_per_frame=7,images=21,
        resolution_wh=[1080,1200],projection='perspective',units='mm',depth_type='camera_z',depth_policy=load(SRC/'geometry_pass_manifest.json')['policy'],
        mask_definition='Biological skin mesh only, excluding fur, implants, transparent floor and reflections; actual non-glass scene occluders retained.',
        temporal_reliable=False,frame_order='Independent accepted static poses, NOT a video or animation sequence',
        source_poses=[{k:p[k] for k in ['pose','source_frame','frame_index','blend','blend_sha256']} for p in m['poses']],
        surface_curve='65 fixed material samples from dorsal rest mesh X=0 at neck-to-tail-root levels. Not internal spine, not view silhouette, not replacing Core23 06/07.',
        curve_endpoints='Virtual dorsal surface points, not the internal bone05 or bone08 coordinates.',
        quality_scope='Numerical geometry/loader QA, not evidence of real-domain model accuracy. Three poses are insufficient for standalone foundation-model training.',
        keypoint_definition_version='pymousestudio_manual_surface_core23_v3',rgb_geometry_caveat='RGB includes optical glass/reflections and fur; geometric depth and masks do not trace reflected or refracted image paths.',
        pfm_flip_y=True,files=inventory)
    save(OUT/'metadata.json',metadata)
    for file in ['independent_geometry_qa.json','loader_qa.json','conversion_report.json']:shutil.copy2(SRC/file,OUT/file)
    review=dict(rgb_all21_visually_reviewed=True,mask_curve_all21_visually_reviewed=True,depth_views_reviewed=['quadruped A','quadruped D'],
       findings=['Reflections of feet/tail and glass floor remain outside skin masks.','All bottom views have zero directly visible dorsal-line samples.','Visible dorsal line follows body arch, while occluded sections and implant occlusion are suppressed.','Fur can extend beyond biological skin mask, intentionally not claimed as exact fur-outline segmentation.','Depth side preview retains far cylinder wall; bottom preview retains cylinder but skips intervening glass floor.','Accepted material appearance unchanged, including bright tail/feet highlights and residual synthetic appearance.'],
       limits=['Visibility uses raster depth tolerance .75mm; exact silhouette occlusion deserves additional stress tests.','Only two full-scene depth previews visually checked; all21 animal-depth fields independently ray-checked.'])
    save(OUT/'visual_qa.json',review);save(CYCLE/'visual_qa.json',review)
    for p in (CYCLE/'visual_review').glob('*.jpg'):shutil.copy2(p,DELIVERY/p.name)
    for file in ['metadata.json','independent_geometry_qa.json','loader_qa.json','visual_qa.json']:shutil.copy2(OUT/file,DELIVERY/file)
    intro='新数据集：三种静态姿态 × 七个标定视角，共21图，1080×1200。不是连续时序。绿色为生物皮肤网格分割；橙色仅画可见背侧体表线，底视不强行显示被遮挡背线。毛发、植入硬件、玻璃及倒影不属于该skin mask。全景深度是毫米单位相机Z，侧视保留远侧筒壁，底视跳过挡在鼠体前的玻璃底板。数值几何和1/2/7视角dataloader检查已通过；不代表真实数据上的训练收益。'
    html='<meta charset="utf-8"><title>体表曲线首批数据检查</title><style>body{font:18px/1.6 sans-serif;max-width:1600px;margin:30px auto;background:#111b28;color:#eee}img{max-width:100%}a{color:#8cdcff}</style><h1>PyMouseStudio 体表曲线数据检查</h1><p>'+intro+'</p>'
    for pose in ['climb','quadruped','headraise']:
        html+=f'<h2>{pose}</h2><a href="{pose}_mask_visible_curve.jpg"><img src="{pose}_mask_visible_curve.jpg"></a><p>未按可见性过滤的诊断投影：<a href="{pose}_curve_projections.jpg">打开</a></p>'
    html+='<p><a href="independent_geometry_qa.json">独立深度检查</a> · <a href="loader_qa.json">加载检查</a> · <a href="visual_qa.json">视觉检查及限制</a></p>'
    (DELIVERY/'index.html').write_text(html,encoding='utf-8')
    for rel in [p.name for p in DELIVERY.glob('*.jpg')]:assert(DELIVERY/rel).is_file()
    report=dict(dataset=OUT.relative_to(ROOT).as_posix(),delivery=str(DELIVERY),payload_files=len(inventory),bytes=sum(p['bytes'] for p in inventory),images=21,poses=3,all_copied_hashes_verified=True,
        max_view_median_depth_error_mm=max(v['depth_median_abs_mm'] for v in qa['views']),max_view_p95_depth_error_mm=max(v['depth_p95_abs_mm'] for v in qa['views']))
    save(CYCLE/'publication.json',report)
    state=load(CYCLE/'state.json');state.update(stage='pilot_dataset_published_curve_exploration_next',updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),active_pid=None,active_job=None,next='Run synthetic surface-curve stress tests and design balanced point/anisotropic/curve controls using broader existing synthetic + eligible real data; three-pose pilot alone insufficient. Read publication.json and visual_qa.json.');state['completed'].append('render_and_validate_new_multiview_dataset');save(CYCLE/'state.json',state)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
