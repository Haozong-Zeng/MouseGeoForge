"""Create presentation-only delivery copies, reopen them, render all four checked views."""
import bpy,json,sys,time,datetime as dt,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v064_final_delivery';out.mkdir(exist_ok=False);project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002';delivery=project/'DELIVERY';delivery.mkdir(exist_ok=False)
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));files={'climb':state['best_blend'],**state['pose_bank']};labels={'climb':'Climb','quadruped':'Quadruped','headraise':'HeadRaise'};expected={'climb':31,'quadruped':111,'headraise':216}
manifest=dict(start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),appearance_frozen=True,poses={},complete=False)
def write():(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
def guard():assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
def points():
    bpy.context.view_layer.update();ob=bpy.data.objects['MS_Mouse'];ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return q
def cams():return {c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
def gpu():
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
    for d in prefs.devices:
        d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
        if d.use:devices.append(d.name)
    assert devices;bpy.context.scene.cycles.device='GPU';return devices
def select(c):
    s=bpy.context.scene;s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
for tag,file in files.items():
    guard();bpy.ops.wm.open_mainfile(filepath=file,load_ui=False);s=bpy.context.scene;assert s.frame_current==expected[tag];before=points();camera_before=cams();source_hash=hashlib.sha256(Path(file).read_bytes()).hexdigest()
    s.frame_start=s.frame_end=expected[tag];s.use_preview_range=False;s.frame_set(expected[tag]);s['replica_delivery_pose']=tag;s['replica_validated_frames']=str(expected[tag]);s['replica_temporal_status']='INDEPENDENT STATIC POSE; not a validated continuous animation';s['replica_render_unit_note']='Geometry numbers are millimeters, scene unit_scale=1; numeric light/DOF settings compensate. Not physical camera f-number or measured lamp watts.'
    s['replica_known_limits']='Approximate real pose/hardware; muzzle and head-bar white highlight unresolved; clean background and incomplete glass marks; no photorealism or GT claim.'
    assert s.render.resolution_x==1080 and s.render.resolution_y==1200;s.render.resolution_percentage=100;s.cycles.samples=128;s.cycles.use_denoising=True;devices=gpu();select('F');s.render.filepath=f'//renders/{labels[tag]}/camF_'
    hair=[o for o in s.objects if o.type=='CURVES'];assert len(hair)==9
    for o in hair:assert o.hide_viewport and not o.hide_render and all(m.show_render for m in o.modifiers)
    dependencies=[im for im in bpy.data.images if im.source in ['FILE','MOVIE','SEQUENCE','TILED']];assert len(dependencies)==132 and all(im.packed_file or im.packed_files for im in dependencies);assert len(bpy.data.libraries)==0
    note=bpy.data.texts.new('README_REALCAPTURE_DELIVERY')
    note.write(f'''PyMouseStudio RealCapture Replica — {labels[tag]}
Validated static frame: {expected[tag]}. This is one independent pose, not a motion sequence.
Default view: Replica_FR2_camF (perspective), 1080x1200, Cycles GPU/OptiX, 128 samples.
Hair/whisker CURVES are disabled in the viewport for responsiveness, enabled in renders.
All 132 image-file dependencies are packed. No linked .blend library is required.
Use camera objects Replica_FR2_camA..G. A is bottom; B..G are side cameras.
When changing cameras, set pixel_aspect_x=max(1,1/r), pixel_aspect_y=max(1,r),
where r is that camera object's custom property replica_pixel_aspect_y_over_x.
Geometry numerical coordinates are millimeters although scene unit_scale is 1.
Do not interpret numerical light powers/DOF f-stops as measured real apparatus specifications.
Approximate pose/hardware. Remaining issues: muzzle shape, implant endpoint highlight,
glass scratches/deposits, lighting/noise and exact silhouette mismatch.
Not a ground-truth reconstruction or publication-ready photorealistic dataset.
No image-space style transfer or private-data upload was used in this delivery.
Inherited source assets retain their existing rights; no blanket public redistribution license granted.
Original source and all iteration files were preserved. Full report is in the cycle output directory.
''')
    assert float(np.linalg.norm(points()-before,axis=1).max())==0 and cams()==camera_before
    target=delivery/f'PyMouseStudio-RealCapture-{labels[tag]}-Static.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    # Reload the file from disk, so rendering tests the saved artifact rather than unsaved memory.
    bpy.ops.wm.open_mainfile(filepath=str(target),load_ui=False);s=bpy.context.scene;devices=gpu();after=points();error=float(np.linalg.norm(after-before,axis=1).max());assert error==0 and cams()==camera_before;assert s.frame_current==s.frame_start==s.frame_end==expected[tag];assert all(o.hide_viewport and not o.hide_render for o in s.objects if o.type=='CURVES')
    folder=out/tag;folder.mkdir();item=dict(source=file,source_sha256=source_hash,delivery_blend=str(target),delivery_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frame=expected[tag],skin_max_change_after_reopen_mm=error,cameras_unchanged=True,packed_image_dependencies=132,libraries=0,devices=devices,resolution=[1080,1200],default_camera='F',seed=s.cycles.seed,animated_seed=s.cycles.use_animated_seed,renders=[]);manifest['poses'][tag]=item;write()
    for c in 'ABDF':
        guard();select(c);s.render.filepath=str(folder/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);item['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==source_hash
manifest['complete']=True;manifest['end_utc']=dt.datetime.now(dt.timezone.utc).isoformat();write();print('FINAL_DELIVERY_SAVED_REOPENED_RENDERED',flush=True)
