"""Final physical-geometry hypothesis: small true edge bevels, MNEE reception retained."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v063_implant_bevel';out.mkdir(exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath;ob=bpy.data.objects['ReplicaAsset_立方体'];assert ob.cycles.is_caustics_receiver
scale=[ob.matrix_world.to_3x3().col[i].length for i in range(3)];assert max(scale)/min(scale)<1.0001
mouse=bpy.data.objects['MS_Mouse'];bones={p.name:[list(r) for r in p.matrix] for p in bpy.data.objects['DeformationSystem'].pose.bones};cam_before=[list(r) for r in bpy.data.objects['Replica_FR2_camF'].matrix_world]
def pts():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();p=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return p
before=pts();bevel=ob.modifiers.new('Replica Physical Edge Bevel v063','BEVEL');bevel.segments=5;bevel.limit_method='ANGLE';bevel.angle_limit=.5
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True;s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
m=dict(source=source,world_scale=scale,frame=s.frame_current,devices=devices,bar_caustics_receiver=True,candidates={},renders=[],limitations=['Edge-radius hypothesis only, not a measured implant shape.', 'Uses actual bevel modifier geometry, not Bevel shader node; documented MNEE Bevel-node limitation avoided.', 'Receiver remains enabled, so MNEE curved-glass brightness inaccuracies can persist.'])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
for tag,width in [('a_radius015mm',.15),('b_radius040mm',.4)]:
    bevel.width=width/sum(scale)*3;after=pts();assert float(np.linalg.norm(after-before,axis=1).max())==0;assert cam_before==[list(r) for r in s.camera.matrix_world];assert all(bones[p.name]==[list(r) for r in p.matrix] for p in bpy.data.objects['DeformationSystem'].pose.bones)
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v063-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['candidates'][tag]=dict(blend=str(target),world_bevel_width_mm=width,skin_max_change_mm=0,bones_unchanged=True,camera_unchanged=True);write()
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc);s.render.filepath=str(out/f'{tag}_camF.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,file=s.render.filepath,seconds=time.monotonic()-t));write()
