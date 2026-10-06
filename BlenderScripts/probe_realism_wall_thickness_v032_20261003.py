"""Thickness-only optical diagnostic; thin-wall candidates need renewed paw contact."""
import bpy,sys,json,time,datetime as dt
import numpy as np
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002'/globals().get('OUT_TAG','v032_wall_thickness');out.mkdir(exist_ok=False);s=bpy.context.scene;cylinder=bpy.data.objects['Part_GlassCylinder160mm'];original=np.array([v.co for v in cylinder.data.vertices]);rad=np.linalg.norm(original[:,:2],axis=1)
(out/'mesh_audit.json').write_text(json.dumps(dict(local_radii_unique=np.unique(np.round(rad,6)).tolist(),matrix_world=[list(r) for r in cylinder.matrix_world],bounds=[list(v) for v in cylinder.bound_box]),indent=2));assert rad.min()>0 and rad.min()/rad.max()>.8
sx=float(cylinder.matrix_world.to_scale().x);sy=float(cylinder.matrix_world.to_scale().y);assert abs(sx-sy)<1e-5;outer=float(rad.max());inner=float(rad.min());original_thickness=(outer-inner)*sx
assert abs(outer*sx-60)<.05,(outer,inner,sx)
mouse=bpy.data.objects['MS_Mouse'];ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();p=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();max_mouse_r=float(np.linalg.norm(p[:,:2],axis=1).max())
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
m=dict(source=bpy.data.filepath,original_radial_range_local=[inner,outer],original_thickness_mm=original_thickness,outer_radius_mm=outer*sx,local_radii_unique=np.unique(np.round(rad,3)).tolist(),max_mouse_radius_mm=max_mouse_r,devices=devices,candidates={},renders=[],limitations=['Thickness is an unmeasured hypothesis, not recovered hardware geometry.','Reducing wall thickness while keeping outer radius fixed moves the inner contact surface outward. Mouse unchanged for optical isolation: thin candidates lose exact climbing-paw contact and cannot be promoted directly.','Calibrated camera poses/K unchanged; refracted image positions can still change. Not a dataset-label export.'])
cam=bpy.data.objects['Replica_FR2_camF'].data;m['dof_api_audit']=dict(sensor_width=cam.sensor_width,lens=cam.lens,fstop_min=cam.dof.bl_rna.properties['aperture_fstop'].hard_min,fstop_max=cam.dof.bl_rna.properties['aperture_fstop'].hard_max,sensor_width_max=cam.bl_rna.properties['sensor_width'].hard_max,unit_scale=s.unit_settings.scale_length)
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
write()
for tag,thickness in [('a_control',original_thickness),('b_1p5mm',1.5),('c_0p8mm',.8)]:
    newrad=outer-(outer-rad)*(thickness/original_thickness);coords=original.copy();coords[:,:2]*=(newrad/rad)[:,None]
    for v,co in zip(cylinder.data.vertices,coords):v.co=co
    cylinder.data.update();bpy.context.view_layer.update()
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v032-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    m['candidates'][tag]=dict(blend=str(target),thickness_mm=thickness,inner_radius_mm=outer*sx-thickness,closest_skin_wall_clearance_mm=outer*sx-thickness-max_mouse_r);write()
    for c in 'FD':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];ratio=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/ratio);s.render.pixel_aspect_y=max(1.,ratio);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
