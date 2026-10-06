"""Separate room illumination from high-contrast animal direct lighting."""
import bpy,json,sys,time,math,datetime as dt
from pathlib import Path
from mathutils import Vector
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v019_room_balance';out.mkdir(exist_ok=False)
s=bpy.context.scene;s.frame_set(31);source=bpy.data.filepath
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=48;s.cycles.use_denoising=True;s.render.resolution_percentage=50;s.render.use_persistent_data=True
m=dict(source_blend=source,frame=31,devices=devices,renders=[],candidate_blends={},stages=[],limitations=['Room albedo and light power are hypotheses, not measured apparatus photometry.','Numerical scene is millimeters with unit_scale1; powers not actual measured watts.','No geometry, pose or camera modification.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
for stage,views in [('a_wall_albedo','AF'),('b_wall_lights','AF'),('c_direct_half','ABDF'),('d_skin_balance','AF')]:
    changes=[]
    if stage=='a_wall_albedo':
        ob=bpy.data.objects['Part_WhiteHexagonBackground'];mat=ob.data.materials[0].copy();mat.name='Replica_Backdrop_Gray045';ob.data.materials[0]=mat;shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        for link in list(shader.inputs['Base Color'].links):mat.node_tree.links.remove(link)
        shader.inputs['Base Color'].default_value=(.45,.45,.45,1);shader.inputs['Roughness'].default_value=.85;shader.inputs['Emission Strength'].default_value=0;changes.append(dict(backdrop_albedo=.45))
    elif stage=='b_wall_lights':
        for i,degrees in enumerate(range(0,360,60)):
            theta=math.radians(degrees);direction=Vector((math.cos(theta),math.sin(theta),0));data=bpy.data.lights.new(f'Replica_WallWash_{i}','SPOT');data.energy=1.5e6;data.spot_size=math.radians(110);data.spot_blend=.8;data.shadow_soft_size=10
            ob=bpy.data.objects.new(data.name,data);s.collection.objects.link(ob);ob.location=direction*180+Vector((0,0,620));target=direction*600+Vector((0,0,430));ob.rotation_euler=(target-ob.location).to_track_quat('-Z','Y').to_euler();changes.append(dict(name=ob.name,energy=data.energy,position=list(ob.location),target=list(target)))
    elif stage=='c_direct_half':
        for name in ['Replica_KeyPanel','Replica_RimPanel','Replica_BottomFill']:
            data=bpy.data.objects[name].data;before=data.energy;data.energy*=.5;changes.append(dict(name=name,before=before,after=data.energy))
    else:
        mouse=bpy.data.objects['MS_Mouse']
        for i,gray in [(2,.30),(3,.30),(4,.12),(5,.30)]:
            mat=mouse.data.materials[i].copy();mat.name=f'Replica_BalancedSkin_{i}';mouse.data.materials[i]=mat;shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');shader.inputs['Base Color'].default_value=(gray,gray,gray,1);changes.append(dict(slot=i,albedo=gray))
    m['stages'].append(dict(stage=stage,changes=changes));write()
    for c in views:
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
        s.render.filepath=str(out/f'{stage}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(stage=stage,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
    target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v019-{stage}.blend';assert not target.exists()
    s.render.resolution_percentage=100;s.cycles.samples=64;bpy.ops.wm.save_as_mainfile(filepath=str(target));m['candidate_blends'][stage]=str(target);write();s.render.resolution_percentage=50;s.cycles.samples=48
