"""Compare linear/DQ skinning and rotation-minimizing tail frames from bind pose."""
import bpy,sys,json,time,datetime as dt
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/globals().get('OUTPUT_RELATIVE','output/pymousestudio_realism_cycle_20261002/v029_tail_transport');out.mkdir(exist_ok=False)
s=bpy.context.scene;arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse'];names=[f'Tail{i}_M' for i in range(12)];mods=[m for m in mouse.modifiers if m.type=='ARMATURE'];assert mods
original={n:arm.pose.bones[n].matrix.copy() for n in names};posed={n:arm.matrix_world@original[n] for n in names};rest={n:arm.matrix_world@arm.data.bones[n].matrix_local for n in names};p=np.array([posed[n].translation for n in names]);r=np.array([rest[n].translation for n in names]);inv=arm.matrix_world.inverted()
def frames(points,normal):
    tangents=[]
    for i in range(len(points)):
        t=Vector(points[min(i+1,len(points)-1)]-points[max(i-1,0)]).normalized();tangents.append(t)
    normal=Vector(normal);normal=(normal-tangents[0]*normal.dot(tangents[0])).normalized();result=[]
    for i,t in enumerate(tangents):
        if i:normal=tangents[i-1].rotation_difference(t)@normal;normal=(normal-t*normal.dot(t)).normalized()
        result.append(Matrix((t,normal,t.cross(normal))).transposed())
    return result
rf=frames(r,Vector((0,0,1)));root_deform=posed[names[0]].to_3x3()@rest[names[0]].to_3x3().inverted();pf=frames(p,root_deform@rf[0].col[1])
transport={n:Matrix.Translation(Vector(p[i]))@(pf[i]@rf[i].transposed()).to_4x4()@Matrix.Translation(Vector(-r[i]))@rest[n] for i,n in enumerate(names)}
groups={g.index:g.name for g in mouse.vertex_groups};dominant=[groups[max(v.groups,key=lambda g:g.weight).group] if v.groups else 'none' for v in mouse.data.vertices]
def geometry():
    ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();tail=[]
    for i,n in enumerate(names):
        ids=[j for j,d in enumerate(dominant) if d==n]
        if len(ids)<4:continue
        a=(q[ids]-p[i])@np.array(pf[i]);cross=a[:,1:];vals=np.linalg.eigvalsh(np.cov(cross.T));tail.append(dict(bone=n,transverse_cov_eigenvalues=vals.tolist(),aspect_proxy=float(np.sqrt(max(vals[-1],1e-10)/max(vals[0],1e-10)))))
    return dict(min_z=float(q[:,2].min()),below_floor=int((q[:,2]<339.5).sum()),below_floor025=int((q[:,2]<339.25).sum()),wall=int((np.linalg.norm(q[:,:2],axis=1)>56.470588).sum()),radial_max=float(np.linalg.norm(q[:,:2],axis=1).max()),tail_cross_section_proxy=tail)
m=dict(source=bpy.data.filepath,frame=s.frame_current,modifier_initial=[dict(name=x.name,preserve_volume=x.use_deform_preserve_volume) for x in mods],source_reference='https://docs.blender.org/manual/en/4.4/modeling/modifiers/deform/armature.html',candidates={},renders=[])
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=64;s.cycles.use_denoising=True;s.render.use_persistent_data=True
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
for tag,dq,reframe in globals().get('CANDIDATES',[('a_linear',False,False),('b_preserve_volume',True,False),('c_transport_dq',True,True)]):
    for n in names:arm.pose.bones[n].matrix=inv@transport[n] if reframe else original[n];bpy.context.view_layer.update()
    for mod in mods:mod.use_deform_preserve_volume=dq
    bpy.context.view_layer.update();geom=geometry();after=np.array([arm.matrix_world@arm.pose.bones[n].head for n in names]);geom['tail_centerline_max_error_mm']=float(np.linalg.norm(after-p,axis=1).max())
    prefix=globals().get('BLEND_PREFIX','PyMouseStudio-RealCapture-Replica-v029');target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/{prefix}-{tag}.blend';assert not target.exists();s.render.resolution_percentage=100;bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['candidates'][tag]=dict(blend=str(target),geometry=geom);write()
    for c in 'AD':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];ratio=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/ratio);s.render.pixel_aspect_y=max(1.,ratio);s.render.resolution_percentage=50;s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
