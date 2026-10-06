"""Reversible monotone lower-lip compression in rest space, no pose/camera edits."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
root=Path(sys.argv[sys.argv.index('--')+1]);version=globals().get('TRIAL_VERSION','v051');out=root/'output/pymousestudio_realism_cycle_20261002'/globals().get('OUTPUT_NAME','v051_closed_mouth');out.mkdir(exist_ok=False);s=bpy.context.scene;ob=bpy.data.objects['MS_Mouse'];assert ob.data.shape_keys is None
rest=np.array([v.co[:] for v in ob.data.vertices]);cams={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)
# Rest +Y is dorsal/up; front is +Z. Positive derivative in Y at max strength avoids a folded rest-space map.
fall=smooth((rest[:,2]-3.05)/.4)*smooth((4.3-rest[:,2])/.25)*smooth((.8-np.abs(rest[:,0]))/.4)*smooth((1.2-rest[:,1])/.5)
delta=np.zeros_like(rest);delta[:,1]=.24*fall;selected=np.linalg.norm(delta,axis=1)>1e-9
mapping='Y += .24 * smoothstep((Z-3.05)/.4) * smoothstep((4.3-Z)/.25) * smoothstep((.8-abs(X))/.4) * smoothstep((1.2-Y)/.5); shape value .5 or1. Minimum dYnew/dY >= .28 at value1.'
if globals().get('DELTA_MODE')=='fill_concavity':
    fall=smooth((rest[:,2]-3.25)/.35)*smooth((4.35-rest[:,2])/.23)*smooth((.55-np.abs(rest[:,0]))/.43)*smooth((rest[:,1]-.82)/.45)*smooth((1.7-rest[:,1])/.3)
    delta[:,1]=-.27*fall;selected=np.linalg.norm(delta,axis=1)>1e-9
    mapping='Y -= .27 * smoothstep((Z-3.25)/.35) * smoothstep((4.35-Z)/.23) * smoothstep((.55-abs(X))/.43) * smoothstep((Y-.82)/.45) * smoothstep((1.7-Y)/.3). Move central mouth-concavity surface ventrally, not lift lips. Minimum dYnew/dY >= .1.'
def skin():
    bpy.context.view_layer.update();ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in mesh.vertices]);normals=np.array([p.normal[:] for p in mesh.polygons]);faces=[list(p.vertices) for p in mesh.polygons];ev.to_mesh_clear();return q,normals,faces
before,norm0,faces=skin()
def hair(q):
    bvh=BVHTree.FromPolygons(q.tolist(),faces);rows=[]
    for name in ['Mouth.002','Whisker.002','BaseFur.006']:
        hair_ob=bpy.data.objects[name];old=hair_ob.hide_viewport;old_set=hair_ob.hide_get();hair_ob.hide_viewport=False;hair_ob.hide_set(False);bpy.context.view_layer.update();ev=hair_ob.evaluated_get(bpy.context.evaluated_depsgraph_get());distances=[]
        for i in np.linspace(0,len(ev.data.curves)-1,min(256,len(ev.data.curves))).astype(int):
            p=ev.matrix_world@ev.data.curves[int(i)].points[0].position;distances.append(float(bvh.find_nearest(p)[3]))
        rows.append(dict(object=name,curves=len(ev.data.curves),sample_count=len(distances),root_distance_quantiles_mm=np.quantile(distances,[0,.5,.9,1]).tolist()));hair_ob.hide_viewport=old;hair_ob.hide_set(old_set)
    bpy.context.view_layer.update();return rows
baseline_hair=hair(before);basis=ob.shape_key_add(name='Basis');key=ob.shape_key_add(name='Replica_ClosedMouth_'+version)
for i,p in enumerate(key.data):p.co=rest[i]+delta[i]
key.slider_min=0;key.slider_max=1;key.value=0
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
m=dict(source=bpy.data.filepath,frame=s.frame_current,selected_vertices=int(selected.sum()),rest_delta_max_local=float(np.linalg.norm(delta,axis=1).max()),mapping=mapping,baseline_hair=baseline_hair,devices=devices,candidates={},renders=[],limitations=['Local shape trial, not anatomical ground truth.','Positive rest-map derivative is not a full posed surface intersection certificate.','Hair root checks are fixed sparse samples, not exhaustive surface collision.'])
np.savez_compressed(out/'shape_delta.npz',rest=rest,delta=delta,selected=selected)
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
for tag,value in globals().get('CANDIDATE_VALUES',[('a_half',.5),('b_full',1.)]):
    key.value=value;ob.data.update();after,norm,faces=skin();changes=np.linalg.norm(after-before,axis=1);assert float(changes[~selected].max())<.00001
    for c,x in cams.items():assert x==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world]
    geometry=dict(max_change_mm=float(changes.max()),unselected_max_change_mm=float(changes[~selected].max()),reversed_face_normals=int(((norm*norm0).sum(1)<0).sum()),min_skin_z=float(after[:,2].min()),floor_vertices=int((after[:,2]<339.5-.001).sum()),wall_vertices=int((np.linalg.norm(after[:,:2],axis=1)>56.470588+.001).sum()),camera_matrices_unchanged=True)
    hairs=hair(after);target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-{version}-{tag}.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True);m['candidates'][tag]=dict(blend=str(target),shape_value=value,geometry=geometry,hair=hairs);write()
    for c in 'ABF':
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
        s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'{tag}_cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(tag=tag,camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
