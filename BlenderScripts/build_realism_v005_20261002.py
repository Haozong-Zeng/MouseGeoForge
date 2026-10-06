"""Local articulated tail/finger correction on an independent action at frame31."""
import bpy,json,sys,time,math,datetime as dt
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002'
variant=globals().get('VARIANT','v005_local_contact');version=globals().get('BLEND_VERSION','v005');solver=globals().get('TAIL_SOLVER','projected')
out=base/variant;out.mkdir(exist_ok=False)
target=root/f'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-{version}.blend';assert not target.exists()
s=bpy.context.scene;s.frame_set(31);bpy.context.view_layer.update()
arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse']
original_action=arm.animation_data.action
arm.animation_data.action=original_action.copy();arm.animation_data.action.name='Replica_Climb_LocalContact_'+version
before_bases={b.name:b.matrix_basis.copy() for b in arm.pose.bones}
before_world={b.name:arm.matrix_world@b.matrix for b in arm.pose.bones}
def positions():
    ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();p=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return p
initial=positions();group_names={g.index:g.name for g in mouse.vertex_groups}
dominant=[group_names[max(v.groups,key=lambda g:g.weight).group] if v.groups else 'none' for v in mouse.data.vertices]
names=[f'Tail{i}_M' for i in range(12)]
old=np.array([before_world[n].translation for n in names]);lengths=np.linalg.norm(np.diff(old,axis=0),axis=1)
radius_clearance=[]
for i,n in enumerate(names):
    ids=[j for j,d in enumerate(dominant) if d==n]
    radius_clearance.append(max(.9,old[i,2]-float(initial[ids,2].min())+.3) if ids else .9)
floor=np.array(radius_clearance)+339.5
# Curve the distal chain laterally, then project centerline with fixed segment lengths.
p=old.copy();pivot=old[3,:2].copy()
for i in range(4,12):
    angle=math.radians(60)*((i-3)/8)**1.5
    r=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    p[i,:2]=pivot+r@(old[i,:2]-pivot);p[i,2]=max(p[i,2],floor[i])
for it in range(500):
    p[:4]=old[:4]
    for i in range(3,11):
        d=p[i+1]-p[i];norm=np.linalg.norm(d);corr=d*(norm-lengths[i])/max(norm,1e-9)
        if i==3:p[i+1]-=corr
        else:p[i]+=.5*corr;p[i+1]-=.5*corr
    p[4:,2]=np.maximum(p[4:,2],floor[4:])
    for i in range(4,12):
        r=np.linalg.norm(p[i,:2])
        if r>52:p[i,:2]*=52/r
if solver=='smooth_arc':
    p=old.copy();heading=math.atan2(old[3,1]-old[2,1],old[3,0]-old[2,0]);total=float(lengths[3:].sum());distance=0.
    for i in range(3,11):
        midpoint=distance+float(lengths[i])/2;u=midpoint/total
        theta=heading+math.radians(140)*(.5*u+.5*u*u)
        elevation=math.atan(-.34*math.exp(-midpoint/12))
        tangent=np.array([math.cos(theta)*math.cos(elevation),math.sin(theta)*math.cos(elevation),math.sin(elevation)])
        p[i+1]=p[i]+lengths[i]*tangent;distance+=float(lengths[i])
inv=arm.matrix_world.inverted()
for i,n in enumerate(names):
    if i<3:continue
    a=old[min(i+1,11)]-old[max(i-1,0)]
    b=p[min(i+1,11)]-p[max(i-1,0)]
    rotation=Vector(a).rotation_difference(Vector(b)).to_matrix().to_4x4()
    world=Matrix.Translation(Vector(p[i]))@rotation@Matrix.Translation(Vector(-old[i]))@before_world[n]
    arm.pose.bones[n].matrix=inv@world;bpy.context.view_layer.update()
tail_before_refine=positions()
# Move the tail skin safely above floor by small per-joint lifts; keep centerline audit explicit.
refinement=[]
for iteration in range(4):
    current=positions();lifts=[]
    for i,n in enumerate(names):
        ids=[j for j,d in enumerate(dominant) if d==n]
        lifts.append(max(0.,339.6-float(current[ids,2].min())) if ids and i>=4 else 0.)
    if max(lifts)<.05:break
    worlds={n:arm.matrix_world@arm.pose.bones[n].matrix for n in names}
    for i,n in enumerate(names):
        if i<4:continue
        m=worlds[n].copy();m.translation.z+=lifts[i]
        arm.pose.bones[n].matrix=inv@m;bpy.context.view_layer.update()
    refinement.append(lifts)
# Curl right palm and all descendants around its base, leaving the forearm and left paw intact.
palm=arm.pose.bones['Fingers1_R'];subtree=[palm]+list(palm.children_recursive)
subtree.sort(key=lambda b:len(b.parent_recursive))
palm_world={b.name:arm.matrix_world@b.matrix for b in subtree};pivot=np.array(palm_world[palm.name].translation)
finger_ids=[v.index for v in mouse.data.vertices if sum(g.weight for g in v.groups if ('Finger' in group_names[g.group] and group_names[g.group].endswith('_R')))>.65]
current=positions();distal=current[finger_ids][np.linalg.norm(current[finger_ids,:2],axis=1).argsort()[-50:]].mean(0)
inward=np.r_[-distal[:2]/np.linalg.norm(distal[:2]),0.]
axis=np.cross(distal-pivot,inward);axis/=np.linalg.norm(axis)
trials=[];chosen=None
for degrees in range(0,46,2):
    transform=Matrix.Translation(Vector(pivot))@Matrix.Rotation(math.radians(degrees),4,Vector(axis))@Matrix.Translation(Vector(-pivot))
    for b in subtree:
        b.matrix=inv@transform@palm_world[b.name];bpy.context.view_layer.update()
    q=positions();maximum=float(np.linalg.norm(q[finger_ids,:2],axis=1).max());trials.append([degrees,maximum])
    if maximum<=80*60/85-.15:chosen=degrees;break
assert chosen is not None,('Palm collision unresolved',trials)
modified=[b for b in arm.pose.bones if not np.allclose(np.array(before_bases[b.name]),np.array(b.matrix_basis),atol=1e-7)]
for b in modified:
    b.keyframe_insert('location',frame=31);b.keyframe_insert('rotation_quaternion',frame=31);b.keyframe_insert('scale',frame=31)
s.frame_set(30);s.frame_set(31);bpy.context.view_layer.update();final=positions()
global_ground_lift=0.
if solver=='smooth_arc':
    # Submillimetric whole-rig lift eliminates the inherited right-foot skin penetration.
    global_ground_lift=max(0.,339.51-float(final[:,2].min()))
    bpy.data.objects['Replica_GlobalMousePlacement'].location.z+=global_ground_lift;bpy.context.view_layer.update();final=positions()
after=np.array([arm.matrix_world@arm.pose.bones[n].head for n in names])
foot_ids=[i for i,n in enumerate(dominant) if 'Toe' in n or 'Ankle' in n]
counts={}
for name in sorted(set(dominant)):
    ids=[i for i,n in enumerate(dominant) if n==name];q=final[ids]
    low=int((q[:,2]<339.25).sum());wall=int((np.linalg.norm(q[:,:2],axis=1)>80*60/85).sum())
    if low or wall:counts[name]=dict(below_floor_025=low,outside_wall=wall)
measure=dict(skin_min_z=float(final[:,2].min()),foot_min_z=float(final[foot_ids,2].min()),radial_max=float(np.linalg.norm(final[:,:2],axis=1).max()),
    vertices_below_floor_025=int((final[:,2]<339.25).sum()),vertices_below_floor_1mm=int((final[:,2]<338.5).sum()),vertices_outside_wall=int((np.linalg.norm(final[:,:2],axis=1)>80*60/85).sum()),
    offending_groups=counts,tail_segment_lengths_before=lengths.tolist(),tail_segment_lengths_after=np.linalg.norm(np.diff(after,axis=0),axis=1).tolist(),
    tail_head_positions=after.tolist(),tail_refinement_lifts=refinement,tail_solver=solver,global_ground_lift_mm=global_ground_lift,palm_rotation_degrees=chosen,palm_trials=trials,modified_bones=[b.name for b in modified],keyframe_roundtrip_verified=True)
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.render.engine='CYCLES';s.cycles.device='GPU';s.cycles.samples=24;s.cycles.use_denoising=True
s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_compositing=False;s.render.use_sequencer=False;s.render.use_persistent_data=True;s.render.image_settings.file_format='PNG'
def camera(c):
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
camera('F');bpy.ops.wm.save_as_mainfile(filepath=str(target));records=[]
m=dict(blend=str(target),frame=31,original_action=original_action.name,new_action=arm.animation_data.action.name,geometry=measure,devices=devices,renders=records,
    limitations=['Only frame31 corrected. Adjacent frames of the copied source action are NOT validated.','Tail centerline constrained before skin refinement; actual segment-length changes reported, bone scale not intentionally changed.','Hair/whisker collisions and exact target pose are not certified. Materials and lighting unchanged.'])
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write();print('V005_GEOMETRY',json.dumps(measure),flush=True);s.render.resolution_percentage=50
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    camera(c);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True)
    records.append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write();print('V005_COMPLETE',c,flush=True)
