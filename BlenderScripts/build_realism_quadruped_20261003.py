"""Independent static quadruped initialization from the same original source rig."""
import bpy,json,sys,time,math,datetime as dt
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
from mathutils.bvhtree import BVHTree
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/globals().get('OUT_TAG','v024_quadruped');out.mkdir(exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath
arm=bpy.data.objects['DeformationSystem'];mouse=bpy.data.objects['MS_Mouse'];placement=bpy.data.objects['Replica_GlobalMousePlacement'];placement.matrix_world=Matrix.Identity(4)
audit=json.loads((base/'source_pose_range.json').read_text());source_blend=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-BrownFur-ClimbWall.blend'
with bpy.data.libraries.load(str(source_blend),link=False) as (data_from,data_to):
    assert audit['action'] in data_from.actions;data_to.actions=[audit['action']]
action=data_to.actions[0];arm.animation_data.action=action
if len(action.slots):arm.animation_data.action_slot=action.slots[0]
pose_frame=globals().get('POSE_FRAME',111);session=globals().get('REFERENCE_SESSION','FR2');real_frame=globals().get('REFERENCE_FRAME',1104)
s.frame_set(pose_frame);bpy.context.view_layer.update();ref=next(r for r in audit['sampled_frames'] if r['frame']==pose_frame);errors={n:float(np.linalg.norm(np.array(arm.matrix_world@arm.pose.bones[n].head)-p)) for n,p in ref['bones'].items()};assert max(errors.values())<.002,errors
# Freeze the selected pose in this independent project, not a new temporal sequence.
bases={b.name:b.matrix_basis.copy() for b in arm.pose.bones};arm.animation_data.action=None
for b in arm.pose.bones:b.matrix_basis=bases[b.name]
for track in arm.animation_data.nla_tracks:track.mute=True
bpy.context.view_layer.update()
ranking=json.loads((base/globals().get('RANKING_FILE','pose_seed_fixedscale_ranking.json')).read_text());entry=next(r for r in ranking['results'] if r['session']==session);seed=next(r for r in entry['top_candidates'] if r['source_frame']==pose_frame);placement.matrix_world=Matrix(seed['global_matrix']);bpy.context.view_layer.update()
for bone_name,angle in globals().get('LOCAL_PITCHES',[]):
    pb=arm.pose.bones[bone_name];pivot=arm.matrix_world@pb.head;forward=arm.matrix_world@arm.pose.bones['HeadEnd_M'].head-pivot;axis=forward.cross(Vector((0,0,1))).normalized();rot=Quaternion(axis,math.radians(angle)).to_matrix().to_4x4()
    pb.matrix=arm.matrix_world.inverted()@Matrix.Translation(pivot)@rot@Matrix.Translation(-pivot)@arm.matrix_world@pb.matrix;bpy.context.view_layer.update()
groups={g.index:g.name for g in mouse.vertex_groups};dominant=[groups[max(v.groups,key=lambda g:g.weight).group] if v.groups else 'none' for v in mouse.data.vertices]
body_ids=np.array([i for i,n in enumerate(dominant) if not n.startswith('Tail') or n in ['Tail0_M','Tail1_M','Tail2_M']])
foot_ids={}
for side in ['L','R']:
    for kind in ['fore','hind']:
        parts=['Finger','Wrist'] if kind=='fore' else ['Toe','Ankle'];foot_ids[kind+side]=np.array([v.index for v in mouse.data.vertices if sum(g.weight for g in v.groups if groups[g.group].endswith('_'+side) and any(p in groups[g.group] for p in parts))>.65])
def positions():
    ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();p=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return p
p=positions();allfoot=np.concatenate(list(foot_ids.values()));ground=339.51-float(p[allfoot,2].min());placement.location.z+=ground;bpy.context.view_layer.update();p=positions()
# Common translation only; prefer the smallest correction keeping body and tail root inside.
deltas=sorted([(x*x+y*y,x,y) for x in np.arange(-25,25.01,.5) for y in np.arange(-25,25.01,.5)]);best=None
for cost,x,y in deltas:
    maximum=float(np.linalg.norm(p[body_ids,:2]+[x,y],axis=1).max())
    if best is None or maximum<best[0]:best=(maximum,x,y)
    if maximum<=54.5:best=(maximum,x,y);break
placement.location.x+=best[1];placement.location.y+=best[2];bpy.context.view_layer.update();p=positions()
tail_names=[f'Tail{i}_M' for i in range(12)];world={n:arm.matrix_world@arm.pose.bones[n].matrix for n in tail_names};old=np.array([world[n].translation for n in tail_names]);lengths=np.linalg.norm(np.diff(old,axis=0),axis=1);inv=arm.matrix_world.inverted();radii=[]
for i,n in enumerate(tail_names):
    ids=[j for j,k in enumerate(dominant) if k==n];t=old[min(i+1,11)]-old[max(i-1,0)];t/=np.linalg.norm(t);v=p[ids]-old[i];perp=v-np.sum(v*t,axis=1)[:,None]*t
    radii.append(float(np.clip(np.quantile(np.linalg.norm(perp,axis=1),.95)+.15,.45,4.5)) if ids else .8)
body_polys=[list(poly.vertices) for poly in mouse.data.polygons if not any(dominant[j].startswith('Tail') for j in poly.vertices)];bvh=BVHTree.FromPolygons([Vector(v) for v in p],body_polys,all_triangles=False)
heading=math.atan2(old[2,1]-old[1,1],old[2,0]-old[1,0]);candidates=[]
for angle in list(range(-300,-59,10))+list(range(60,301,10)):
    points=old.copy();travel=0.;total=float(lengths[2:].sum())
    for i in range(2,11):
        u=(travel+lengths[i]/2)/total;theta=heading+math.radians(angle)*u;target_z=339.5+radii[i+1]+.15;elev=math.asin(float(np.clip((target_z-points[i,2])/lengths[i],-.5,.5)))
        points[i+1]=points[i]+lengths[i]*np.array([math.cos(theta)*math.cos(elev),math.sin(theta)*math.cos(elev),math.sin(elev)]);travel+=lengths[i]
    wall=float(np.maximum(np.linalg.norm(points[3:,:2],axis=1)+np.array(radii[3:])-56.2,0).max());body=0.;selfhit=0.
    for i in range(3,12):
        loc,norm,_,distance=bvh.find_nearest(Vector(points[i]));signed=float((Vector(points[i])-loc).dot(norm));body+=max(0,radii[i]-signed)
        for j in range(3,i-2):selfhit+=max(0,radii[i]+radii[j]+.3-np.linalg.norm(points[i]-points[j]))
    score=100*wall+15*body+20*selfhit+.003*abs(angle);candidates.append(dict(angle=angle,score=score,wall=wall,body_penalty=body,self_penalty=selfhit,points=points))
candidates.sort(key=lambda x:x['score']);trials=[];selected=None
if globals().get('WALL_AWARE',False):
    # The first attempt fixed Tail0..2 and forced the curve out through the wall.
    # Keep only the anatomical root position fixed; permit a smooth root turn.
    candidates=[];heading=math.atan2(old[1,1]-old[0,1],old[1,0]-old[0,0]);total=float(lengths.sum())
    for initial in range(-120,121,15):
        for turn in range(-240,241,15):
            points=old.copy();travel=0.
            for i in range(11):
                u=(travel+lengths[i]/2)/total;theta=heading+math.radians(initial+turn*u)
                target_z=339.5+radii[i+1]+.15+globals().get('TAIL_EXTRA_CLEARANCE',0.);elev=math.asin(float(np.clip((target_z-points[i,2])/lengths[i],-.65,.65)))
                points[i+1]=points[i]+lengths[i]*np.array([math.cos(theta)*math.cos(elev),math.sin(theta)*math.cos(elev),math.sin(elev)]);travel+=lengths[i]
            wall=float(np.maximum(np.linalg.norm(points[1:,:2],axis=1)+np.array(radii[1:])-55.8,0).sum());body=0.;selfhit=0.
            for i in range(2,12):
                loc,norm,_,distance=bvh.find_nearest(Vector(points[i]));signed=float((Vector(points[i])-loc).dot(norm));body+=max(0,radii[i]+.5-signed)
                for j in range(2,i-2):selfhit+=max(0,radii[i]+radii[j]+1.-np.linalg.norm(points[i]-points[j]))
            score=100*wall+50*body+30*selfhit+.01*abs(initial)+.003*abs(turn)
            candidates.append(dict(angle=turn,initial=initial,score=score,wall=wall,body_penalty=body,self_penalty=selfhit,points=points))
    candidates.sort(key=lambda x:x['score'])
def apply_tail(points):
    for i,n in enumerate(tail_names):
        if i<3 and not globals().get('WALL_AWARE',False):continue
        a=old[min(i+1,11)]-old[max(i-1,0)];b=points[min(i+1,11)]-points[max(i-1,0)];rot=Vector(a).rotation_difference(Vector(b)).to_matrix().to_4x4();arm.pose.bones[n].matrix=inv@Matrix.Translation(Vector(points[i]))@rot@Matrix.Translation(Vector(-old[i]))@world[n];bpy.context.view_layer.update()
for candidate in candidates[:24 if globals().get('WALL_AWARE',False) else 8]:
    apply_tail(candidate['points']);q=positions();wall=int((np.linalg.norm(q[:,:2],axis=1)>56.470588).sum());below=int((q[:,2]<339.25).sum());trial=dict(angle=candidate['angle'],wall_vertices=wall,below_floor_vertices=below,proxy_score=candidate['score']);trials.append(trial)
    rank=(wall+below,candidate['score'])
    if selected is None or rank<selected[0]:selected=(rank,candidate)
apply_tail(selected[1]['points']);q=positions();after=np.array([arm.matrix_world@arm.pose.bones[n].head for n in tail_names])
geometry=dict(skin_min_z=float(q[:,2].min()),radial_max=float(np.linalg.norm(q[:,:2],axis=1).max()),wall_vertices=int((np.linalg.norm(q[:,:2],axis=1)>56.470588).sum()),below_floor_vertices_025=int((q[:,2]<339.25).sum()),feet={k:dict(min_z=float(q[ids,2].min()),p01_z=float(np.quantile(q[ids,2],.01)),p50_z=float(np.median(q[ids,2]))) for k,ids in foot_ids.items()},tail_length_max_error_mm=float(np.max(np.abs(np.linalg.norm(np.diff(after,axis=0),axis=1)-lengths))))
s.frame_start=pose_frame;s.frame_end=pose_frame;s['replica_pose_reference']=f'{session} frame{real_frame} approximate static pose; not temporal data';s['replica_source_pose_frame']=pose_frame
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=48;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002'/globals().get('BLEND_NAME','PyMouseStudio-RealCapture-Replica-v024-quadruped-candidate.blend');assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target))
m=dict(source_working_blend=source,candidate_blend=str(target),source_action_blend=str(source_blend),source_pose_frame=111,static_frame=111,reference_session='FR2',reference_frame=1104,source_pose_roundtrip_errors_mm=errors,seed=seed,ground_lift_mm=ground,body_xy_shift_mm=[best[1],best[2]],global_placement=[list(row) for row in placement.matrix_world],tail_turn_degrees=selected[1]['angle'],tail_trials=trials,tail_clearance_radii_mm=radii,geometry=geometry,devices=devices,renders=[],limitations=['Static independent pose, not a continuous temporal sequence.','Same original rig action used as geometric initialization; local tail corrected, real diagnostic3D NOTGT.','Skin vertex checks not full surface intersection or hair/whisker certification.','Same scale and materials as accepted climb; silhouette/reference fit still needs visual review.'])
m.update(source_pose_frame=pose_frame,static_frame=pose_frame,reference_session=session,reference_frame=real_frame,local_pitches=globals().get('LOCAL_PITCHES',[]))
m['tail_initial_turn_degrees']=selected[1].get('initial',0)
m['tail_root_only_fixed']=bool(globals().get('WALL_AWARE',False))
def write():(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
write();print('V024_GEOMETRY',json.dumps(geometry),flush=True)
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.resolution_percentage=50;s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
