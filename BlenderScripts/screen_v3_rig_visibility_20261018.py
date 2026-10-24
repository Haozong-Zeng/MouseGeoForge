"""Model-free deterministic rig rejection sampling on admitted TRAIN motions.

Same geometric observability rule for TRAIN/development/sealed camera configs.
No RGB rendering, foundation inference, labels from TEST, or scene saving.
"""
import argparse,datetime as dt,hashlib,json,math,random,sys,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_visible_surface_truth_20261007 import geometry,trace


def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    root=Path(a.root);out=root/'output/bottom_geometry_v3_local12h_20261018';design=json.loads((out/'render_preparation_attempt0/render_design.json').read_text());cache=[]
    hard=dt.datetime.fromisoformat(json.loads((out/'plan.json').read_text())['optimization_deadline_utc'])
    for source in design['sources']:
        assert source['motion_group'] in ('mouse_climb_wall_91','moseq_sample10_all_correlated_variants')
        bpy.ops.wm.open_mainfile(filepath=str(root/source['path']));scene=bpy.context.scene
        for collection in bpy.data.collections:
            if collection.name.startswith('CameraItems'):collection.hide_render=True
        for obj in bpy.data.objects:
            if obj.type=='CURVES':obj.hide_viewport=True
            if obj.name.startswith(('Part_Camera','Part_CamLen','CameraItems','Part_PiCam','Part_CNC-Camera','Part_BottomCamScrew','Part_SideCamScrew')):
                obj.hide_render=True
                for child in obj.children_recursive:child.hide_render=True
        for f in source['frames']:
            scene.frame_set(f);bpy.context.view_layer.update()
            tree,_,_,_,instances,_,_=geometry(scene,dict(meshes=[source['mesh']],source_depth_policy={}), 'V3_VISIBILITY_QA')
            cache.append(dict(source=source['id'],frame=f,anchor=source['anchor'],tree=tree,instances=instances))
            print('VIS_CACHE',source['id'],f,flush=True)
    side=96;yy,xx=np.mgrid[:side,:side];uv=np.stack((xx+.5,yy+.5,np.ones_like(xx)),-1).reshape(-1,3);cv=Matrix(((1,0,0),(0,-1,0),(0,0,-1)))
    def check(rig):
        result=[];fx=side*rig['lens_mm']/36.;K=np.array([[fx,0,side/2],[0,fx,side/2],[0,0,1.]])
        rays_camera=uv@np.linalg.inv(K).T
        for item in cache:
            target=Vector(item['anchor'])
            for i,relative in enumerate(rig['relative_positions']):
                loc=target+Vector(relative);q=(target-loc).to_track_quat('-Z','Y')@Quaternion((0,0,1),math.radians(rig['roll_deg'][i]))
                R=np.array(cv@q.to_matrix().transposed());rays=rays_camera@R;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
                _,_,faces=trace(item['tree'],np.array(loc),rays,3000);valid=faces>=0;body=np.zeros(len(faces),bool);body[valid]=item['instances'][faces[valid]]>0
                result.append(dict(source=item['source'],frame=item['frame'],view=i,body_pixels=int(body.sum())))
        return result
    def candidate(old,attempt):
        seed=int(hashlib.sha256(f"visibility_v1:{old['id']}:{attempt}".encode()).hexdigest()[:16],16);rng=random.Random(seed);r=dict(old)
        extrap=r.get('dev_tier')=='extrapolation';index=int(r['id'][-2:])
        r['lens_mm']=rng.uniform(26,29) if extrap and index%2 else rng.uniform(56,60) if extrap else rng.uniform(30,55)
        theta=rng.uniform(-math.pi,math.pi);radius=rng.uniform(105,125) if extrap else rng.uniform(0,95)
        height=rng.uniform(250,280) if extrap else rng.uniform(150,245)
        a=rng.uniform(0,2*math.pi);d=rng.uniform(215,290);angles=[a,a+rng.uniform(1.5,2.7)]
        r['relative_positions']=[[radius*math.cos(theta),radius*math.sin(theta),-height]]+[[d*math.cos(t),d*math.sin(t),rng.uniform(45,110)] for t in angles]
        r['roll_deg']=[rng.uniform(-30,30),rng.uniform(-8,8),rng.uniform(-8,8)]
        return r
    accepted=[];rows=[]
    for old in design['rigs']:
        attempts=[]
        for attempt in range(33):
            if (out/'STOP').exists() or dt.datetime.now(dt.timezone.utc)>=hard:raise RuntimeError('Finite preparation cutoff')
            rig=old if attempt==0 else candidate(old,attempt);counts=check(rig);good=min(v['body_pixels'] for v in counts)>=16
            attempts.append(dict(attempt=attempt,rig=rig,views=counts,passed=good))
            if good:accepted.append(rig);break
        else:raise RuntimeError('No admissible camera config within32fixed resamples: '+old['id'])
        rows.append(dict(id=old['id'],changed=attempt>0,attempts=attempts))
        write(out/'visibility_prescreen_status.json',dict(completed=len(rows),total=len(design['rigs']),last_rig=old['id'],resamples=attempt))
        print('VIS_RIG',old['id'],attempt,min(v['body_pixels'] for v in counts),flush=True)
    write(out/'visibility_prescreen.json',dict(passed=True,rigs=accepted,rows=rows,model_calls=0,RGB_renders=0,
        sources=design['sources'],rule='96x96 primary-ray GT visibility on all2motions x3frames x3views; at least16body pixels each; unchanged final384px QA requires>=100. Max32fixed seeded resamples; same policy fortrain/dev/sealed.',
        changed_rigs=[r['id'] for r in rows if r['changed']],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':main()
