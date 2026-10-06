"""Read-only deterministic sample audit of evaluated hair length and contact."""
import bpy,json,sys,time
import numpy as np
from pathlib import Path
from mathutils.bvhtree import BVHTree
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/globals().get('AUDIT_OUTPUT','v021_groom_variation/contact_audit.json');assert not out.exists();project=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002'
rows=[]
for version in globals().get('AUDIT_VERSIONS',['v020-b_chiang060','v021-a_direction_variation','v021-b_length_variation']):
    path=project/f'PyMouseStudio-RealCapture-Replica-{version}.blend'
    if Path(bpy.data.filepath)!=path:bpy.ops.wm.open_mainfile(filepath=str(path))
    s=bpy.context.scene;s.frame_set(31);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();mouse=bpy.data.objects['MS_Mouse'].evaluated_get(dg);mesh=mouse.to_mesh();verts=[mouse.matrix_world@v.co for v in mesh.vertices];polys=[list(p.vertices) for p in mesh.polygons];bvh=BVHTree.FromPolygons(verts,polys,all_triangles=False);mouse.to_mesh_clear()
    for name in ['BaseFur.004','BaseFur.005','BaseFur.006']:
        ob=bpy.data.objects[name];ob.hide_viewport=False;ob.hide_set(False);bpy.context.view_layer.update();ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());data=ev.data
        lengths=[];signed=[];radii=[];zs=[];root_distances=[]
        for i in np.linspace(0,len(data.curves)-1,min(768,len(data.curves))).astype(int):
            points=[ev.matrix_world@p.position for p in data.curves[int(i)].points];lengths.append(sum((b-a).length for a,b in zip(points,points[1:])))
            near,normal,_,distance=bvh.find_nearest(points[0]);root_distances.append(distance)
            samples=points[1:]
            if globals().get('RESAMPLE_POINTS',0):
                segment_lengths=np.array([(b-a).length for a,b in zip(points,points[1:])]);cumulative=np.r_[0,np.cumsum(segment_lengths)];samples=[]
                for target in np.linspace(0,cumulative[-1],int(RESAMPLE_POINTS)+1)[1:]:
                    k=min(len(points)-2,max(0,int(np.searchsorted(cumulative,target,side='right')-1)));factor=float((target-cumulative[k])/max(segment_lengths[k],1e-10));samples.append(points[k].lerp(points[k+1],factor))
            for p in samples:
                near,normal,_,distance=bvh.find_nearest(p);signed.append(float((p-near).dot(normal)));radii.append(float(np.hypot(p.x,p.y)));zs.append(float(p.z))
        signed=np.asarray(signed);radii=np.asarray(radii);zs=np.asarray(zs)
        rows.append(dict(version=version,object=name,evaluated_curves=len(data.curves),sample_curves=len(lengths),length_quantiles_mm=np.quantile(lengths,[0,.1,.5,.9,1]).tolist(),root_distance_quantiles_mm=np.quantile(root_distances,[0,.5,.9,1]).tolist(),sample_points=len(signed),signed_skin_distance_quantiles_mm=np.quantile(signed,[0,.01,.1,.5,.9,1]).tolist(),points_inside_skin_over_02mm=int((signed<-.2).sum()),points_beyond_cylinder_over_02mm=int((radii>56.470588+.2).sum()),points_below_floor_over_025mm=int((zs<339.5-.25).sum())))
        ob.hide_viewport=True;bpy.context.view_layer.update();out.write_text(json.dumps(dict(method='Fixed evenly-spaced evaluated-curve sample; skin distance uses nearest surface normal, not exact solid containment. Baseline-relative diagnostic only, not exhaustive no-intersection certificate.',equal_arc_samples_per_curve=globals().get('RESAMPLE_POINTS',0),rows=rows),indent=2));print(version,name,rows[-1]['length_quantiles_mm'],flush=True)
