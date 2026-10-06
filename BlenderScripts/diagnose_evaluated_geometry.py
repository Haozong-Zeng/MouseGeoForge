"""Read-only source/annotation comparisons across repeated frame evaluation."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',default='1,5,5,4,5,28,5')
    p.add_argument('--exporter-context',action='store_true')
    p.add_argument('--reset-rest',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    for obj in bpy.data.objects:
        if obj.type=='CURVES': obj.hide_viewport=True
    renderer=None
    if args.exporter_context:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from augment_dual_mouse_blendedmvs_core23 import InstanceMaskRenderer
        renderer=InstanceMaskRenderer(bpy.context.scene,[bpy.data.objects[n] for n in ('LP_Black_Mesh','LP_Brown_Mesh')],args.output.parent/'mask_probe')
    meta=[]
    for name in ('LP_Black_Mesh','LP_Brown_Mesh','LP_Black_Armature','LP_Brown_Armature'):
        obj=bpy.data.objects[name]
        modifiers=[dict(name=m.name,type=m.type,viewport=m.show_viewport,render=m.show_render,
                        target=getattr(getattr(m,'object',None),'name',None)) for m in obj.modifiers]
        bones=[]
        if obj.type=='ARMATURE':
            for b in obj.pose.bones:
                if b.constraints:
                    bones.append(dict(bone=b.name,constraints=[dict(name=c.name,type=c.type,
                        influence=c.influence,mute=c.mute,target=getattr(getattr(c,'target',None),'name',None),
                        subtarget=getattr(c,'subtarget',None)) for c in b.constraints]))
        meta.append(dict(name=name,modifiers=modifiers,bone_constraints=bones,
                         shape_keys=[] if obj.type!='MESH' or not obj.data.shape_keys else
                         [(k.name,k.value) for k in obj.data.shape_keys.key_blocks]))
    rows=[]
    for f in map(int,args.frames.split(',')):
        with np.load(args.dataset/f'frame_{f:06d}'/'keypoints.npz') as a:
            a=dict(a)
        if args.reset_rest:
            for name in a['armature_names']: bpy.data.objects[str(name)].data.pose_position='REST'
            bpy.context.view_layer.update()
            for name in a['armature_names']: bpy.data.objects[str(name)].data.pose_position='POSE'
            bpy.context.view_layer.update()
        bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
        for m,name in enumerate(a['mesh_names']):
            bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
            dg=bpy.context.evaluated_depsgraph_get()
            ob=bpy.data.objects[str(name)]
            ev=ob.evaluated_get(dg); mesh=ev.to_mesh()
            try:
                ids=a['support_vertex_ids'][m]
                actual=np.asarray([ev.matrix_world@mesh.vertices[int(i)].co for i in ids.flatten()]).reshape(a['support_world_xyz'][m].shape)
                error=np.linalg.norm(actual-a['support_world_xyz'][m],axis=-1).max(-1)
            finally: ev.to_mesh_clear()
            arm=bpy.data.objects[str(a['armature_names'][m])]
            joints=np.asarray([arm.matrix_world@arm.pose.bones[str(b)].head for b in a['bones']])
            joint_error=np.linalg.norm(joints-a['joint_world_xyz'][m],axis=-1)
            r=dict(iteration=len(rows)//2,frame=f,mesh=str(name),support_max=float(error.max()),
                joint_max=float(joint_error.max()),
                support_per_keypoint={str(n):float(e) for n,e in zip(a['names'],error) if e>.001},
                joints_per_keypoint={str(n):float(e) for n,e in zip(a['names'],joint_error) if e>.001})
            rows.append(r); print(r,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(source=bpy.data.filepath,blender=bpy.app.version_string,
                                          objects=meta,comparisons=rows),indent=2)+'\n')
    if renderer: renderer.close()


if __name__=='__main__': main()
