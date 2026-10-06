"""Read-only reopening and dependency audit of current three working pose files."""
import bpy,json,sys,datetime as dt
from pathlib import Path
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);base=root/'output/pymousestudio_realism_cycle_20261002';out=base/'v061_delivery_preflight';out.mkdir(exist_ok=False)
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));files={'climb':state['best_blend'],**state['pose_bank']};report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'read_only':True,'poses':{}}
def dep(kind,block):
    path=getattr(block,'filepath','');packed=bool(getattr(block,'packed_file',None)) or bool(getattr(block,'packed_files',[]));resolved=str(Path(bpy.path.abspath(path)).resolve()) if path and path!='<builtin>' else None
    return dict(kind=kind,name=block.name,users=block.users,path=path,resolved=resolved,packed=packed,exists=Path(resolved).is_file() if resolved else None)
for tag,file in files.items():
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    bpy.ops.wm.open_mainfile(filepath=file,load_ui=False);s=bpy.context.scene;bpy.context.view_layer.update();dependencies=[]
    for im in bpy.data.images:
        if im.source in ['FILE','MOVIE','SEQUENCE','TILED']:dependencies.append(dep('image',im))
    for kind,items in [('library',bpy.data.libraries),('font',bpy.data.fonts),('sound',bpy.data.sounds),('movieclip',bpy.data.movieclips),('cache_file',bpy.data.cache_files),('volume',bpy.data.volumes)]:
        for block in items:
            if getattr(block,'filepath','') not in ['', '<builtin>']:dependencies.append(dep(kind,block))
    mouse=bpy.data.objects['MS_Mouse'];ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();q=np.array([ev.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear()
    cameras={}
    for c in 'ABCDEFG':
        cam=bpy.data.objects['Replica_FR2_cam'+c];cameras[c]=dict(type=cam.data.type,lens=cam.data.lens,matrix=[list(r) for r in cam.matrix_world],pixel_aspect_y_over_x=cam.get('replica_pixel_aspect_y_over_x'),dof=cam.data.dof.use_dof)
    hair=[dict(name=o.name,viewport_hidden=o.hide_get(),render_hidden=o.hide_render,viewport_disabled=o.hide_viewport,modifiers=[dict(name=m.name,type=m.type,viewport=m.show_viewport,render=m.show_render) for m in o.modifiers]) for o in s.objects if o.type=='CURVES']
    item=dict(file=file,blender=bpy.app.version_string,frame=s.frame_current,frame_range=[s.frame_start,s.frame_end],scene_camera=s.camera.name,resolution=[s.render.resolution_x,s.render.resolution_y,s.render.resolution_percentage],pixel_aspect=[s.render.pixel_aspect_x,s.render.pixel_aspect_y],render_engine=s.render.engine,cycles_device=s.cycles.device,samples=s.cycles.samples,unit_scale=s.unit_settings.scale_length,view_transform=s.view_settings.view_transform,dependencies=dependencies,missing_unpacked=[d for d in dependencies if not d['packed'] and not d['exists']],cameras=cameras,hair=hair,skin_vertex_count=len(q),skin_min_z=float(q[:,2].min()),floor_vertices=int((q[:,2]<339.499).sum()),wall_vertices=int((np.linalg.norm(q[:,:2],axis=1)>56.471588).sum()),support_collection='Replica_CameraSupportDetails_v059' in bpy.data.collections,shape_keys=[k.name for k in mouse.data.shape_keys.key_blocks] if mouse.data.shape_keys else [],libraries=len(bpy.data.libraries))
    report['poses'][tag]=item;(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('PREFLIGHT',tag,'missing',len(item['missing_unpacked']),'hair',len(hair),'floor',item['floor_vertices'],'wall',item['wall_vertices'],flush=True)
report['complete']=len(report['poses'])==3;(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
