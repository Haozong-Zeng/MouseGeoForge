"""Read-only background Blender audit of cloned scene; emits machine-readable state."""
import bpy,json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--')+1]);s=bpy.context.scene
def seq(v):return [float(x) for x in v]
objects=[]
for o in bpy.data.objects:
    r=dict(name=o.name,type=o.type,hide_render=o.hide_render,location=seq(o.location),dimensions=seq(o.dimensions),parent=o.parent.name if o.parent else None,materials=[m.name for m in o.data.materials if m] if hasattr(o.data,'materials') else [])
    if o.type=='CAMERA':r.update(lens=o.data.lens,camera_type=o.data.type,matrix_world=[seq(v) for v in o.matrix_world])
    if o.type=='ARMATURE':r['bones']=[p.name for p in o.pose.bones]
    objects.append(r)
images=[dict(name=i.name,path=i.filepath,packed=bool(i.packed_file),exists=Path(bpy.path.abspath(i.filepath)).is_file() if i.filepath else None) for i in bpy.data.images]
prefs=bpy.context.preferences.addons['cycles'].preferences; types=[x.identifier for x in prefs.bl_rna.properties['compute_device_type'].enum_items];devices=[];device_error=None
try:
    prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[dict(name=d.name,type=d.type,id=d.id) for d in prefs.devices]
except Exception as e:device_error=str(e)
data=dict(blender=bpy.app.version_string,file=bpy.data.filepath,frame_range=[s.frame_start,s.frame_end],frame=s.frame_current,unit_scale=s.unit_settings.scale_length,engine=s.render.engine,resolution=[s.render.resolution_x,s.render.resolution_y,s.render.resolution_percentage],camera=s.camera.name if s.camera else None,view_transform=s.view_settings.view_transform,exposure=s.view_settings.exposure,objects=objects,images=images,libraries=[dict(path=l.filepath,exists=Path(bpy.path.abspath(l.filepath)).is_file()) for l in bpy.data.libraries],gpu_devices=devices,device_error=device_error)
out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');print('REALISM_AUDIT_COMPLETE',str(out),flush=True)
