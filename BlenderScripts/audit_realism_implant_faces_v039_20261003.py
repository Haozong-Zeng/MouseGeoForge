"""Inspect per-face material indices, then test hiding bar to identify white endpoint."""
import bpy,json,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002/v039_implant_faces';out.mkdir(exist_ok=False);s=bpy.context.scene;ob=bpy.data.objects['ReplicaAsset_立方体']
m=dict(source=bpy.data.filepath,objects={o.name:dict(slots=[dict(link=x.link,material=x.material.name if x.material else None) for x in o.material_slots],faces=[dict(index=p.index,material_index=p.material_index) for p in o.data.polygons],modifiers=[dict(name=x.name,type=x.type) for x in o.modifiers]) for o in bpy.data.objects if o.name.startswith('ReplicaAsset_')},renders=[])
(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=64;s.cycles.use_denoising=True;s.render.resolution_percentage=50;s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
ob.hide_render=True;s.render.filepath=str(out/'hidden_bar_camF.png');bpy.ops.render.render(write_still=True);m['renders'].append(s.render.filepath);(out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
