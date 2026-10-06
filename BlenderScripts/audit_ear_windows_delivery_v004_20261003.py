import bpy,contextlib,hashlib,io,json,math,runpy,sys
from pathlib import Path
root=Path(sys.argv[-1]);out=root/'output/pymousestudio_ear_cycle_20261003/windows_audit_v004';out.mkdir(exist_ok=False)
delivery=Path('C:/Users/55395/Downloads/PyMouseStudio-EarStudy-20261003/DELIVERY');helper=delivery/'render_views.py';records=[]
for title,frame in [('Climb',31),('Quadruped',111),('HeadRaise',216)]:
    p=delivery/f'PyMouseStudio-EarStudy-{title}-Static.blend';before=hashlib.sha256(p.read_bytes()).hexdigest();bpy.ops.wm.open_mainfile(filepath=str(p));s=bpy.context.scene
    assert s.frame_start==s.frame_end==s.frame_current==frame
    assert s.camera.name=='Replica_FR2_camF'
    mouse=bpy.data.objects['MS_Mouse'];ear=mouse.data.materials[4];bs=next(n for n in ear.node_tree.nodes if n.type=='BSDF_PRINCIPLED');gray=list(bs.inputs['Base Color'].default_value);assert all(math.isclose(x,.6,abs_tol=1e-6) for x in gray[:3]);assert bs.inputs['Emission Strength'].default_value==0
    images=[i for i in bpy.data.images if i.source=='FILE'];packed=sum(bool(i.packed_file) or bool(i.packed_files) for i in images);missing=[i.name for i in images if not(i.packed_file or i.packed_files) and not Path(bpy.path.abspath(i.filepath)).is_file()]
    assert packed==len(images) and not missing and not bpy.data.libraries
    hairs=[dict(name=o.name,viewport_disabled=o.hide_viewport,render_disabled=o.hide_render) for o in bpy.data.objects if o.type=='CURVES'];assert len(hairs)==9 and all(x['viewport_disabled'] and not x['render_disabled'] for x in hairs)
    capture=io.StringIO();sys.argv=[str(helper),'--','--views','ACEF','--dry-run']
    with contextlib.redirect_stdout(capture):runpy.run_path(str(helper),run_name='__main__')
    info=json.loads(next(x.split('=',1)[1] for x in capture.getvalue().splitlines() if x.startswith('DELIVERY_DRY_RUN=')))
    assert info['devices'] and info['views']==list('ACEF');assert hashlib.sha256(p.read_bytes()).hexdigest()==before
    records.append(dict(pose=title,frame=frame,blend_sha256_unchanged=before,ear_basecolor=gray,ear_material=ear.name,image_file_dependencies=len(images),packed_images=packed,missing_images=missing,external_libraries=0,hair_settings=hairs,render_helper=info))
(out/'audit.json').write_text(json.dumps(dict(complete=True,read_only=True,records=records),ensure_ascii=False,indent=2));print('EAR_WINDOWS_AUDIT_COMPLETE')
