"""Approximate camera support rails/electronics, outside arena and behind optical centers."""
import bpy,json,sys,time,datetime as dt
from pathlib import Path
from mathutils import Matrix,Vector
import numpy as np
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002'/globals().get('OUT_NAME','v059_camera_supports');out.mkdir(parents=True,exist_ok=False);s=bpy.context.scene;source=bpy.data.filepath;mouse=bpy.data.objects['MS_Mouse']
def skin():
    bpy.context.view_layer.update();ev=mouse.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();v=np.array([ev.matrix_world@v.co for v in me.vertices]);ev.to_mesh_clear();return v
before=skin();cams={c:[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in 'ABCDEFG'}
lights={o.name:dict(energy=o.data.energy,matrix=[list(r) for r in o.matrix_world]) for o in s.objects if o.type=='LIGHT'}
collection=bpy.data.collections.new('Replica_CameraSupportDetails_v059');s.collection.children.link(collection);created=[]
white=bpy.data.materials['Replica_WhiteCameraCase'];black=bpy.data.materials['Replica_CameraBlackPlastic'];steel=bpy.data.materials['Replica_CameraBrushedMetal']
def finish(o,name,mat,parent,loc):
    o.name=name
    for co in list(o.users_collection):co.objects.unlink(o)
    collection.objects.link(o);o.parent=parent;o.location=loc;o.data.materials.append(mat)
    if o.type=='MESH':
        o.cycles.is_caustics_receiver=True
        for p in o.data.polygons:p.use_smooth=True
    created.append(o.name);return o
def box(name,dims,loc,mat,parent,bevel=.6):
    bpy.ops.mesh.primitive_cube_add(size=1);o=bpy.context.object;o.data.transform(Matrix.Diagonal(Vector((*dims,1))));finish(o,name,mat,parent,loc)
    if bevel:mod=o.modifiers.new('Support edge bevel','BEVEL');mod.width=bevel;mod.segments=3
    return o
for c in 'BCDEFG':
    holder=bpy.data.objects['Replica_CameraProxy_'+c]
    box('Replica_SupportRail_'+c,(130,6,26),(7,-54,42),white,holder)
    box('Replica_ControllerBox_'+c,(37,13,28),(-31,-44.5,52),black,holder,1.)
    box('Replica_SupportBracket_'+c,(18,28,14),(7,-37,38),black,holder)
    # Two mounting screws on the visible front edge of the pale support rail.
    for i,x in enumerate([-45,55]):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=1.2,depth=.8)
        finish(bpy.context.object,f'Replica_RailScrew_{c}_{i}',black,holder,(x,-54,28.6))
    cu=bpy.data.curves.new('Replica_InstrumentWire_'+c,'CURVE');cu.dimensions='3D';cu.bevel_depth=.8;cu.bevel_resolution=3;sp=cu.splines.new('BEZIER');sp.bezier_points.add(4)
    for b,p in zip(sp.bezier_points,[(-20,-15,38),(-47,-22,46),(-57,-35,51),(-43,-43,55),(-31,-38,54)]):b.co=p;b.handle_left_type='AUTO';b.handle_right_type='AUTO'
    wire=bpy.data.objects.new('Replica_InstrumentWire_'+c,cu);collection.objects.link(wire);wire.parent=holder;cu.materials.append(black);created.append(wire.name)
    # Thin physical ribbon loop: no billboards or image-space overlays.
    centers=[(21,0,38),(40,-4,48),(55,-20,60),(47,-39,66),(19,-47,54)]
    verts=[(x+sign*4.5,y,z) for x,y,z in centers for sign in [-1,1]];faces=[(2*i,2*i+1,2*i+3,2*i+2) for i in range(len(centers)-1)]
    me=bpy.data.meshes.new('Replica_RibbonMesh_'+c);me.from_pydata(verts,[],faces);me.update();ribbon=bpy.data.objects.new('Replica_RibbonCable_'+c,me);collection.objects.link(ribbon);ribbon.parent=holder;me.materials.append(black);mod=ribbon.modifiers.new('Ribbon thickness','SOLIDIFY');mod.thickness=.3;mod=ribbon.modifiers.new('Ribbon smoothing','SUBSURF');mod.levels=2;mod.render_levels=2;created.append(ribbon.name)
bpy.context.view_layer.update();after=skin();error=float(np.linalg.norm(after-before,axis=1).max());assert error==0
assert all(cams[c]==[list(r) for r in bpy.data.objects['Replica_FR2_cam'+c].matrix_world] for c in cams)
assert all(lights[o.name]==dict(energy=o.data.energy,matrix=[list(r) for r in o.matrix_world]) for o in s.objects if o.type=='LIGHT')
bounds={}
for o in collection.objects:
    p=np.array([o.matrix_world@Vector(v) for v in o.bound_box]);bounds[o.name]=dict(min_radius_mm=float(np.linalg.norm(p[:,:2],axis=1).min()),zmin=float(p[:,2].min()),zmax=float(p[:,2].max()));assert bounds[o.name]['min_radius_mm']>100
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=128;s.cycles.use_denoising=True;s.render.resolution_x=1080;s.render.resolution_y=1200;s.render.resolution_percentage=100;s.render.use_persistent_data=True
s.camera=bpy.data.objects['Replica_FR2_camF'];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r)
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002'/globals().get('TARGET_NAME','PyMouseStudio-RealCapture-Replica-v059-camera-supports.blend');assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
m=dict(source=source,blend=str(target),frame=s.frame_current,skin_max_change_mm=error,cameras_unchanged=True,lights_unchanged=True,created=created,bounds=bounds,devices=devices,renders=[],asset_provenance='Procedural primitives authored for this task. No download, real image projection, or brand logo.',limitations=['Approximate support/electronics proxies, not exact CAD or measured dimensions.', 'Background occlusion and indirect lighting/reflections may change despite frozen light parameters.', 'Bounding box corner radii are coarse outside-arena sanity checks, not general triangle collision certification.'])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
write()
for c in 'FABD':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];r=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/r);s.render.pixel_aspect_y=max(1.,r);s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
