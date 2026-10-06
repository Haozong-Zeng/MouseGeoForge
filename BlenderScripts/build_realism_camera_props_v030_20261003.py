"""Procedural camera-housing proxies registered to frozen calibrated camera poses."""
import bpy,json,sys,time,datetime as dt
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
root=Path(sys.argv[sys.argv.index('--')+1]);out=root/'output/pymousestudio_realism_cycle_20261002'/globals().get('OUT_TAG','v030_camera_props');out.mkdir(exist_ok=False);s=bpy.context.scene
camera_before={c:bpy.data.objects['Replica_FR2_cam'+c].matrix_world.copy() for c in 'ABCDEFG'};source=bpy.data.filepath;hidden=[]
prefixes=('Part_PiCamSide.','Part_SideCamScrew.','Part_4inchOpticPost.','Part_8inchOpticPost.','Part_90degreeClamp.','RotatableOpticClampPart1.','RotatableOpticClampPart2.')
for o in s.objects:
    if o.name.startswith(prefixes):hidden.append(o.name);o.hide_render=True;o.hide_set(True)
collection=bpy.data.collections.new('Replica_CalibratedCamera_VisualProxies');s.collection.children.link(collection);created=[]
def material(name,color,rough,metal=0.):
    mat=bpy.data.materials.new(name);mat.use_nodes=True;mat.node_tree.nodes.clear();bs=mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled');output=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(bs.outputs['BSDF'],output.inputs['Surface']);bs.inputs['Base Color'].default_value=(color,color,color,1);bs.inputs['Roughness'].default_value=rough;bs.inputs['Metallic'].default_value=metal;return mat
white=material('Replica_WhiteCameraCase',.62,.58);black=material('Replica_CameraBlackPlastic',.009,.44);steel=material('Replica_CameraBrushedMetal',.45,.32,.8);lens=material('Replica_DarkLensFront',.006,.10,.3);collar=material('Replica_PalePostCollar',.45,.60)
def finish(o,name,mat,parent=None,location=(0,0,0)):
    o.name=name
    for c in list(o.users_collection):c.objects.unlink(o)
    collection.objects.link(o);o.data.materials.append(mat)
    if parent:o.parent=parent
    o.location=location
    for p in o.data.polygons:p.use_smooth=True
    created.append(o.name);return o
def box(name,dimensions,loc,mat,parent=None,bevel=.5):
    bpy.ops.mesh.primitive_cube_add(size=1);o=bpy.context.object;o.data.transform(Matrix.Diagonal(Vector((*dimensions,1))));finish(o,name,mat,parent,loc)
    if bevel:mod=o.modifiers.new('Physical edge bevel','BEVEL');mod.width=bevel;mod.segments=3
    return o
def cylinder(name,radius,depth,loc,mat,parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=radius,depth=depth);return finish(bpy.context.object,name,mat,parent,loc)
for c in 'BCDEFG':
    holder=bpy.data.objects.new('Replica_CameraProxy_'+c,None);collection.objects.link(holder);holder.matrix_world=camera_before[c];holder['note']='Visual approximation only. All mesh starts behind optical center; render camera calibration unchanged.'
    box('Replica_Case_'+c,(50,45,24),(-7,0,22),white,holder,1.)
    box('Replica_Back_'+c,(47,42,5),(-7,0,36.5),black,holder,.6)
    cylinder('Replica_LensBarrel_'+c,11,9,(0,0,5.5),black,holder)
    bpy.ops.mesh.primitive_torus_add(major_radius=9.5,minor_radius=1.4,major_segments=64,minor_segments=16);finish(bpy.context.object,'Replica_LensRing_'+c,steel,holder,(0,0,1.7))
    cylinder('Replica_LensGlass_'+c,7.9,.6,(0,0,.9),lens,holder)
    cylinder('Replica_AuxPort_'+c,3.5,.7,(-24,-5,9.4),black,holder)
    for i,(x,y) in enumerate([(-29,-18),(-29,18),(15,-18),(15,18)]):cylinder(f'Replica_CaseScrew_{c}_{i}',.85,.7,(x,y,9.4),black,holder)
    box('Replica_Mount_'+c,(25,6,34),(-7,-23,24),black,holder,.7)
    position=holder.matrix_world@Vector((-7,0,45));z0=327.5;z1=705
    cylinder('Replica_Post_'+c,2.4,z1-z0,(position.x,position.y,(z0+z1)/2),steel)
    box('Replica_PostCollar_'+c,(10,9,30),(position.x,position.y,holder.matrix_world.translation.z+70),collar,None,.7)
    # A compact cable loop behind the case; no external asset or image texture.
    cu=bpy.data.curves.new('Replica_CableCurve_'+c,'CURVE');cu.dimensions='3D';cu.bevel_depth=1.5;cu.bevel_resolution=3;sp=cu.splines.new('BEZIER');sp.bezier_points.add(4)
    for b,p in zip(sp.bezier_points,[(-24,-16,37),(-39,-30,52),(-24,-45,68),(12,-42,61),(14,-18,38)]):b.co=p;b.handle_left_type='AUTO';b.handle_right_type='AUTO'
    cable=bpy.data.objects.new('Replica_Cable_'+c,cu);collection.objects.link(cable);cable.parent=holder;cu.materials.append(black);created.append(cable.name)
bpy.context.view_layer.update();assert max(float(np.abs(np.array(bpy.data.objects['Replica_FR2_cam'+c].matrix_world)-np.array(camera_before[c])).max()) for c in 'ABCDEFG')==0
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices();devices=[]
for d in prefs.devices:
    d.use=d.type=='OPTIX' and 'RTX PRO 6000' in d.name
    if d.use:devices.append(d.name)
assert devices;s.cycles.device='GPU';s.cycles.samples=64;s.cycles.use_denoising=True;s.render.resolution_percentage=100;s.render.use_persistent_data=True
target=root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Replica-20261002/PyMouseStudio-RealCapture-Replica-v030-camera-props.blend';assert not target.exists();bpy.ops.wm.save_as_mainfile(filepath=str(target))
m=dict(source=source,blend=str(target),frame=s.frame_current,hidden_original_equipment=hidden,created=created,calibrated_camera_matrices_unchanged=True,mouse_pose_materials_unchanged=True,proxy_dimensions_mm=dict(case=[50,45,24],lens_outer_diameter=22,post_diameter=4.8),asset_provenance='Original procedural primitives authored for this task. No downloaded asset, no texture upload, no brand logos. Dimension estimates from development reference only, not measured CAD.',devices=devices,renders=[],limitations=['Visual proxy, not a measured hardware reconstruction.','Occlusion/reflections can change when physical equipment moves. Existing scene lights and LED meshes unchanged.'])
def write():(out/'manifest.json').write_text(json.dumps(m,indent=2))
write()
for c in 'ABDF':
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,3,6,59,16,tzinfo=dt.timezone.utc)
    s.camera=bpy.data.objects['Replica_FR2_cam'+c];ratio=float(s.camera['replica_pixel_aspect_y_over_x']);s.render.pixel_aspect_x=max(1.,1/ratio);s.render.pixel_aspect_y=max(1.,ratio);s.render.resolution_percentage=50;s.render.filepath=str(out/f'cam{c}.png');t=time.monotonic();bpy.ops.render.render(write_still=True);m['renders'].append(dict(camera=c,file=s.render.filepath,seconds=time.monotonic()-t));write()
