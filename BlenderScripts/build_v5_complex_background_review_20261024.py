"""Finite, review-only scene authoring. Never save over a source or launch a dataset job.

Run with Blender's --background --python, optionally -- --render-only to render
the already saved review file. The environment is a single world-space assembly;
the three v3 TRAIN cameras use the same objects, materials and lighting.
"""
import bpy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from mathutils import Vector, Quaternion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/v5_complex_background_review_20261024'
SEED = 20261024
PREFIX = 'V5_REVIEW_'


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def material(name, color, roughness=.65, metallic=0, texture_scale=None):
    m = bpy.data.materials.new(PREFIX + name)
    m.use_nodes = True
    n = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    n.inputs['Base Color'].default_value = (*color, 1)
    n.inputs['Roughness'].default_value = roughness
    n.inputs['Metallic'].default_value = metallic
    if texture_scale:
        noise = m.node_tree.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = texture_scale
        noise.inputs['Detail'].default_value = 3
        ramp = m.node_tree.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .22
        ramp.color_ramp.elements[1].position = .8
        ramp.color_ramp.elements[0].color = (*(x * .88 for x in color), 1)
        ramp.color_ramp.elements[1].color = (*(min(1, x * 1.08) for x in color), 1)
        m.node_tree.links.new(noise.outputs['Fac'], ramp.inputs[0])
        m.node_tree.links.new(ramp.outputs['Color'], n.inputs['Base Color'])
    return m


def author():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'MouseGeoForge_V5_complex_background_REVIEW.blend'
    if destination.exists():
        raise FileExistsError('Review project already exists; use --render-only or a new output name.')
    cfg = json.loads((ROOT / 'output/bottom_geometry_v3_local12h_20261018/render_config_shard0.json').read_text())
    source = cfg['sources'][0]
    source_path = ROOT / source['path']
    source_hash = hashlib.file_digest(source_path.open('rb'), 'sha256').hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source_path))
    scene = bpy.context.scene
    scene.frame_set(49)
    hidden = []
    for coll in bpy.data.collections:
        if coll.name.startswith('CameraItems'):
            coll.hide_render = True
            coll.hide_viewport = True
    for o in list(scene.objects):
        if o.type == 'LIGHT' or o.name in ('Building_Floor', 'Part_WhiteHexagonBackground', 'Part_TopCover') or o.name.startswith(('Part_Camera', 'Part_CamLen', 'CameraItems', 'Part_PiCam', 'Part_CNC-Camera', 'Part_BottomCamScrew', 'Part_SideCamScrew')):
            o.hide_render = True
            o.hide_set(True)
            hidden.append(o.name)
            for child in o.children_recursive:
                child.hide_render = True
                child.hide_set(True)
                hidden.append(child.name)
        # Keep the source's render policy for animated hair.
        if o.type == 'CURVES':
            o.hide_viewport = True
    env = bpy.data.collections.new('V5 REVIEW | shared asymmetric laboratory')
    scene.collection.children.link(env)
    geometry = []

    def register(o, name, mat=None):
        o.name = PREFIX + name
        for c in list(o.users_collection):
            c.objects.unlink(o)
        env.objects.link(o)
        if mat:
            o.data.materials.append(mat)
        o['review_environment_seed'] = SEED
        geometry.append(o)
        return o

    def box(name, pos, size, mat, bevel=0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        o = register(bpy.context.object, name, mat)
        o.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if bevel:
            mod = o.modifiers.new('soft manufactured edges', 'BEVEL')
            mod.width = bevel
            mod.segments = 2
        return o

    def rod(name, a, b, radius, mat):
        d = Vector(b) - Vector(a)
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=d.length, location=(Vector(a) + Vector(b)) / 2)
        o = register(bpy.context.object, name, mat)
        o.rotation_mode = 'QUATERNION'
        o.rotation_quaternion = d.to_track_quat('Z', 'Y')
        return o

    def cable(name, points, radius, mat):
        d = bpy.data.curves.new(PREFIX + name, 'CURVE')
        d.dimensions = '3D'
        d.resolution_u = 12
        d.bevel_depth = radius
        d.bevel_resolution = 3
        p = d.splines.new('BEZIER')
        p.bezier_points.add(len(points) - 1)
        for v, co in zip(p.bezier_points, points):
            v.co = co
            v.handle_left_type = 'AUTO'
            v.handle_right_type = 'AUTO'
        o = bpy.data.objects.new(PREFIX + name, d)
        scene.collection.objects.link(o)
        register(o, name, mat)
        # Opaque environment cables must also participate in the existing
        # mesh-only primary-ray truth policy (which excludes animated hair).
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.convert(target='MESH')
        return bpy.context.object

    warm = material('warm plaster', (.40, .36, .29), texture_scale=17)
    cool = material('cool plaster', (.29, .36, .37), texture_scale=23)
    ceiling = material('ceiling paint', (.53, .53, .47), texture_scale=12)
    floor = material('epoxy floor', (.15, .17, .16), .73, texture_scale=25)
    metal = material('brushed aluminum', (.25, .28, .29), .4, .7, 110)
    dark = material('charcoal polymer', (.035, .043, .046), .72)
    panel = material('muted olive panels', (.16, .20, .145), .7, texture_scale=8)
    cardboard = material('fiber storage', (.30, .20, .105), .85, texture_scale=35)
    cable_mat = material('rubber cable', (.017, .021, .019), .8)
    paper = material('plain paper', (.64, .61, .50), .9)
    # All coordinates are in the original source's millimetre-like Blender units.
    # Preserve the animated glass floor at z=337.5..339.5 and cylinder contact geometry.
    box('room floor', (0, 0, -9), (1400, 1400, 12), floor)
    box('room back wall', (0, 690, 470), (1400, 18, 960), cool)
    box('room front wall', (0, -690, 470), (1400, 18, 960), warm)
    box('room left wall', (-690, 0, 470), (18, 1400, 960), warm)
    box('room right wall', (690, 0, 470), (18, 1400, 960), cool)
    box('ceiling', (0, 0, 950), (1400, 1400, 18), ceiling)

    # Structural ceiling details are intentionally non-centred. These are shared
    # physical features in bottom/side rays, never camera-facing image backdrops.
    box('ceiling service tray', (-110, 35, 898), (86, 1020, 23), metal, 3)
    box('ceiling junction', (102, 102, 918), (94, 72, 35), dark, 4)
    rod('ceiling conduit A', (-270, -640, 922), (-270, 490, 922), 8, metal)
    rod('ceiling conduit elbow', (-270, 490, 922), (410, 490, 922), 8, metal)
    cable('ceiling service cable', [(125, -620, 922), (116, -170, 900), (185, 150, 891), (360, 570, 920)], 4, cable_mat)
    for j, y in enumerate((-410, -195, 54, 310)):
        box('ceiling tray brace %02d' % j, (-110, y, 876), (130, 12, 9), dark, 1)
    box('ceiling access panel', (250, -170, 936), (190, 135, 7), panel, 3)
    # Irregular shallow wall panels, storage and service rails supply texture at
    # more than one depth without adding scenery to the animal's motion volume.
    rng = random.Random(SEED)
    for i, x in enumerate((-475, -230, 115, 440)):
        z = rng.uniform(390, 640)
        box('back wall panel %02d' % i, (x, 675, z), (rng.uniform(110, 180), 16, rng.uniform(135, 230)), panel if i % 2 else paper, 2)
    for i, y in enumerate((-420, -125, 270, 510)):
        box('left wall seam %02d' % i, (-675, y, 470), (10, 9, 860), metal, 1)
    for z in (155, 290):
        box('storage shelf %d' % z, (430, 548, z), (320, 190, 12), metal, 3)
    for i, (x, y, z, sx, sy, sz) in enumerate(((355, 545, 205, 110, 122, 88), (487, 530, 204, 106, 132, 86), (415, 550, 348, 166, 120, 105))):
        box('storage bin %02d' % i, (x, y, z), (sx, sy, sz), cardboard if i != 1 else panel, 4)
        box('bin lid %02d' % i, (x, y, z + sz / 2), (sx + 4, sy + 4, 5), dark, 1)
    # Six fixture groups, above and below the glass; all outside the motion cylinder.
    for i, (x, y, high) in enumerate(((-180, 160, 580), (205, 85, 455), (-205, -165, 265))):
        box('post base %02d' % i, (x, y, 36), (66, 54, 16), dark, 3)
        rod('instrument post %02d' % i, (x, y, 44), (x, y, high), 8, metal)
        box('post clamp %02d' % i, (x, y, high - 55), (30, 27, 24), dark, 3)
        rod('post cross arm %02d' % i, (x, y, high - 55), (x + 42, y + 15, high - 55), 5, metal)
    box('above floor housing', (-181, 173, 566), (64, 54, 52), dark, 5)
    box('below floor controller', (194, -114, 205), (82, 52, 44), panel, 4)
    for k in range(6):
        box('controller vent %02d' % k, (172 + k * 7, -141, 206), (3, 2, 20), dark)
    cable('below floor harness', [(194, -115, 183), (214, -173, 120), (101, -226, 63), (-86, -237, 40)], 3.2, cable_mat)
    cable('rear hanging service', [(-180, 174, 565), (-217, 230, 510), (-260, 257, 330), (-288, 240, 125)], 3.5, cable_mat)
    # Non-centred rim clamp and small scratches outside the central contact area.
    box('offset rim clamp', (-107, 58, 346), (22, 30, 13), dark, 2)
    rod('rim clamp screw', (-107, 58, 351), (-107, 58, 363), 3.5, metal)
    for i in range(7):
        a = rng.uniform(1.7, 3.6)
        r = rng.uniform(96, 115)
        x, y = r * math.cos(a) - 1, r * math.sin(a) - 1
        cable('peripheral glass scuff %02d' % i, [(x, y, 339.57), (x + rng.uniform(4, 10), y + rng.uniform(-3, 3), 339.57)], .075, metal)

    world = bpy.data.worlds.new(PREFIX + 'shared world')
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Color'].default_value = (.63, .69, .78, 1)
    bg.inputs['Strength'].default_value = .18
    scene.world = world
    lighting = []
    for name, pos, target, power, size, color in (
        ('warm key', (-260, -170, 775), (0, 0, 340), 36000000, 12, (1, .87, .72)),
        ('cool soft fill', (320, 110, 660), (0, 0, 330), 24000000, 16, (.79, .89, 1)),
        ('rear bounce', (-60, 410, 510), (0, 0, 350), 16000000, 14, (1, .95, .84)),
    ):
        d = bpy.data.lights.new(PREFIX + name, 'SPOT')
        d.energy = power
        d.shadow_soft_size = size
        d.spot_size = math.radians(125)
        d.spot_blend = .75
        d.color = color
        o = bpy.data.objects.new(d.name, d)
        env.objects.link(o)
        o.location = pos
        o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()
        lighting.append(dict(name=o.name, type='SPOT', location=list(pos), target=list(target), energy=power, soft_radius=size, cone_degrees=125, blend=.75, color=list(color)))

    rig = next(r for r in cfg['rigs'] if r['id'] == 'train_rig00')
    cams = []
    anchor = Vector(source['anchor'])
    for i, relative in enumerate(rig['relative_positions']):
        d = bpy.data.cameras.new(PREFIX + 'camera_%d' % i)
        d.lens = rig['lens_mm']
        d.sensor_width = 36
        d.sensor_fit = 'HORIZONTAL'
        d.clip_start = .1
        d.clip_end = 4000
        c = bpy.data.objects.new(d.name, d)
        scene.collection.objects.link(c)
        c.location = anchor + Vector(relative)
        c.rotation_mode = 'QUATERNION'
        c.rotation_quaternion = (anchor - c.location).to_track_quat('-Z', 'Y') @ Quaternion((0, 0, 1), math.radians(rig['roll_deg'][i]))
        c['role'] = 'bottom' if i == 0 else 'side'
        cams.append(c)
    d = bpy.data.cameras.new(PREFIX + 'overview')
    d.lens = 29
    d.clip_end = 4000
    overview = bpy.data.objects.new(d.name, d)
    scene.collection.objects.link(overview)
    overview.location = (490, -540, 720)
    overview.rotation_euler = (Vector((0, 30, 405)) - overview.location).to_track_quat('-Z', 'Y').to_euler()
    overview['review_only_not_dataset_camera'] = True

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 8
    pref = bpy.context.preferences.addons['cycles'].preferences
    try:
        pref.compute_device_type = 'OPTIX'
        pref.get_devices()
        for dev in pref.devices:
            dev.use = dev.type == 'OPTIX'
        scene.cycles.device = 'GPU' if any(dev.use for dev in pref.devices) else 'CPU'
    except (TypeError, RuntimeError):
        scene.cycles.device = 'CPU'
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = True
    scene.view_settings.exposure = 0
    scene.camera = overview
    scene.render.filepath = str(OUT / 'overview.png')
    scene['review_status'] = 'AWAITING USER SCENE REVIEW; NO DATASET RELEASE OR BATCH RENDER'
    scene['environment_seed'] = SEED
    scene['source_blend_sha256'] = source_hash
    scene['review_note'] = 'Shared world; source mouse animation/contact glass/cylinder unchanged. Review cameras 0/1/2 match v3 TRAIN rig00. No view-dependent backgrounds.'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    text = bpy.data.texts.new('READ ME - V5 SCENE REVIEW')
    text.write('REVIEW ONLY. No formal dataset rendered.\nFrame 49, ClimbWall, v3 TRAIN rig00.\nF12: overview. Choose V5_REVIEW_camera_0 / 1 / 2 for bottom / side / side.\nSource mouse, hair, animation, contact glass and cylinder preserved.\nOne shared procedural lab environment and 3 fixed soft spot lights.\nOriginal white room shell / top cover and original lighting hidden.\nOriginal camera equipment hidden as in v3.\nNo camera labels, camera-dependent shading, or new external assets.\nThis is one illustrative seed, not a finalized distribution or visibility receipt.\n')
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=True)
    layout = [dict(name=o.name, type=o.type, matrix_world=[list(row) for row in o.matrix_world], dimensions=list(o.dimensions), materials=[m.name for m in getattr(o.data, 'materials', [])]) for o in geometry]
    save_json('scene_review_manifest.json', dict(status='review_only_not_approved', seed=SEED, source=source['path'], source_sha256=source_hash, frame=49, rig=rig, source_objects_hidden=sorted(set(hidden)), environment=layout, lighting=lighting, world_strength=.18, cameras=[c.name for c in cams], overview=overview.name, blender=bpy.app.version_string, blend=str(destination), script_sha256=hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest()))
    assert hashlib.file_digest(source_path.open('rb'), 'sha256').hexdigest() == source_hash
    print('REVIEW_BLEND_SAVED', str(destination), flush=True)


def refine_existing():
    """Apply the reviewed lighting adjustment without rebuilding original assets."""
    destination = OUT / 'MouseGeoForge_V5_complex_background_REVIEW.blend'
    if destination.exists():
        raise FileExistsError('Archive the prior review file before writing the refinement.')
    scene = bpy.context.scene
    powers = {'warm key': 36000000, 'cool soft fill': 24000000, 'rear bounce': 16000000}
    for name, power in powers.items():
        bpy.data.objects[PREFIX + name].data.energy = power
    for o in list(scene.objects):
        if o.name.startswith(PREFIX) and o.type == 'CURVE':
            bpy.ops.object.select_all(action='DESELECT')
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.convert(target='MESH')
    bpy.ops.object.select_all(action='DESELECT')
    scene.camera = bpy.data.objects[PREFIX + 'overview']
    scene.render.filepath = str(OUT / 'overview.png')
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    previous = json.loads((OUT / 'attempt02_small_spots/scene_review_manifest.json').read_text(encoding='utf-8'))
    for light in previous['lighting']:
        light['energy'] = bpy.data.objects[light['name']].data.energy
    for item in previous['environment']:
        item['type'] = bpy.data.objects[item['name']].type
    previous['blend'] = str(destination)
    previous['script_sha256'] = hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest()
    previous['refinement'] = 'Small physical spot sources; higher diffuse illumination; opaque cables converted to mesh for truth compatibility. Earlier exposure trials archived.'
    save_json('scene_review_manifest.json', previous)
    print('REFINED_REVIEW_SAVED', flush=True)


def render():
    scene = bpy.context.scene
    results = []
    for name, filename in [('overview', 'overview.png'), ('camera_0', 'bottom.png'), ('camera_1', 'side_1.png'), ('camera_2', 'side_2.png')]:
        scene.camera = bpy.data.objects[PREFIX + name]
        scene.render.filepath = str(OUT / filename)
        print('REVIEW_RENDER', filename, flush=True)
        bpy.ops.render.render(write_still=True)
        results.append(dict(camera=scene.camera.name, file=filename))
        save_json('preview_progress.json', dict(completed=results))
    print('REVIEW_PREVIEWS_COMPLETE', flush=True)


if __name__ == '__main__':
    if '--refine-existing' in sys.argv:
        refine_existing()
    elif '--render-only' not in sys.argv:
        author()
    render()
