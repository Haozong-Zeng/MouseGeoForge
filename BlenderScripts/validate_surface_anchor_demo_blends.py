"""Reopen derived review .blends and verify stored groups/marker coordinates."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:]
root=Path(args[0]);finalize='--finalize' in args
folder=root/'surface_geometry' if (root/'surface_geometry').exists() else root
records=[]
for path in sorted(folder.glob('*Review.blend')):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    if finalize:
        # Library exports have no saved active scene. Turn these derived-only
        # review files into ordinary .blends that open directly on the mouse.
        candidates=[s for s in bpy.data.scenes if 'anchor_definition_json' in s]
        assert candidates
        selected=next((s for s in candidates if s.name=='Rest'),candidates[0])
        bpy.context.window.scene=selected
        mesh=next(o for o in selected.objects if o.type=='MESH' and o.name.endswith('_Surface'))
        bpy.context.view_layer.objects.active=mesh;mesh.select_set(True)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':
                    area.spaces.active.region_3d.view_perspective='CAMERA'
                    area.spaces.active.shading.color_type='MATERIAL'
        bpy.context.preferences.filepaths.save_version=0
        bpy.ops.wm.save_as_mainfile(filepath=str(path),compress=True)
    assert 'anchor_definition_json' in bpy.context.scene, 'Review must open on a populated scene'
    views=[]
    for scene in bpy.data.scenes:
        if 'anchor_definition_json' not in scene:continue
        anchors=json.loads(scene['anchor_definition_json'])
        mesh=next(o for o in scene.objects if o.type=='MESH' and o.name.endswith('_Surface'))
        assert len(mesh.data.vertices)==9984
        assert len(mesh.vertex_groups)==6
        errors=[]
        adjacency=[set() for _ in mesh.data.vertices]
        for edge in mesh.data.edges:
            a,b=edge.vertices;adjacency[a].add(b);adjacency[b].add(a)
        for a in anchors:
            key=f"KP{a['display_id']:02d}_{a['name']}"
            vg=mesh.vertex_groups[key+'_anchor']
            weights={v.index:g.weight for v in mesh.data.vertices for g in v.groups if g.group==vg.index}
            assert abs(sum(weights.values())-1)<1e-6
            point=sum((mesh.data.vertices[i].co*w for i,w in weights.items()),Vector())
            marker=next(o for o in scene.objects if o.name.startswith(f"KP{a['display_id']:02d}_surface_anchor"))
            error=(mesh.matrix_world@point-marker.location).length;errors.append(error)
            assert error<1e-4
            ids=set(a['region_vertex_ids']);seen={next(iter(ids))}
            while True:
                grown=seen|{j for i in seen for j in adjacency[i] if j in ids}
                if grown==seen:break
                seen=grown
            assert seen==ids
        assert scene.render.engine=='CYCLES' and scene.cycles.device=='GPU'
        assert scene.render.resolution_x==1280 and scene.render.resolution_y==720
        views.append(dict(scene=scene.name,groups=6,region_connected=True,max_marker_error=max(errors),render_gpu=True))
    assert views
    records.append(dict(blend=path.name,scenes=views))
print(json.dumps(records,indent=2))
print('BLEND_REOPEN_VALIDATION_PASSED')
