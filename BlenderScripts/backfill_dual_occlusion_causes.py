"""Classify missing dual-mouse occlusion causes from evaluated source meshes.

Run with the exact source .blend loaded in background. Does not render or save
the blend. Existing coordinates and visibility labels are never overwritten.
Causes refer to body-mesh support rays, not visibility through rendered fur.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--frame-start', type=int, default=1)
    p.add_argument('--frame-end', type=int, default=400)
    p.add_argument('--support-tolerance', type=float, default=.001)
    p.add_argument('--skip-mismatched', action='store_true')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    for obj in bpy.data.objects:
        if obj.type == 'CURVES':
            obj.hide_viewport=True
            obj.hide_render=True
    source=Path(bpy.data.filepath)
    digest=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    counts, reports, mismatches = Counter(), [], []
    frames=[f for f in sorted(args.dataset.glob('frame_*'))
            if args.frame_start<=int(f.name.split('_')[-1])<=args.frame_end]
    if args.limit: frames=frames[:args.limit]
    for folder in frames:
        with np.load(folder/'keypoints.npz',allow_pickle=False) as saved:
            a=dict(saved)
        if len(a['mesh_names']) != 2:
            raise ValueError('Requires exactly two source meshes')
        bpy.context.scene.frame_set(int(a['frame']))
        bpy.context.view_layer.update()
        dg=bpy.context.evaluated_depsgraph_get()
        trees=[]; errors=[]; mismatch=False
        for m,name in enumerate(a['mesh_names']):
            # Match the original exporter, which re-evaluates frame_set for each
            # instance before reading its mesh and constrained armature.
            bpy.context.scene.frame_set(int(a['frame']))
            bpy.context.view_layer.update()
            dg=bpy.context.evaluated_depsgraph_get()
            obj=bpy.data.objects[str(name)].evaluated_get(dg)
            mesh=obj.to_mesh()
            try:
                world=[obj.matrix_world@v.co for v in mesh.vertices]
                ids=a['support_vertex_ids'][m]
                expected=a['support_world_xyz'][m]
                actual=np.asarray([world[int(i)] for i in ids.flatten()]).reshape(expected.shape)
                error=float(np.linalg.norm(actual-expected,axis=-1).max())
                if error>args.support_tolerance:
                    delta=(expected-actual).reshape(-1,3)
                    detail=dict(frame=int(a['frame']),mesh=str(name),max_error=error,
                                median_translation=np.median(delta,axis=0).tolist(),
                                per_keypoint_max={str(n):float(e) for n,e in zip(a['names'],np.linalg.norm(actual-expected,axis=-1).max(-1)) if e>args.support_tolerance},
                                translation_residual_max=float(np.linalg.norm(delta-np.median(delta,axis=0),axis=-1).max()))
                    if not args.skip_mismatched:
                        raise ValueError(f'Source/frame mismatch: {detail}')
                    mismatches.append(detail)
                    mismatch=True
                    break
                errors.append(error)
                trees.append(BVHTree.FromPolygons(world,[list(poly.vertices) for poly in mesh.polygons]))
            finally:
                obj.to_mesh_clear()
        if mismatch:
            if len(mismatches)%20==1: print('SKIP MISMATCH',mismatches[-1],flush=True)
            continue
        causes=np.full(a['in_frame'].shape,-1,dtype=np.int8)
        # 0 visible, 1 self-only blockers, 2 other-only blockers, 3 mixed; -1 unresolved.
        other_fraction=np.zeros_like(causes,dtype=np.float32)
        self_fraction=np.zeros_like(causes,dtype=np.float32)
        for v,e in enumerate(a['extrinsics']):
            camera=Vector((-e[:3,:3].T@e[:3,3]).tolist())
            for m in range(2):
                for k in range(len(a['names'])):
                    if not a['in_frame'][v,m,k]: continue
                    if a['surface_visible'][v,m,k]:
                        causes[v,m,k]=0
                        continue
                    blockers=[]
                    for target in a['support_world_xyz'][m,k]:
                        ray=Vector(target.tolist())-camera
                        distance=ray.length
                        if distance<=1e-6: continue
                        hits=[]
                        for j,tree in enumerate(trees):
                            hit=tree.ray_cast(camera,ray.normalized(),max(0.,distance-.05))
                            if hit[0] is not None: hits.append((hit[3],j))
                        blockers.append(min(hits)[1] if hits else -1)
                    n=max(1,len(blockers))
                    own=sum(x==m for x in blockers)
                    other=sum(x==1-m for x in blockers)
                    self_fraction[v,m,k]=own/n
                    other_fraction[v,m,k]=other/n
                    # A strict cause requires every support ray to be explained.
                    # Environment/visibility-tolerance disagreements remain unknown.
                    if own+other==n:
                        causes[v,m,k]=3 if own and other else 1 if own else 2
        path=folder/'keypoint_occlusion.npz'
        payload=dict(schema_version=np.asarray(1), names=a['names'], view_ids=a['view_ids'],
                     instance_ids=a['instance_ids'], occlusion_cause=causes,
                     occlusion_cause_valid=causes>=0, other_mouse_blocked_fraction=other_fraction,
                     self_blocked_fraction=self_fraction, source_sha256=np.asarray(digest.hexdigest()),
                     annotation_sha256=np.asarray(hashlib.sha256((folder/'keypoints.npz').read_bytes()).hexdigest()),
                     max_source_support_error=np.asarray(max(errors)))
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                if str(old['source_sha256']) != digest.hexdigest() or not np.array_equal(old['occlusion_cause'],causes):
                    raise ValueError(f'Refusing to overwrite different occlusion provenance: {path}')
                needs_write='annotation_sha256' not in old
        else:
            needs_write=True
        if needs_write:
            with path.with_suffix('.npz.tmp').open('wb') as f: np.savez_compressed(f,**payload)
            path.with_suffix('.npz.tmp').replace(path)
        counts.update(map(str,causes[a['in_frame']].flatten().tolist()))
        reports.append(dict(frame=int(a['frame']),max_support_error=max(errors)))
        if len(reports)%20==0 or len(reports)==1: print(f'OCCLUSION {len(reports)}/{len(frames)} {dict(counts)}',flush=True)
    report=dict(source_blend=str(source), source_sha256=digest.hexdigest(),frames=len(reports),
        requested_frames=len(frames),mismatches=mismatches,
        cause_codes={'-1':'unknown','0':'visible','1':'self','2':'other_mouse','3':'mixed'},
        counts=dict(counts),max_support_error=max((r['max_support_error'] for r in reports),default=None),
        hair_excluded=True,endpoint_tolerance_native=.05,strict_all_support_rays=True,
        records=reports)
    report_name=('occlusion_backfill_report.json' if args.frame_start==1 and args.frame_end==400 and not args.limit
                 else f'occlusion_probe_{args.frame_start}_{args.frame_end}.json')
    (args.dataset/report_name).write_text(json.dumps(report,indent=2)+'\n')
    print('OCCLUSION COMPLETE',dict(counts),flush=True)


if __name__=='__main__': main()
