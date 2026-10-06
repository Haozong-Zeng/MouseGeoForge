"""Plan larger contexts using ONLY already eligible role-separated image IDs."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'eligible_fullview_context_plan_v1'

def main():
    assert not OUT.exists();OUT.mkdir();all_scenes={};summary={}
    for role in ['train','validation']:
        source=CYCLE/f'shared_view_{role}_context_v1/manifest.json';manifest=json.loads(source.read_text());groups=collections.defaultdict(list)
        for rec in manifest['records']:
            m=rec['metadata'];r=m['record'];assert r['role']==role and r.get('session') not in ['GR2','UR2','UR3'] and m['detection_complete']
            scene=r['scene'];assert all_scenes.get(scene,role)==role;all_scenes[scene]=role;groups[scene].append(rec)
        tasks=[];counts=collections.Counter();bykind=collections.Counter()
        for scene,records in sorted(groups.items()):
            records=sorted(records,key=lambda r:r['metadata']['view']);views=[r['metadata']['view'] for r in records];assert len(set(views))==len(views)
            base=records[0]['metadata']['record'];hashes={}
            for rec in records:
                for path,sha in rec['source_hashes'].items():
                    assert path not in hashes or hashes[path]==sha;hashes[path]=sha
            tasks.append(dict(record=dict(base,views=views),image_indices=[r['image_index'] for r in records],observations=[dict(image_index=r['image_index'],metadata=r['metadata']) for r in records],known_source_hashes=hashes))
            counts[len(views)]+=1;bykind[(base['kind'],len(views))]+=1
        assert sum(len(t['image_indices']) for t in tasks)==manifest['images']
        write_json(OUT/f'{role}.json',dict(role=role,tasks=tasks,source_manifest_sha256=fingerprint(source),original_source_sha256=manifest['source_sha256'],contract='Only already eligible supervised images become context views. No unseen/excluded/incomplete/unlabelled extra camera frames added. Larger context is same scene/time and same role; original pixel coordinates/labels unchanged. Context key inherited from firsteligible record is not a new annotation key. Actual full-context RGB re-encoding required, cannot stack pairedmemories. Sourcehash union is provenance; actual loadedfiles must be a verified subset.'))
        summary[role]=dict(images=manifest['images'],scenes=len(tasks),scene_view_count_histogram=dict(counts),by_kind_view_count={f'{k}/{v}':n for (k,v),n in bykind.items()},images_in_3plus_contexts=sum(len(t['image_indices']) for t in tasks if len(t['image_indices'])>=3))
    write_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
