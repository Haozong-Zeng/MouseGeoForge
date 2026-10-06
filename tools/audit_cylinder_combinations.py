"""Audit physical-view coverage beyond marginal top/bottom counts."""
import json
from collections import Counter,defaultdict
from train_cylinder_pose_only import OUTPUT
from run_depth_pose_ablation import write_json


def audit(spec):
    pairs=Counter(); mono=defaultdict(set); seen_pairs=set()
    for slot in spec['arms']['pose_head_only']:
        r=spec['records'][slot[0]]
        if r['source_group']!='mouse_climb_wall_91':continue
        if len(r['views'])==1:
            mono[r['scene']].add(r['views'][0])
        if len(r['views'])==2:
            pairs[','.join(map(str,r['views']))]+=1
            seen_pairs.add(tuple(sorted(r['views'])))
    missing=[]
    for item in spec['evaluation']:
        r=spec['records'][item['key']]
        if item['group']=='side_bottom' and tuple(sorted(r['views'])) not in seen_pairs:
            missing.append(dict(scene=r['scene'],views=r['views']))
    return dict(cylinder_pair_counts=dict(pairs),
        mono_cameras_per_pose={k:sorted(v) for k,v in mono.items()},
        poses_with_only_one_mono_camera=sum(len(v)==1 for v in mono.values()),
        missing_exact_development_side_bottom_pairs=missing,
        limitation='ClimbWall/WashFace share a verified rig. This checks matched-rig IDs, not IDs across unrelated datasets.',
        causal_note='An unseen combination is a coverage confound, not proof it caused the accuracy regression.',
        recommendation='Decouple camera combination/reference order from pose index; cover each side with both bottom and top across visits. Keep old locked runs intact.')


def main():
    result=audit(json.loads((OUTPUT/'protocol.json').read_text()))
    write_json(OUTPUT/'combination_coverage.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
