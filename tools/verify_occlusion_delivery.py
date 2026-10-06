"""Verify additive label delivery and quarantine only this pilot's unhashed sidecars."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vggt-omega'))
from training.keypoint_contract import read_validity, read_occlusion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quarantine-unverified',action='store_true')
    args=parser.parse_args()
    root=ROOT/'dataset/BlenderPyMouse'
    out=ROOT/'output/occlusion_pose_v1'
    quarantined=[]; counts=Counter(); causes=Counter()
    for path in sorted(root.glob('*/frame_*/keypoints.npz')):
        with np.load(path,allow_pickle=False) as a:
            flags=read_validity(a,path)
            counts['frames']+=1
            counts['validity_sidecars']+=int(path.with_name('keypoint_validity.npz').exists())
            counts['coordinate_valid']+=int((flags['xy_label_valid'] & a['in_frame']).sum())
            counts['coordinate_changes_from_legacy']+=int(np.count_nonzero((flags['xy_label_valid'] & a['in_frame']) != a['in_frame']))
            p=path.with_name('keypoint_occlusion.npz')
            if p.exists():
                with np.load(p,allow_pickle=False) as c:
                    legacy='annotation_sha256' not in c
                    assert int(c['schema_version'])==1
                if legacy:
                    if not args.quarantine_unverified:
                        raise ValueError(f'Unverified sidecar; explicitly quarantine it before delivery: {p}')
                    # These are the initial pilot's artifacts, NOT original data.
                    # Keep a recoverable copy outside the training dataset.
                    target=out/'quarantine_unverified_causes'/path.parent.parent.name/path.parent.name/p.name
                    if not p.resolve().is_relative_to(root.resolve()) or not target.resolve().is_relative_to(out.resolve()):
                        raise ValueError('Unsafe quarantine path')
                    if target.exists(): raise FileExistsError(target)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    p.replace(target)
                    quarantined.append(dict(source=str(p.relative_to(ROOT)),destination=str(target.relative_to(ROOT))))
                else:
                    c=read_occlusion(a,path)
                    counts['verified_occlusion_frames']+=1
                    causes.update(map(str,c[a['in_frame']].flatten().tolist()))
    payload=dict(counts=dict(counts),occlusion_cause_counts=dict(causes),quarantined=quarantined,
                 original_rgb_depth_coordinates_and_blends_untouched=True)
    # Preserve the recovery ledger on verification reruns.
    report=out/'delivery_verification.json'
    if report.exists():
        old=json.loads(report.read_text())
        payload['quarantined']=old.get('quarantined',[])+quarantined
    report.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
