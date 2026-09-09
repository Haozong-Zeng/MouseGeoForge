"""Verify a published pilot through the real loader and the original-file hashes."""
import argparse
from complete_subsampled_blender_datasets import (
    BASE,DATA,OUT,WHITE,read,validate_scene,validate_loader,verify_original,
)
from core23_temporal_format import write_json_atomic

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default=WHITE)
args=parser.parse_args()
job=next(j for j in read(OUT/'plan.json')['jobs'] if j['dataset']==args.dataset)
scenes=[DATA/job['dataset']/f'frame_{fid:06d}' for fid in job['missing_frames']]
scenes=[s for s in scenes if s.exists()]
assert scenes,'No newly published scenes to validate'
for scene in scenes:
    validate_scene(scene,job)
validate_loader(scenes)
count=verify_original(job['dataset'])
report={'passed':True,'dataset':job['dataset'],'new_frames':[s.name for s in scenes],
        'original_file_hashes_verified':count,'multi_instance_loader_passed':True}
write_json_atomic(OUT/'pilot_integration_validation.json',report)
print(report,flush=True)
