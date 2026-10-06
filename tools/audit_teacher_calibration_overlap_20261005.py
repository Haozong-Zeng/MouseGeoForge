"""Audit known-source overlap before considering any specialist teacher initialization."""
import sys,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
C=ROOT/'output/singleview_fusion_cycle_20261005';O=C/'teacher_calibration_overlap_v1'
def main():
    m=json.loads((ROOT/'output/bottom_anchor_cycle_20261002/superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text());entries=[e for e in m['entries'] if e['role']=='train'];rows=torch.load(C/'mixed_local_rgb_inputs_v1/evaluation_only.pt',weights_only=False,map_location='cpu');matches=[]
    for r in rows:
        z=r['metadata']
        if z['kind']!='real' or z['stage']!='calibration':continue
        frame=int(re.search(r'frame_(\d+)$',z['scene']).group(1));ee=[e for e in entries if e['session']==z['source'] and e['frame']==frame];matches.append(dict(index=r['index'],source=z['source'],frame=frame,view=z['view'],bottom=z['physical_bottom'],teacher_train_cameras=sorted({e['camera'] for e in ee}),source_frame_overlap=bool(ee)))
    result=dict(current_real_cal_images=len(matches),current_real_cal_source_frames=len({(e['source'],e['frame']) for e in matches}),images_with_source_frame_in_teacher_train=sum(e['source_frame_overlap'] for e in matches),teacher_fr3_train_entries=sum(e['session']=='FR3' for e in entries),examples=matches,conclusion='Previously fine-tuned specialist is NOT a held-out FR3 baseline and must not initialize this current source-held-out experiment. Original official Quadruped weights used in mixed experiments are unaffected. This verifies source/frame provenance, not equal annotation versions or pixel hashes; old labels predate current repairs. Do not relabel or retrain using CAL.')
    O.mkdir(exist_ok=True);(O/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
if __name__=='__main__':main()
