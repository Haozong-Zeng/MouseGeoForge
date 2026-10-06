"""Small TRAIN/calibration RGB export for high-resolution branch feasibility."""
import sys,json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image
from train_pixel_adapter_matched_20261005 import prepare_rows
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'highres_preflight_inputs_v1'

def main():
    OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();torch.set_num_threads(2);rows=prepare_rows();quality=torch.load(CYCLE/'point_quality_inputs_v1/inputs.pt',weights_only=False,map_location='cpu');rng=random.Random(17);chosen=[]
    for stage,n in [('fit',8),('calibration',6)]:
        for kind in ['real','single','dual']:
            pool=[i for i,r in enumerate(rows) if r['metadata']['stage']==stage and r['metadata']['kind']==kind and r['valid'].any()]
            if kind=='real':
                bottom=[i for i in pool if rows[i]['metadata']['physical_bottom']];other=[i for i in pool if i not in bottom];ids=rng.sample(bottom,n//2)+rng.sample(other,n-n//2)
            else:ids=rng.sample(pool,n)
            chosen.extend(ids)
    assert len(chosen)==len(set(chosen))==42;assert all(rows[i]['metadata']['stage']!='validation' for i in chosen)
    manifests=[];evaluation=[];hashes={};inference=[]
    for i in chosen:
        r=rows[i];item=load_item(r['record']);rgb=item['batch']['images'][0,r['record']['views'].index(r['metadata']['view'])].permute(1,2,0).contiguous();a=(rgb*255).round().to(torch.uint8);assert torch.equal(a.float()/255,rgb)
        path=OUT/'images'/f'{i:04d}.png';Image.fromarray(a.numpy()).save(path);assert torch.equal(torch.tensor(np.asarray(Image.open(path)).copy()).float()/255,rgb);hashes.update(item['source_hashes'])
        manifests.append(dict(index=i,image=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_hw=list(a.shape[:2])))
        inference.append(dict(index=i,tokens=quality['tokens'][i],boxes=r['baseline']['boxes'],baseline_xy=r['baseline']['xy']))
        evaluation.append(dict(index=i,metadata=r['metadata'],gt=r['gt'],valid=r['valid'],pixels=r['pixels']))
    write_json(OUT/'inference_manifest.json',manifests);torch.save(inference,OUT/'inference.pt');torch.save(evaluation,OUT/'evaluation_only.pt');write_json(OUT/'source_hashes.json',hashes)
    write_json(OUT/'protocol.json',dict(seed=17,images=42,fit=24,calibration=18,development_validation=0,indices=chosen,sampling='seed17 random known-qualified TRAIN rows;8fit/6cal eachkind, halfrealbottom. Existing source-stage split unchanged; no selection by errors.',features='one shared native stride4 HRNet32ch feature per image, fullframepad32 noresize; exactfloat32 cache, no teachercoordinate labels',provenance='Teacher already saw some underlying TRAIN/cal real data; frozen base saw allTRAIN. Calibration only holds out new readout, not an independent external test.',planned_steps=300,arms=['appearance','zero_appearance'],budget='sameinitial learnedreadout/schedule/AdamW1e-3cosine1e-4 wd1e-4,3images/update oneeachkind; fixed300endpoint, no development evaluation',objective='samequalifiednosefeet normalizedpixelSmoothL1 beta.01 plus.01 GaussianCE sigma8inputpixels; no masks/labels forward',criteria='Before any broad cache/pilot: require appearance fit mean <=12px and <=half original in EACH real/single/dual, evaluate calibration separately and compare zeroappearance to test image feature utility. Failure not automatic training extension.'))
    print(json.dumps(dict(images=len(chosen),indices=chosen)))
if __name__=='__main__':main()
