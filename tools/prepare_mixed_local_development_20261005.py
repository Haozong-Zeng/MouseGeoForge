"""Fixed final-only development cohort, separated from current local-model FIT/CAL."""
import sys,json,hashlib,collections,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
C=ROOT/'output/singleview_fusion_cycle_20261005';OUT=C/'mixed_local_development_inputs_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();torch.set_num_threads(2);rows=prepare_rows();chosen=[i for i,r in enumerate(rows) if r['metadata']['stage']=='validation'];assert len(chosen)==144
    fit=torch.load(C/'mixed_local_fulltrain_rgb_inputs_v1/evaluation_only.pt',weights_only=False,map_location='cpu');oldscenes={r['metadata']['scene'] for r in fit};assert all(rows[i]['metadata']['scene'] not in oldscenes for i in chosen)
    protocol=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),indices=chosen,images=144,kind_counts=dict(collections.Counter(rows[i]['metadata']['kind'] for i in chosen)),source_counts=dict(collections.Counter(rows[i]['metadata']['source'] for i in chosen)),checkpoint_policy='Predeclared originalmixed800, native4000, rotated4000 fixed endpoints only. No best-checkpointselection, earlystop, labelrepair, augmentationtuning or post-development training inthiscycle.',purpose='All originaldevelopment rows, no baseline-detection qualification or error-based selection. FR2/AZR2L2 reused by previous work, not pristine external test. None in currentmixed FIT/internalCAL. LockedGR2/UR2/UR3 neverread.',inputs='Same actual RGB pipeline/longside768, roundtrip PNGexact; labels in evaluation_only.pt separate fromRGBinference manifest. No GT ROI/count/domain/pointinputforward.',labels='All known realcoordinates visibilityUNKNOWN; synthetic explicitvisible eligible, occluded separately. Missingcoordinatesignore, includingimageswithnoeligiblepoints.',production_changed=False)
    write(OUT/'protocol.json',protocol);cache={};manifest=[];evaluation=[];hashes={};pointcounts=collections.Counter();empty=collections.Counter()
    for n,i in enumerate(chosen):
        r=rows[i];m=r['metadata'];assert m['role']=='validation';key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in cache:cache[key]=load_item(r['record'])
        item=cache[key];batch=item['batch'];v=r['record']['views'].index(m['view']);t=instance_targets_from_batch(batch);rgb=batch['images'][0,v].permute(1,2,0);a=(rgb*255).round().byte();assert torch.equal(a.float()/255,rgb);path=OUT/'images'/f'{i:04d}.png';Image.fromarray(a.numpy()).save(path);assert torch.equal(torch.tensor(np.asarray(Image.open(path)).copy()).float()/255,rgb)
        target=t['keypoint_xy'][0,v,:,KP].clone();known=t['keypoint_valid'][0,v,:,KP].clone()&t['instance_valid'][0,v,:,None];vv=t['keypoint_visibility_valid'][0,v,:,KP].clone();visible=t['keypoint_visible'][0,v,:,KP].clone();eligible=known if m['kind']=='real' else known&vv&visible
        if m['kind']=='real':assert not vv.any() and m['source'] in ['FR2','AZR2L2']
        assert torch.isfinite(target[eligible]).all();manifest.append(dict(index=i,image=str(path.relative_to(ROOT)),sha256=sha(path),image_hw=list(a.shape[:2])));evaluation.append(dict(index=i,metadata=m,target=target,coordinate_valid=known,visibility_valid=vv,visible=visible,eligible=eligible,pixels=r['pixels'],instance_valid=t['instance_valid'][0,v].clone()));pointcounts[m['kind']]+=int(eligible.sum());empty[m['kind']]+=int(not eligible.any());hashes.update(item['source_hashes']);occ=ROOT/r['record']['scene']/'keypoint_occlusion.npz'
        if occ.exists():hashes[str(occ.relative_to(ROOT))]=sha(occ)
        if (n+1)%24==0:print('RGB',n+1,flush=True)
    torch.save(evaluation,OUT/'evaluation_only.pt');write(OUT/'inference_manifest.json',manifest);write(OUT/'source_hashes.json',hashes);protocol.update(eligible_point_counts=dict(pointcounts),images_without_eligible_points=dict(empty),png_bytes=sum((ROOT/z['image']).stat().st_size for z in manifest),roundtrip_exact=True);write(OUT/'protocol.json',protocol);print(json.dumps({k:v for k,v in protocol.items() if k!='indices'},indent=2))
if __name__=='__main__':main()
