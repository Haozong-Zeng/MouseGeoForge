"""Export exact 144 current inference inputs, separate from evaluation labels."""
import datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image
from train_pixel_adapter_matched_20261005 import prepare_rows
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'specialist_roi_inputs_v1'
def main():
    assert not OUT.exists();OUT.mkdir();(OUT/'images').mkdir();torch.set_num_threads(2);rows=prepare_rows();records=[];hashes={};evaluation=[]
    for i,r in enumerate(rows):
        if r['metadata']['stage']!='validation':continue
        assert r['record'].get('session') not in ['GR2','UR2','UR3'];item=load_item(r['record']);vi=r['record']['views'].index(r['metadata']['view']);rgb=item['batch']['images'][0,vi].cpu();h,w=rgb.shape[-2:]
        u8=(rgb*255).round().clamp(0,255).byte();restored=u8.float()/255;delta=float((restored-rgb).abs().max());assert delta==0.,('Cannot silently quantize',i,delta)
        path=OUT/'images'/f'{i:04d}.png';Image.fromarray(u8.permute(1,2,0).numpy()).save(path);decoded=torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1).float()/255;assert torch.equal(decoded,rgb)
        records.append(dict(index=i,image=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),image_hw=[h,w],pad_hw=[(-h)%32,(-w)%32],boxes=r['baseline']['boxes'].tolist(),objectness_logits=r['baseline']['objectness_logits'].tolist(),baseline_xy=r['baseline']['xy'].tolist(),pixel_roundtrip_max_abs=delta))
        evaluation.append(dict(index=i,metadata=r['metadata'],record=r['record'],gt=r['gt'],valid=r['valid'],pixels=r['pixels']));hashes.update(item['source_hashes'])
    assert len(records)==144;torch.save(evaluation,OUT/'evaluation_only.pt');write_json(OUT/'inference_manifest.json',records);write_json(OUT/'source_hashes.json',hashes)
    write_json(OUT/'protocol.json',dict(images=144,scope='48real32single64dual fixed development; one no-training specialist ROI diagnostic, not heldoutarchitectureablation.',input='Exact original load_item floatRGB, verifiedlossless uint8 roundtrip ALLpixels before PNGexport; detectorpredictions andall6slots only. GTstoredseparately andnotreadbyinference.',padding='Right/bottomzeroRGB to32multiple; noresize/noGTcrop; heatmapdecoder excludespadding.',prohibited='No dataset-kindgate,GTcount,GTboxorGTpoint inforward; no tails/abdomenchanges, no newpseudoGT.',bytes=sum(p.stat().st_size for p in (OUT/'images').iterdir()),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('COMPLETE',len(records),'bytes',sum(p.stat().st_size for p in (OUT/'images').iterdir()))
if __name__=='__main__':main()
