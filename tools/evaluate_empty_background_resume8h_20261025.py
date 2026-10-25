"""Independent empty-DEV diagnostic; never admit negatives to the current pair."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from mousegeoforge.semseg_training import MouseSemseg,digest,letterbox,make_encoder,native_mask,write_json
from mousegeoforge.models.mouse_semseg import semantic_scores
from calibrated_mouse_data_20261025 import gpu_lock

OUT=ROOT/'output/calibrated_mouse_10h_20261025'
EMPTY=ROOT/'output/calibrated_mouse_resume8h_20261025/empty_background_v1'


def qa():
    if (EMPTY/'manifest.json').exists():
        previous=json.loads((EMPTY/'qa.json').read_text());assert previous['passed']
        assert previous['manifest_sha256']==digest(EMPTY/'manifest.json');return
    cfg=json.loads((EMPTY/'render_config.json').read_text())
    completion=json.loads((EMPTY/'formal/completion.json').read_text())
    assert completion['passed'] and completion['device']=='CPU' and completion['model_calls']==0
    assert len(completion['records'])==12 and cfg['cycles_samples']==32
    assert json.loads((EMPTY/'smoke_visual_review.json').read_text())['passed']
    for relative,pin in cfg['code_pins'].items():assert digest(ROOT/relative)==pin
    design=json.loads((ROOT/cfg['design']).read_text())
    for source in design['sources']:assert digest(ROOT/source['path'])==cfg['source_sha256'][source['id']]
    names=dict(climb='PyMouseStudio-BrownFur-ClimbWall-Core23-ComplexBackgroundV5-EmptyBackground-Dev',
        moseq='PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200-ComplexBackgroundV5-EmptyBackground-Dev')
    datasets={source:ROOT/'dataset/BlenderPyMouse'/name for source,name in names.items()}
    for dataset in datasets.values():dataset.mkdir(exist_ok=False)
    counters=dict(climb=0,moseq=0);items=[]
    for record in completion['records']:
        positive=json.loads((ROOT/record['positive_record']).read_text());source=positive['motion_id']
        assert positive['split']=='val' and positive['dev_condition']=='dev_primary'
        assert record['hidden_mouse_family']==cfg['approved_mouse_family'][source]
        assert not set(record['hidden_mouse_family'])&set(record['remaining_visible_meshes'])
        folder=datasets[source]/f"frame_{counters[source]:06d}";counters[source]+=1;folder.mkdir()
        for sub in ['blended_images','masks','cams','provenance']: (folder/sub).mkdir()
        metadata=[]
        for view in record['views']:
            index=view['view'];image=ROOT/view['image'];assert digest(image)==view['sha256'] and view['device']=='CPU'
            rgb=np.array(Image.open(image).convert('RGB'));assert rgb.shape==(384,384,3)
            np.testing.assert_allclose(view['K'],positive['views'][index]['K'],atol=1e-8)
            np.testing.assert_allclose(view['E'],positive['views'][index]['E'],atol=1e-8)
            lossless=folder/'provenance'/f'rgb_{index}.png';shutil.copyfile(image,lossless)
            Image.fromarray(rgb).save(folder/'blended_images'/f'{index:08d}.jpg',quality=95,subsampling=0)
            label=folder/'masks'/f'{index:08d}.png';Image.fromarray(np.zeros((384,384),np.uint8)).save(label)
            k,e=np.array(view['K']),np.array(view['E'])
            lines=['extrinsic',*[' '.join(f'{v:.17g}' for v in row) for row in e],'','intrinsic',
                   *[' '.join(f'{v:.17g}' for v in row) for row in k],'']
            (folder/'cams'/f'{index:08d}_cam.txt').write_text('\n'.join(lines)+'\n')
            item=dict(id=record['id']+f'_v{index}',image=lossless.relative_to(ROOT).as_posix(),
                image_sha256=digest(lossless),label=label.relative_to(ROOT).as_posix(),label_sha256=digest(label),
                source=source,rig=positive['rig'],environment=positive['environment_id'],bottom=index==0,
                split='diagnostic_dev',mouse_instance_count=0,depth_available=False,K=view['K'],E=view['E'])
            items.append(item);metadata.append(item)
        write_json(folder/'scene_meta.json',dict(split='diagnostic_dev',training_admission=False,
            mouse_instance_count=0,keypoint_instances=[],depth_available=False,
            source_render_record=record['positive_record'],views=metadata,
            note='2D empty-background diagnostic; no mouse depth or keypoint GT exists'))
    assert len(items)==36
    for dataset in datasets.values():
        (dataset/'README.md').write_text('Empty-background DEV diagnostic only. Full-image RGB, known-empty binary masks and calibrated cameras. No mouse depth/keypoint GT; not admitted to current TRAIN/DEV pair. Camera units mm, OpenCV world-to-camera. Raw lossless RGB is in provenance; standard JPEG in blended_images.\n')
    write_json(EMPTY/'manifest.json',dict(schema='mouse-empty-diagnostic-v1',items=items,
        training_admission=False,no_threshold_selection=True,frame_fpr_rules=['any foreground pixel','any component>=20px','any component>=100px'],
        registered_model='M common TRAIN endpoint EMA; no DEV tuning',source_render_completion_sha256=digest(EMPTY/'formal/completion.json')))
    write_json(EMPTY/'qa.json',dict(passed=True,views=len(items),scenes=12,source_files_unchanged=True,
        camera_replay_passed=True,empty_object_inventory_passed=True,
        manifest_sha256=digest(EMPTY/'manifest.json'),canonical_roots=[p.relative_to(ROOT).as_posix() for p in datasets.values()],
        depth_available=False,no_current_training_admission=True))


def evaluate():
    qa()
    destination=EMPTY/'M_common_endpoint_eval';destination.mkdir(exist_ok=False)
    progress=json.loads((OUT/'semseg_campaign_s17_v1/progress.json').read_text())
    checkpoint=Path(progress['parents']['M']);assert digest(checkpoint)=='30a50a5cfb67b7639b1d2e24988789a11460b8787ed9eea46963b2e9e148b8dc';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['protocol']['arm']=='M' and saved['step']==progress['common_endpoint']
    items=json.loads((EMPTY/'manifest.json').read_text())['items'];results=[]
    with gpu_lock(),torch.inference_mode():
        encoder=make_encoder(ROOT/'vggt-omega/vggt_omega_1b_512.pt')
        model=MouseSemseg(encoder).cuda().eval().requires_grad_(False)
        parameters=dict(model.named_parameters())
        for name,value in saved['ema'].items():
            assert name.startswith('head.');parameters[name].copy_(value)
        for row in items:
            assert digest(ROOT/row['image'])==row['image_sha256'] and digest(ROOT/row['label'])==row['label_sha256']
            rgb=np.array(Image.open(ROOT/row['image']).convert('RGB'));target=np.zeros(rgb.shape[:2],np.uint8)
            image,label,meta=letterbox(rgb,target,512)
            with torch.autocast('cuda',dtype=torch.bfloat16):predictions=model(image[None].cuda(),'M');scores=semantic_scores(predictions[-1],(512,512))
            mask=native_mask(scores.argmax(1)[0].cpu().numpy(),meta).astype('uint8')
            _,_,stats,_=cv2.connectedComponentsWithStats(mask,8);areas=stats[1:,cv2.CC_STAT_AREA]
            cv2.imwrite(str(destination/(row['id']+'.png')),mask*255)
            results.append(dict(id=row['id'],source=row['source'],rig=row['rig'],bottom=row['bottom'],
                false_foreground_pixels=int(mask.sum()),foreground_fraction=float(mask.mean()),
                any_foreground=bool(mask.any()),component_ge20px=bool((areas>=20).any()),component_ge100px=bool((areas>=100).any())))
    summary={key:float(np.mean([r[key] for r in results])) for key in ['foreground_fraction','any_foreground','component_ge20px','component_ge100px']}
    write_json(destination/'evaluation.json',dict(summary=summary,views=results,full_denominator=36,
        checkpoint_sha256=digest(checkpoint),common_endpoint=saved['step'],weights='M EMA',no_threshold_selection=True,
        no_postprocess_or_component_deletion=True,synthetic_only=True,independent_real_background_claim=False,
        unique_rgb_hashes=len({r['image_sha256'] for r in items})))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['qa','evaluate']);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    qa() if args.command=='qa' else evaluate()
