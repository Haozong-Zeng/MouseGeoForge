"""Pinned official Quadruped weights and CPU-only transfer audit; no training."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from bottom_anchor_common_20261002 import ROOT, OUT, budget


def main():
    _,_,check=budget(600)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['DLClight']='True'
    repo='mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped'
    revision='1ad30fb80cd666f1e5c91578d1cf63bccfa84f80'
    filename='superanimal_quadruped_hrnet_w32.pt'
    size=118094850
    digest='a42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e'
    dest=OUT/'superanimal_quadruped_preparation_v1';dest.mkdir(exist_ok=True)
    path=dest/filename
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
    assert size<150_000_000  # bounded one-pose-weight download, no detector/dataset
    if not path.exists():
        part=dest/(filename+'.partial')
        with urllib.request.urlopen(url,timeout=30) as response:
            with part.open('xb') as f:
                total=0
                while chunk:=response.read(1024*1024):
                    check();total+=len(chunk);assert total<=size
                    f.write(chunk)
        assert total==size
        assert hashlib.sha256(part.read_bytes()).hexdigest()==digest
        part.rename(path)
    assert path.stat().st_size==size and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    import torch
    from deeplabcut.core.config import read_config_as_dict
    from deeplabcut.modelzoo import build_weight_init
    from deeplabcut.pose_estimation_pytorch.models.model import PoseModel,filter_state_dict
    from deeplabcut.utils import auxiliaryfunctions
    torch.set_num_threads(4);torch.manual_seed(17)
    source=ROOT/'DeepLabCut/deeplabcut/modelzoo/model_configs/hrnet_w32.yaml'
    cfg=read_config_as_dict(source)
    model_cfg=cfg['model'];head=model_cfg['heads']['bodypart']
    head['target_generator']['num_heatmaps']=7
    head['heatmap_config']['channels']=[32,7]
    head['target_generator']['gradient_masking']=True
    head['target_generator']['background_weight']=0.
    project_cfg=auxiliaryfunctions.read_config(OUT/'dlc_75to150_v1/project/config.yaml')
    wi=build_weight_init(project_cfg,super_animal='superanimal_quadruped',model_name='hrnet_w32',
        detector_name=None,with_decoder=False,customized_pose_checkpoint=path)
    model=PoseModel.build(model_cfg,weight_init=wi,pretrained_backbone=False).cpu().eval()
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    backbone=filter_state_dict(checkpoint['model'],'backbone')
    assert all(torch.equal(model.backbone.state_dict()[k],v) for k,v in backbone.items())
    with torch.inference_mode():
        output=model(torch.zeros(1,3,768,704))
    shapes={h:{k:list(v.shape) for k,v in values.items()} for h,values in output.items()}
    assert all(torch.isfinite(v).all() for values in output.values() for v in values.values())
    report=dict(model='SuperAnimal-Quadruped',not_topviewmouse=True,architecture='hrnet_w32',
        repo=repo,revision=revision,url=url,bytes=size,sha256=digest,checkpoint=str(path),
        backbone_tensors_verified_exact=len(backbone),output_shapes=shapes,
        device='CPU',GPU_training_started=False,weight_initialization=wi.to_dict(),
        strategy='Official SuperAnimal pretrained backbone transfer; all 7 output channels freshly initialized. Not ImageNet-only and not decoder fine-tuning.',
        target_ids_1based=[1,6,7,15,16,17,18],
        semantic_audit={
            'nose':'Plausible nose correspondence; no decoder reuse in this conservative baseline.',
            'dorsal_surface_6_7':'No established match to quadruped neck/back landmarks; must train new outputs.',
            'paws_15_16_17_18':'Paw names match limb identity but exact paw-center versus endpoint semantics not established; new outputs avoid asserting equivalence.'},
        input_policy='Fixed whole image704x768; no GT or predicted ROI crop, no detector, same masked missing labels and frozen split.',
        next='Create isolated derived project with this audited model config and weight_init, method bu, detector None; GPU smoke-test missing-label gradient and full training after DLC200. Audit augmentation and validation transforms. Do not auto-adapt validation video.',
        official_reference='https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html')
    (dest/'model_config.json').write_text(json.dumps(model_cfg,indent=2,default=str))
    (dest/'preflight.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':main()
