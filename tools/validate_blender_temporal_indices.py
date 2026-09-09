"""Read split/FrameToFrame scenes through the existing multi-instance loader."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.finetune_blendedmvs import BlendedMVSDataset,load_scene
from training.temporal_window_index import TemporalWindowIndex
from core23_temporal_format import write_json_atomic

torch.set_num_threads(2)
catalog=json.loads((BASE/'dataset/temporal_catalog.json').read_text())['datasets']
ids=[r['dataset_id'] for r in catalog if r['path'].startswith('BlenderPyMouse/') and r['temporal_eligible']]
windows=TemporalWindowIndex(BASE/'dataset',window_size=4,frame_step=2,window_stride=1,dataset_ids=ids)
report=[]
for rec,path,frames,rows,_ in windows.sequences:
    # Check every scene reference and every stored transition, then decode representative scenes.
    for row in rows:
        assert (path/row['scene']/'keypoints.npz').is_file()
        assert row['motion_source_id']==rec['motion_source_id']
        assert row['split']==rec['recommended_split']
    selected=[rows[i] for i in sorted({0,len(rows)//2,len(rows)-1})]
    scenes=[load_scene(path/r['scene']) for r in selected]
    views=min(2,min(len(s.frames) for s in scenes))
    ds=BlendedMVSDataset(scenes=scenes,num_views=views,image_width=128,image_height=128,
        samples_per_scene=1,seed=5706,use_masked_images=False,pfm_flip_y=False,
        canonicalize_camera=False,scale_mode='raw',top_srcs=views,load_keypoints=True)
    ds.max_instances=2
    shapes=[]
    for i in range(len(scenes)):
        sample=ds[i]
        assert sample['images'].shape==(views,3,128,128)
        assert sample['actual_instance_count']==1
        assert sample['instance_keypoint_xy'].shape[1]==2
        assert not sample['instance_valid'][:,1].any()
        assert sample['instance_keypoint_valid'][:,0].any()
        assert torch.isfinite(sample['images']).all()
        shapes.append(list(sample['instance_keypoint_xy'].shape))
    report.append({'dataset':rec['dataset_id'],'frame_count':len(frames),
        'source_frame_step':rec['source_frame_step'],'motion_source_id':rec['motion_source_id'],
        'split':rec['recommended_split'],'loaded_scene_names':[r['scene'] for r in selected],
        'instance_keypoint_shapes':shapes})
    print(f'{rec["dataset_id"]}: first/middle/last scenes passed',flush=True)
write_json_atomic(BASE/'output/temporal_dataset_audit/blender_index_validation.json',
    {'passed':True,'dataset_count':len(report),'loaded_scenes':sum(len(r['loaded_scene_names']) for r in report),
     'windows_checked':len(windows),'datasets':report})
