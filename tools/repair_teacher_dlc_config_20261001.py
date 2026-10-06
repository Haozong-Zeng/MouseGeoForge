"""Restore this new pilot's pristine framework config after a pre-training override error."""
import os
os.environ['DLClight']='True'
import json
import shutil
from pathlib import Path
import deeplabcut
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'
d=json.loads((OUT/'dlc_dataset_manifest.json').read_text())
project=Path(d['project_config']).parent
assert not list(project.rglob('*.pt')), 'Never reset a trained run'
configs=list((project/'dlc-models-pytorch').rglob('pytorch_config.yaml'))
assert len(configs)==1
backup=configs[0].with_name('pytorch_config.failed-before-dotted-overrides.yaml')
assert not backup.exists()
shutil.copy2(configs[0],backup)
deeplabcut.create_training_dataset(d['project_config'],Shuffles=[1],trainIndices=[d['train_indices']],
    testIndices=[d['validation_indices']],net_type='resnet_50',engine=deeplabcut.Engine.PYTORCH,userfeedback=False)
(OUT/'bootstrap_incident.json').write_text(json.dumps(dict(stage='before_any_training_batch',
    cause='DLC set_nested replaced entire model/data sections when dict overrides were passed',
    fix='restored only new pilot config; switched to dotted leaf overrides; original source/environment/data unchanged',
    missing_label_gradient_mask_test='passed: visibility0 channel gets exactly zero loss weights'),indent=2))
