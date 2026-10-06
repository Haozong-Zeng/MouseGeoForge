"""Frozen RGB predictions for all existing21 dual TRAIN frames, no learning."""
import audit_composite_allviews_20261005 as audit
import torch
audit.OUT=audit.CYCLE/'dual_train_allview_affinity_context_v1'
if __name__=='__main__':
    data=torch.load(audit.CYCLE/'singleview_fusion_inputs_seed17_v1/train.pt',map_location='cpu',weights_only=False)
    scenes={m['record']['scene'] for m in data['metadata'] if m['record']['kind']!='real' and m['expected_instances']==2}
    validation=torch.load(audit.CYCLE/'singleview_fusion_inputs_seed17_v1/validation.pt',map_location='cpu',weights_only=False)
    assert len(scenes)==21 and scenes.isdisjoint({m['record']['scene'] for m in validation['metadata']})
    del data,validation
    try:audit.main({2:21},source_role='train')
    except Exception:
        if audit.OUT.exists():audit.write_json(audit.OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
