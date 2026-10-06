"""Exhaust existing dual development scenes; no threshold/weight selection."""
import audit_composite_allviews_20261005 as audit
import torch
audit.OUT=audit.CYCLE/'dual_allscene_association_audit_v1'
if __name__=='__main__':
    data=torch.load(audit.CYCLE/'singleview_fusion_inputs_seed17_v1/validation.pt',map_location='cpu',weights_only=False)
    scenes={m['record']['scene'] for m in data['metadata'] if m['record']['kind']!='real' and m['expected_instances']==2}
    assert len(scenes)==16
    del data
    try:audit.main({2:len(scenes)})
    except Exception:
        if audit.OUT.exists():audit.write_json(audit.OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
