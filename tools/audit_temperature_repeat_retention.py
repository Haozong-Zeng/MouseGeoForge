"""Saved-prediction-only explanation of the second-seed dual PCK gate."""
from collections import Counter
import json
import numpy as np
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint
from mousegeoforge.retention_geometry_audit import matched_views


def main():
    torch.set_num_threads(2);out=ROOT/'output/cylinder_temperature_seed29_v1'
    result=json.loads((out/'results.json').read_text());assert json.loads((out/'delivery_verification.json').read_text())['passed']
    first=ROOT/'output/cylinder_adaptive_temperature_v1';base=ROOT/'output/cylinder_source_mix_v1'
    folders=dict(original=base/'initial',seed17_off=base/'source_mix',seed17_on=first/'adaptive_temperature',seed29_off=out/'temperature_off',seed29_on=out/'temperature_on')
    points={};detections={};sources={};parity=[]
    for tag,folder in folders.items():
        points[tag]={};detections[tag]={}
        for idx,r in enumerate(result['baseline']['records']):
            if r['group']!='old_dual_pair':continue
            path=folder/f'prediction_{idx:03d}.pt';sources[str(path.relative_to(ROOT))]=fingerprint(path)
            s=torch.load(path,weights_only=True,map_location='cpu');local=[]
            for item in matched_views(s):
                key=(idx,item['view'],item['instance'])
                detections[tag][key]=dict(scene=r['scene'],role=item['role'],found=item['found'],iou=item['iou'],score=item['score'])
                for k in torch.where(item['valid'])[0].tolist():
                    p=dict(role=item['role'],part='tail' if 7<=k<14 else 'body',nme=float(item['errors'][k]),found=item['found'])
                    points[tag][(*key,k)]=p;local.append(p)
            parity.append(abs(float(np.mean([p['found'] and p['nme']<=.1 for p in local]))-s['metrics']['detection_aware_pck_0.1']))
    assert max(parity)<1e-7
    summary=[]
    for tag in folders:
        for role in ('all','side','top'):
            ds=[p for p in detections[tag].values() if role=='all' or p['role']==role]
            failure=Counter('score_and_box' if p['score']<.5 and p['iou']<.5 else 'score_only' if p['score']<.5 else 'box_only' for p in ds if not p['found'])
            for part in ('all','body','tail'):
                ps=[p for p in points[tag].values() if (role=='all' or p['role']==role) and (part=='all' or p['part']==part)]
                summary.append(dict(checkpoint=tag,role=role,part=part,points=len(ps),nme=float(np.mean([p['nme'] for p in ps])),
                    matched_pck=float(np.mean([p['nme']<=.1 for p in ps])),detection_pck=float(np.mean([p['found'] and p['nme']<=.1 for p in ps])),
                    found=sum(p['found'] for p in ds),instance_views=len(ds),failure_reasons=dict(failure)))
    counterfactual=[]
    for coords,gates in [('seed29_on','seed17_on'),('seed29_on','seed29_off'),('seed17_on','seed29_on')]:
        assert points[coords].keys()==points[gates].keys()
        for part in ('all','body','tail'):
            keys=[k for k,p in points[coords].items() if part=='all' or p['part']==part]
            counterfactual.append(dict(coordinates=coords,borrowed_detection_gate=gates,part=part,
                pck=float(np.mean([points[gates][k]['found'] and points[coords][k]['nme']<=.1 for k in keys]))))
    transitions=[]
    for key in detections['seed17_on']:
        a=detections['seed17_on'][key];b=detections['seed29_on'][key]
        if a['found']!=b['found']:transitions.append(dict(record=key[0],view=key[1],instance=key[2],scene=a['scene'],role=a['role'],seed17=a,seed29=b))
    report=dict(summary=summary,diagnostic_counterfactuals=counterfactual,detection_transitions=transitions,
        source_sha256=sources,saved_metric_max_difference=max(parity),encoder_forwards=0,optimizer_steps=0,
        caveat='All rows use identical saved GT-box-assisted correspondences. Borrowing detection gates is an explanation, not a deployable correction or revised acceptance criterion.')
    assert all(fingerprint(ROOT/p)==h for p,h in sources.items())
    write_json(out/'dual_retention_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','summary')},indent=2))
    print(json.dumps([r for r in summary if r['role']=='all'],indent=2))


if __name__=='__main__':main()
