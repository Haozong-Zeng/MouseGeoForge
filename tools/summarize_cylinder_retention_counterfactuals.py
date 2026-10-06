"""Fixed-matching detection-gate attribution on already audited outputs."""
import json
import numpy as np
from run_depth_pose_ablation import ROOT,write_json,fingerprint

OUT=ROOT/'output/cylinder_retention_geometry_v1'


def main():
    assert json.loads((OUT/'report.json').read_text())['passed']
    path=OUT/'dual_points.json';digest=fingerprint(path);points=json.loads(path.read_text())
    by={tag:{(p['scene'],p['role'],p['instance'],p['keypoint']):p for p in points if p['checkpoint']==tag} for tag in ('original','v2','mixture')}
    results=[]
    for reference in ('original','v2'):
        a,b=by[reference],by['mixture'];assert a.keys()==b.keys()
        for role in ('all','top','side'):
            for part in ('all','body','tail'):
                keys=[k for k in a if (role=='all' or a[k]['role']==role) and (part=='all' or (a[k]['part']=='tail')==(part=='tail'))]
                def pck(coords,gates):return float(np.mean([coords[k]['nme']<=.1 and gates[k]['found'] for k in keys]))
                results.append(dict(reference=reference,role=role,part=part,count=len(keys),
                    reference_pck=pck(a,a),mixture_pck=pck(b,b),mixture_xy_reference_gate=pck(b,a),reference_xy_mixture_gate=pck(a,b)))
    assert fingerprint(path)==digest
    write_json(OUT/'gate_counterfactuals.json',dict(points_sha256=digest,
        scope='GT-matched instance-level counterfactual gates, not deployable box fixes or query reassignment; effects interact and are not additive causal estimates.',results=results))
    print(json.dumps([r for r in results if r['part']=='body'],indent=2))


if __name__=='__main__':main()
