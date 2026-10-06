"""No-retraining projection diagnostic; no validation threshold optimization."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from mousegeoforge.surface_curve_projection import project_curve_four_controls
from mousegeoforge.surface_curve_supervision import points_to_polyline
from review_sparse_real_curve_adaptation_20261003 import crossings
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';SOURCE=CYCLE/'curve_heatmap_supervision_v1';OUT=CYCLE/'curve_projection_diagnostic_v1'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    saved=torch.load(CYCLE/'expanded_curve_features_v1/frozen_inputs.pt',map_location='cpu',weights_only=False);meta=saved['metadata'];gt=saved['tensors']['gt'];pixels=saved['tensors']['pixels'];arrays={};rows=[]
    with np.load(SOURCE/'predictions.npz') as z:
        for seed in [17,29,43]:
            name=f'heatmap_seed{seed}';p=torch.from_numpy(z[name]);q=project_curve_four_controls(p)
            assert torch.equal(p[:,[0,-1]],q[:,[0,-1]])
            arrays[name]=p.numpy();arrays[f'projected_seed{seed}']=q.numpy()
    # Oracle only measures representational approximation of known GT; never
    # used to choose a projection strength or select model checkpoints.
    arrays['oracle_gt_projection']=project_curve_four_controls(gt).numpy()
    for arm,arr in arrays.items():
        for role in ['train','validation']:
            values=collections.defaultdict(list);by_source=collections.defaultdict(list)
            for i,m in enumerate(meta):
                if m['role']!=role:continue
                p=torch.from_numpy(arr[i])*pixels[i];g=gt[i]*pixels[i]
                dist=points_to_polyline(p,g,torch.ones(len(g),dtype=torch.bool))[0].sqrt()
                rev=points_to_polyline(g,p,torch.ones(len(p),dtype=torch.bool))[0].sqrt()
                values['pred_to_gt_px'].extend(dist.tolist());values['gt_to_pred_px'].extend(rev.tolist());values['endpoints_px'].extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).tolist())
                values['self_crossing'].append(float(crossings(arr[i])>0));values['gt_self_crossing'].append(float(crossings(gt[i].numpy())>0))
                by_source[m['dataset']].extend(dist.tolist())
            rows.append(dict(arm=arm,role=role,metrics={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()},pred_gt_by_source={k:float(np.mean(v)) for k,v in by_source.items()}))
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(summary=rows,
        source_sha256=fingerprint(SOURCE/'predictions.npz'),protocol='Fixed4controlcubicfit ofpredicted33uniformmaterialu samples, constrainedsameendpoints. No training/parameterselection/GTinput. OracleGTprojection isapproximationfloor only, notmodelresult.',
        limitations='Cannotfixendpointlocalization or guaranteeanatomy/no loops. Samefrozendevelopmentvalidation. No realpseudoGT or modelpromotion.'))
    import review_diverse_curve_features_20261003 as review
    review.BASE=OUT;review.INPUTS=CYCLE/'expanded_curve_features_v1/frozen_inputs.pt';review.DISPLAY_ARMS=['heatmap_seed17','projected_seed17'];review.main()
    path=OUT/'visual_review/index.html';html=path.read_text().replace('<h1>Frozen-model curve feature comparison</h1>','<h1>Fixed endpoint-preserving cubic projection</h1><p>NOretraining. Leftprediction=6controlheatmapcurve; right=4controlleast-squaresprojection withsameendpoints. Materialu, not2Darclength. NoGTusedtoproject predictions. Separate oracleGTfit iscapacitydiagnostic only.</p>');path.write_text(html)
    print(json.dumps([r for r in rows if r['role']=='validation'],indent=2))


if __name__=='__main__':main()
