"""Record the bounded continuation decision without changing source annotations."""
import json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'

def main():
    rows=[]
    for name,arm in [('decoder_unfrozen_extended2400_v1','unfrozen'),('bottom_balanced_extended2400_v1','balanced')]:
        for result in json.loads((CYCLE/name/'results.json').read_text())['results']:
            history=json.loads((CYCLE/name/f'{arm}_seed{result["seed"]}_history.json').read_text())
            keys=['real_validation/reliable','real_validation/camA/feet','real_validation/camD/feet','synthetic_validation/dual','real_train/camA/feet']
            rows.append(dict(run=name,seed=result['seed'],final={k:result['metrics'][k]['error_mean'] for k in keys},late_curve=[dict(step=h['step'],metrics={k:h['metrics'][k]['error_mean'] for k in keys}) for h in history[-3:]]))
    decision=dict(rows=rows,decision='Proceed to matched full-RGB 1/2/all-view audit, NOT another training extension.',
        evidence='Balanced2400 improves all three seeds versus uniform2400: bottom feet63.43/68.59/69.96 versus89.21/95.41/89.52px; reliable real36.92/40.41/39.17 versus40.65/43.45/41.01px. Syntheticdual worsens by0.22-0.29px; camDfeet worsens by1.88-2.69px. Late2000to2400 bottom improvement is0.83-1.32px for balanced, so do not infer large further gains from more steps.',
        visual_review=['04_real.jpg: hind feet improve in some queries, front feet still collapse near head/abdomen; duplicateboxes persist.','09_real.jpg: side points remain imperfect; aggregate side regression must not be hidden by overall mean.','02_synthetic.jpg: both instances still detected, semanticpoint precision remains imperfect.'],
        scope='Errors are in the resized model input pixel coordinates, conditional on box/objectness association. 48 real development images, only8 camA images32footpoints; not independent external validation.',
        next='Validate both seed17 uniform2400 and balanced2400 with full encoder from RGB, same previous36 tasks and no GT query selection; exact cached all-query parity first. No production promotion, no label edits, no fabricated curves.',
        production_changed=False)
    path=CYCLE/'bottom_balanced_extended2400_v1/decision.json';path.write_text(json.dumps(decision,indent=2)+'\n')
    shutil.copy2(path,CYCLE/'review_bottom_balanced_extended2400_v1/decision.json')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004/review_bottom_balanced_extended2400_v1')
    shutil.copy2(path,dest/'decision.json')
    plan=json.loads((CYCLE/'plan.json').read_text())
    plan['next_command']='Matched full-RGB decoder audit is running sequentially: validate_core23_live_views_20261004.py --decoder-adapted uniform then --decoder-adapted balanced. Check actualPID/status before starting anything. Outputs live_view_decoder_uniform2400_audit_v1 and live_view_decoder_balanced2400_audit_v1. Once BOTH complete, render review_core23_live_queries_20261004.py with each same flag, inspect rawqueries and aggregate 1/2/all-view detector/gated-PCK/anchor metrics. Read bottom_balanced_extended2400_v1/decision.json. No further blind extension or promotion. If gains survive, choose next bounded test from failure evidence; protect locked sessions and harddeadline.'
    (CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
