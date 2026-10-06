"""Record fixed-final context-control results and next validation gate."""
import datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'view_context_control_v1'

def main():
    results=json.loads((BASE/'results.json').read_text())['results'];assert len(results)==6
    compact=[]
    for r in results:
        metrics={k:v for k,v in r['metrics'].items() if ('_validation/' in k and (k.endswith('/reliable') or k.endswith('/dual') or '/cam' in k))}
        history=json.loads((BASE/f'{r["arm"]}_seed{r["seed"]}_history.json').read_text())
        compact.append(dict(arm=r['arm'],seed=r['seed'],step=r['step'],metrics=metrics,late_curve=[dict(step=h['step'],single_real=h['metrics']['single/real_validation/reliable']['error_mean'],paired_real=h['metrics']['paired/real_validation/reliable']['error_mean']) for h in history[-3:]]))
    decision=dict(results=compact,
        scope='Same1223TRAIN images,112validation images.48real and64syntheticvalidation. Only encoder context changed; fresh800steps from per-seed balanced2400, shared image schedules, same bottom exposure, identical parameters and loss. Fixed final checkpoints, no best-val selection. Full decoder gradients verified; original labels and locked sessions preserved.',
        next='Render review_view_context_control_20261004.py once, inspect raw single/paired context queries. Run full RGB validate_core23_live_views_20261004.py --context-control paired then --context-control mixed, same36tasks and seed17, with cached paired and direct singleRGB parity checks. No more training until this gate and visual review. Then review_core23_live_queries_20261004.py with corresponding flag and compare fixed detector counts, matched-anchor/gatedPCK, source/camera and syntheticdual regressions. Do not infer independent-test success from these development scenes.',
        promotion=False)
    (BASE/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004');(dest/'view_context_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=decision['next'];(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(next=decision['next'],updated_utc=dt.datetime.now(dt.timezone.utc).isoformat());(CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    for r in results:
        print(json.dumps(dict(arm=r['arm'],seed=r['seed'],single_real=r['metrics']['single/real_validation/reliable']['error_mean'],paired_real=r['metrics']['paired/real_validation/reliable']['error_mean'],single_bottom=r['metrics']['single/real_validation/camA/feet']['error_mean'],paired_bottom=r['metrics']['paired/real_validation/camA/feet']['error_mean'],single_dual=r['metrics']['single/synthetic_validation/dual']['error_mean'],paired_dual=r['metrics']['paired/synthetic_validation/dual']['error_mean'])))

if __name__=='__main__':main()
