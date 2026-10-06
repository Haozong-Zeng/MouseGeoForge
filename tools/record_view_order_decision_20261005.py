"""Record checked order sensitivity and a bounded causal probe, not promotion."""
import datetime as dt,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'review_view_order_v1'

def main():
    report=json.loads((OUT/'results.json').read_text());assert report['original_order_replay_cases']==36 and report['max_original_order_delta']==0 and report['physical_view_gt_alignment'] and report['rgb_alignment_checked']
    decision=dict(conclusion='Input order sensitivity is real in this frozen candidate, not an array-index/GT/RGB mismatch. First-frame special tokens are a plausible mechanism, NOT yet proven sole cause.',
        evidence='Same physical anchor: real paired46.62to63.89px after reversal; real all55.71to62.45. Syntheticall16.24to40.50; detection21/32to14/32. Original-order36task replay exactlyequal. Labels aligned by cameraID exactlyequal; visualization RGB exactsame. Prediction-only set matching mean per-view nose/feet displacement realpaired35.66px,realall32.24px; not an accuracy metric.',
        visual=['00_real_order.jpg inspected: samecamA/image, pointlocations and boxes change acrossorder; reversal doesnot reliably improve, some frontfeet stillnearface.','03_synthetic_order.jpg inspected: identicalsideimage, all_reversed box and feet shift upward versus originalall.'],
        next='Next bounded causal diagnostic BEFORE more training: runtime-only shared special-token intervention in the frozen aggregator. Copy first-frame camera_token and register_token values to the other-frame slots (allframes use the SAME pretrainedfirst token); preserve originals and do not edit checkpoint files or production factory. Compare exact same1/2/all and reversed-order60tasks, seed17 mixedpose+mixeddetector. Single-view should remain unchanged as a control. Audit source code for any other frame-order operations; quantify physical-view aligned order displacement and accuracy separately. Add tests for token expansion/permutation behavior and explicit off/default preservation. It may restore symmetry yet damage pretrainedgeometry, so DO NOT assume it is a deployable fix or select a best order using GT. If causal evidence supports it, decide between TRAIN-only random reference/order augmentation and an explicitly order-invariant per-view2D branch retaining geometry reference separately; no blind addedsteps, no lockedholdout access. Current GPUjobs complete; checkPID beforestarting. All previous reports retained, harddeadline unchanged.',
        production_changed=False)
    report['visual_review_pending']=False;report['visual_review']=decision['visual'];(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004/review_view_order_v1');shutil.copy2(OUT/'results.json',dest/'results.json');shutil.copy2(OUT/'decision.json',dest/'decision.json')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=decision['next'];(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='input_view_order_sensitivity_confirmed',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=decision['next'])
    if 'physical_view_order_audit_and_visual_review' not in state['completed']:state['completed'].append('physical_view_order_audit_and_visual_review')
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
