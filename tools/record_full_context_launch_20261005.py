"""Persist the larger eligible-context stage and exact extraction gates."""
import datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/core23_spatial_cycle_20261004'

def main():
    gates={}
    for role in ['train','validation']:
        path=CYCLE/f'fullshared_view_{role}_context_v1/manifest.json'
        if path.exists():
            m=json.loads(path.read_text());assert m['full2view_golden_cases']>0 and m['max_full2view_golden_delta']<2e-5
            gates[role]=dict(images=m['images'],two_view_goldens=m['full2view_golden_cases'],max_delta=m['max_full2view_golden_delta'],original_parity=m['original_parity'])
    started=(CYCLE/'full_context_pose_control_v1/status.json').exists()
    next_action=('Actualfullshared extraction chain session63633 uses prepare_reversed_view_context_20261005.py --shared-tokens --eligible-full then same --role validation. Outputs fullshared_view_train_context_v1 andfullshared_view_validation_context_v1, sourceplan eligible_fullview_context_plan_v1. Checkactualstatus/PID/failure andmanifestgates beforetraining; no duplicateGPUjobs. Script/train/review full-context flagsIMPLEMENTED and6samplertests passed. '+('Training ALREADYLAUNCHED: train_view_context_control_20261004.py --full-context-control, outputfull_context_pose_control_v1. DO NOTduplicate. ' if started else 'Training notyetconfirmed launched; onceBOTHmanifestscomplete andGPUlockfree launch train_view_context_control_20261004.py --full-context-control ONCE. ')+
        'Matched800steps3seeds17/29/43, arms paired/full, initialshared_token_pose_control_v1/shared800pose andfixedmixed800ROI detector. Both50percentsingle; pairedarm50percentshared2view, fullarm25percentshared2view25percentfulleligibleshared. Same supervisedimages andbottomexposure. ActualRGBfullcontext re-encoding withruntime firsttoken sharing/restoration, notstackedpairedmemory. TRAIN1223 VAL112 sameimages, labelsroles/sourcehashes exact, eligibleonly; originalgeometry keptseparate. Two-viewfull controls retainoriginalorderedpair topermit tightgoldens againstsharedcache; sourceimage/GT/boxphysicalalignment tested. Initialposegolden againstshared800required. Nativefullcontext canchangefrozendetectorboxes/matching, reportconditional/gatedmetrics andcoverage. TRAIN159scenes5-7views,87synthetic2view; synthetic4/8view stress remainsuntrained. Aftertraining run review_view_context_control_20261004.py --full-context-control ONCE (implemented), inspectsamecamA/camD/dual rawqueries in single/shared/fullcontexts. Compare percontext/camera/reliable/back/dual andtrain/validation againstinitialandpairedcontrol, notjustfullview. FullRGBvalidator trainedfull-context flag NOTYETimplemented; ifjustified addexplicitflagloadingfull_context_pose_control paired/full ckpt, samefrozenmixed800detector andshared2Dprobe preserving originalgeometry. Goldens fromfull_contextpredictions forpaired/single/full; verifyeligiblefullrecord views matchesliveall beforeclaimingparity, syntheticfullcacheonly2doesNOTgoldenmatch4/8liveall. Same60tasks andoriginal/reversed, noGTbestorder. No automaticpromotion/labelchanges/lockedaccess; trainingstop10:18:46UTC hardstop10:48:46UTC unchanged. Earlyhandoffallowed.')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=next_action;(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(next=next_action,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat());entry='eligible_full_context_stage_implemented'
    if entry not in state['completed']:state['completed'].append(entry)
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    (CYCLE/'full_context_launch_decision_v1.json').write_text(json.dumps(dict(gates=gates,training_started=started,next=next_action,production_changed=False),indent=2)+'\n');print(json.dumps(dict(gates=gates,training_started=started)))

if __name__=='__main__':main()
