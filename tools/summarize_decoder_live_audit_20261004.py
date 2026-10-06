"""Matched development-only RGB audits; aggregate counts before ratios."""
import json, datetime as dt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'

def main():
    rows=[]
    names=['live_view_roi_detector_audit_v1','live_view_decoder_uniform2400_audit_v1','live_view_decoder_balanced2400_audit_v1']
    reference=None
    for name in names:
        data=json.loads((CYCLE/name/'results.json').read_text())['reports']
        if reference is None:reference=data
        assert [x['task'] for x in data]==[x['task'] for x in reference]
        assert [x['detector'] for x in data]==[x['detector'] for x in reference], 'Fixed detector must not change'
        for kind in ['real','synthetic']:
            for variant in ['single','paired','all']:
                group=[x for x in data if x['task']['record']['kind']==kind and x['task']['variant']==variant]
                total=sum(x['gated_pose']['total_points'] for x in group)
                row=dict(run=name,kind=kind,variant=variant,images=sum(len(x['task']['record']['views']) for x in group),
                    points=total,tp=sum(x['detector']['tp'] for x in group),gt=sum(x['detector']['gt'] for x in group),
                    predictions=sum(x['detector']['predictions'] for x in group),
                    gated_pck16=sum(x['gated_pose']['pck16_count'] for x in group)/max(total,1),
                    gated_pck32=sum(x['gated_pose']['pck32_count'] for x in group)/max(total,1))
                for metric in ['spatial/all','spatial/anchor','spatial/detected_only']:
                    count=sum(x['metrics'].get(metric,{}).get('n',0) for x in group)
                    row[metric]=sum(x['metrics'].get(metric,{}).get('mean',0)*x['metrics'].get(metric,{}).get('n',0) for x in group)/max(count,1)
                rows.append(row)
    report=dict(rows=rows,detector_identical=True,
        limitations='Seed17; repeated existing development scenes, not external test. Single/paired/all use same anchor but different view counts; spatial/anchor is the matched-view comparison. Conditional point means use GT box association. Gated PCK includes missed detections but does not penalize duplicate false-positive points; detector precision reported separately. No production promotion.')
    path=CYCLE/'decoder_live_comparison_v1.json';path.write_text(json.dumps(report,indent=2)+'\n')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')
    (dest/path.name).write_text(json.dumps(report,indent=2)+'\n')
    decision=dict(
        outcome='Cached gain partly transfers to RGB, but not a universal accuracy gain. No promotion.',
        matched_anchor='Balanced real anchor mean single70.02/paired55.21/all55.94px versus uniform81.31/76.09/69.80px. Single-view weakness persists.',
        caution='Real all-view gatedPCK32 balanced.520 versus original-decoder.549 despite lower mean error. Synthetic paired error19.28 versus uniform18.07. Detection identical, including false positives.',
        inspected=['review_live_decoder_balanced2400_v1/00_real_single.jpg','review_live_decoder_balanced2400_v1/01_real_paired.jpg','review_live_decoder_balanced2400_v1/02_real_all.jpg'],
        visual='Same FR2f584camA: single view has duplicatequeries and feet clustered near face/lower body. Paired balanced shifts hindfeet toward actualpaw but frontfeet remain near face. Allviews do not automatically fix semantic identities.',
        next='Investigate training/inference view-count mismatch: all734 cached source records have exactly2views. Prepare TRAIN-only single-view native-memory variants of the SAME existing images, no new labels or split changes; retain original paired features/control. Initially bound to all complete TRAIN camA plus matched per-session side examples and syntheticsingle/dual retention. A matched pair-only versus single/pair-context augmentation pilot must share initialization, image schedule, bottom exposure, steps, seeds and fixed detector. Recompute encoder features for single-view input: slicing a2view memory is NOT single-view inference. Verify parity on paired path and source hashes before training. Evaluate same fixed1/2/allRGB tasks, conditional and gated metrics. Do not tune on locked sessions. First audit feasible memory/time and image dedup/context identity; no blind moresteps.')
    (CYCLE/'decoder_live_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n')
    (dest/'decoder_live_decision_v1.json').write_text(json.dumps(decision,indent=2)+'\n')
    plan=json.loads((CYCLE/'plan.json').read_text());plan['next_command']=decision['next'];(CYCLE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='decoder_rgb_variable_view_review_complete',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=decision['next'])
    if 'decoder_rgb_matched_context_audit_and_visual_review' not in state['completed']:state['completed'].append('decoder_rgb_matched_context_audit_and_visual_review')
    (CYCLE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    for row in rows:print(json.dumps(row))

if __name__=='__main__':main()
