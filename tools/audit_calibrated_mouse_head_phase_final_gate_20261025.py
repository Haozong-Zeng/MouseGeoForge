"""Independent closure of the completed12000 head phase TRAIN gate and exposure."""
from pathlib import Path
import datetime as dt,hashlib,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mousegeoforge.semseg_training import BalancedSchedule,digest,load_items
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025';RUN=OUT/'semseg_head_phase_v1'
def main():
    target=OUT/'head_phase_final_gate_audit_v1.json';assert not target.exists()
    progress=json.loads((RUN/'progress.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text());assert progress['common_endpoint']==12000
    assert protocol['max_head_steps']==12000 and protocol['min_head_steps']==4000 and protocol['training_only_stopping']
    for p,h in protocol['source_sha256'].items():assert digest(ROOT/p)==h
    items=load_items(OLD/'semseg_items.json',ROOT,verify=False);train_ids={r['id'] for r in items if r['split']=='train'};schedule=BalancedSchedule(items,4,17)
    expected=[[items[i]['id'] for i in schedule.at(step+18000)] for step in range(12000)];canonical=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=digest(__file__),source_pins_unchanged=True,protocol_sha256=digest(RUN/'protocol.json'),progress_sha256=digest(RUN/'progress.json'),items_sha256=digest(OLD/'semseg_items.json'),common_endpoint=12000,rule='Same last5 TRAIN span:loss<=.01mean,IoU/BF512tol2<=.002,overallAND4strataBOTHarms',arms={},DEV_used_for_gate=False)
    exposure_hashes={}
    for arm in ['L','M']:
        folder=RUN/f'{arm}_012000';completion=json.loads((folder/'completion.json').read_text());assert completion['frozen_unchanged'] and completion['full_schedule_completed'] and completion['segment_endpoint']==12000
        checkpoint=folder/'step_012000.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False);p=saved['protocol'];assert p['data_sha256']==canonical and p['steps']==12000 and p['head_phase_exposure_offset']==18000 and p['arm']==arm
        assert saved['step']==12000 and len(saved['metrics'])==12000
        assert not any(v for k,v in saved['requires_grad'].items() if k.startswith('encoder.'))
        assert all(k.startswith('head.') for k in saved['model']) and all(k.startswith('head.') for k in saved['ema'])
        actual=[r['local_ids'] for r in saved['metrics']];assert actual==expected and all(set(ids)<=train_ids for ids in actual)
        exposure_hashes[arm]=hashlib.sha256(json.dumps(actual).encode()).hexdigest()
        last=progress['history'][arm][-5:];monitor_hashes={}
        for item in last:
            path=RUN/f'{arm}_{item["step"]:06d}'/f'train_monitor_{item["step"]:06d}.json';rows=json.loads(path.read_text());assert len(rows)==64 and {r['id'] for r in rows}<=train_ids;monitor_hashes[str(path.relative_to(ROOT))]=digest(path)
            for group in item['metrics']:
                selected=rows if group=='all' else [r for r in rows if f'{r["domain"]}/{r["bottom"]}'==group]
                assert len(selected)==(64 if group=='all' else 16)
                for metric in ['loss','iou','boundary_f1']:
                    values=[r['loss'] if metric=='loss' else r['ref512_tol2'][metric] for r in selected]
                    assert abs(statistics.mean(values)-item['metrics'][group][metric])<1e-12
        checks=[]
        for group in last[0]['metrics']:
            for metric in ['loss','iou','boundary_f1']:
                values=[r['metrics'][group][metric] for r in last];span=max(values)-min(values);tolerance=.01*max(1e-8,statistics.mean(values)) if metric=='loss' else .002
                checks.append(dict(group=group,metric=metric,span=span,tolerance=tolerance,passes=span<=tolerance,relative_span=span/max(1e-8,statistics.mean(values)) if metric=='loss' else None))
        logged_global_mismatch=sum(r['global_ids']!=r['local_ids'] for r in saved['metrics'])
        result['arms'][arm]=dict(checkpoint_sha256=digest(checkpoint),completion_sha256=digest(folder/'completion.json'),monitor_sha256=monitor_hashes,last5_steps=[r['step'] for r in last],all_checks=checks,failed_checks=[r for r in checks if not r['passes']],TRAIN_exposure_examples=48000,actual_local_ids_match_registered_schedule=True,all_sampled_ids_are_TRAIN=True,encoder_frozen=True,auxiliary_global_ids_log_mismatch_steps=logged_global_mismatch,logging_note='Legacy global_ids log omits phase exposure offset; actual local_ids used by optimizer match the correct registered offset. Source unchanged.')
        del saved
    assert exposure_hashes['L']==exposure_hashes['M'];result['actual_exposure_sha256']=exposure_hashes
    result['both_pass']=all(not a['failed_checks'] for a in result['arms'].values());assert result['both_pass']==progress['converged']
    result['encoder_unfreeze_admitted']=result['both_pass'];result['admission_conclusion']='TRAIN gate failed at registeredmax; no encoder F/U admitted' if not result['both_pass'] else 'TRAIN gate passed; separate remaining-budget gate still required'
    target.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_head_phase_final_gate_20261025.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'both_pass':result['both_pass'],'failed':{a:v['failed_checks'] for a,v in result['arms'].items()},'actual_exposure_matched':True}))
if __name__=='__main__':main()
