"""Descriptive fixed-endpoint TRAIN stability report; no checkpoint selection."""
from pathlib import Path
import datetime,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';RUN=OUT/'semseg_head_phase_v1'
def main():
    target=OUT/'head_phase_training_stability_v1.json';assert not target.exists();progress=json.loads((RUN/'progress.json').read_text());assert progress['common_endpoint']==12000
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=digest(__file__),final_gate_audit_sha256=digest(OUT/'head_phase_final_gate_audit_v1.json'),arms={},endpoint_selection='Registered full12000 endpoint; no retrospective checkpoint selection',cause='Not determined. Descriptive late loss/metric instability; no claim of convergence or encoder adaptation.')
    for arm in ['L','M']:
        path=RUN/f'{arm}_012000/step_012000.pt';saved=torch.load(path,map_location='cpu',weights_only=False);rows=saved['metrics'];windows=[]
        for start,end in [(0,2000),(2000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,12000)]:
            selected=rows[start:end];windows.append(dict(update_interval=[start,end],mean_training_loss=statistics.mean(r['loss'] for r in selected),max_training_loss=max(r['loss'] for r in selected),mean_grad_norm=statistics.mean(r['grad_norm'] for r in selected),max_grad_norm=max(r['grad_norm'] for r in selected),final_LRs=selected[-1]['lrs']))
        result['arms'][arm]=dict(checkpoint_sha256=digest(path),raw_training_windows=windows,fixed_TRAIN_EMA_monitor_curve=[dict(step=r['step'],**r['metrics']['all']) for r in progress['history'][arm]],all_checkpoint_model_and_EMA_parameters_finite=all(bool(torch.isfinite(v).all()) for k in ['model','ema'] for v in saved[k].values()))
        assert result['arms'][arm]['all_checkpoint_model_and_EMA_parameters_finite'];del saved
    target.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_head_phase_training_stability_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'descriptive_complete','convergence_claim':False}))
if __name__=='__main__':main()
