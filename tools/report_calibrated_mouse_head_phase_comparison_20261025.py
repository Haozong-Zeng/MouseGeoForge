"""Descriptive preregistered-parent to fixed phase-endpoint native comparison."""
from pathlib import Path
import datetime,json,statistics,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'head_phase_native_comparison_v1.json';assert not target.exists()
    before=json.loads((OUT/'native_metric_audit_v1.json').read_text());registration=json.loads((OUT/'head_phase_registration_v1.json').read_text());audit=json.loads((OUT/'head_phase_final_gate_audit_v1.json').read_text());result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),parent_registration_sha256=sha(OUT/'head_phase_registration_v1.json'),before_native_audit_sha256=sha(OUT/'native_metric_audit_v1.json'),gate_audit_sha256=sha(OUT/'head_phase_final_gate_audit_v1.json'),TRAIN_gate_passed=audit['both_pass'],encoder_F_U_admitted=False,arms={},claim='Descriptive before/after fixed18000 parent to additional fixed12000 phase, same879DEV images; no causal LR claim, independent replication or checkpoint selection',primary_native_GT='Original native labels, historical nearest-mask inference unchanged for both endpoints',existing_depth_ROI_masks='Fixed original M18000 generator retained by preregistered depth/ROI protocols; not retrospectively selected based on this comparison')
    for arm in ['L','M']:
        path=OUT/f'head_phase_{arm}_native_audit_v1.json';after=json.loads(path.read_text());assert after['denominator']==before['denominator']==879
        assert registration['parents'][arm]['sha256']==before['arms'][arm]['checkpoint_sha256']
        assert after['checkpoint_sha256']==audit['arms'][arm]['checkpoint_sha256']
        assert after['evaluation_receipt_sha256']==sha(OUT/f'semseg_head_phase_v1/{arm}_dev/evaluation.json')
        a={r['id']:r for r in before['arms'][arm]['rows']};b={r['id']:r for r in after['rows']};assert len(a)==len(b)==879 and a.keys()==b.keys()
        paired=[]
        for id,r in a.items():
            t=b[id];assert all(r[k]==t[k] for k in ['domain','bottom','group'])
            paired.append(dict(id=id,domain=r['domain'],bottom=r['bottom'],group=r['group'],delta={key:t['native_original'][key]-r['native_original'][key] for key in ['iou','dice','boundary_f1']}))
        groups={}
        for name,rows in [('all',paired),*[(f'{domain}/{bottom}',[r for r in paired if r['domain']==domain and r['bottom']==bottom]) for domain in ['real','synthetic'] for bottom in [True,False]]]:
            groups[name]=dict(images=len(rows),parent_mean_native_iou=before['arms'][arm]['groups'][name]['native_original_iou'],phase_mean_native_iou=after['groups'][name]['native_original_iou'],mean_phase_minus_parent={key:statistics.mean(r['delta'][key] for r in rows) for key in ['iou','dice','boundary_f1']},IoU_improved_images=sum(r['delta']['iou']>1e-12 for r in rows),IoU_degraded_images=sum(r['delta']['iou'] < -1e-12 for r in rows),IoU_unchanged_images=sum(abs(r['delta']['iou'])<=1e-12 for r in rows))
        result['arms'][arm]=dict(parent_checkpoint_sha256=before['arms'][arm]['checkpoint_sha256'],phase_checkpoint_sha256=after['checkpoint_sha256'],phase_native_audit_sha256=sha(path),groups=groups)
    target.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_head_phase_native_comparison_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({a:v['groups']['all'] for a,v in result['arms'].items()}))
if __name__=='__main__':main()
