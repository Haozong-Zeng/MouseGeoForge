"""Retry only failed read-only aggregation, preserving its locked first runner."""
import numpy as np
import audit_cylinder_epipolar_localization as audit

FIRST=audit.OUT
OUT=audit.ROOT/'output/cylinder_epipolar_localization_v1_retry'
ORIGINAL_WRITE=audit.write_json


def summarize_default_temperature(rows):
    if not rows:return dict(points=0)
    # This audit explicitly requested scales=(1,), not a temperature sweep.
    keys=('coarse_px','refined_px','coarse_nme','refined_nme','refinement_delta_px','improvement_px',
          'inside_window','window_oracle_min_px','entropy_normalized','max_probability','logit_range',
          'clipped_window','scale_1_nme')
    return dict(points=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys},
        refinement_improved_fraction=float(np.mean([r['improvement_px']>0 for r in rows])),
        refinement_delta_p95_px=float(np.percentile([r['refinement_delta_px'] for r in rows],95)))


def write_with_provenance(path,value):
    if path==OUT/'protocol.json':
        value['code_sha256']['tools/retry_cylinder_epipolar_localization.py']=audit.fingerprint(audit.ROOT/'tools/retry_cylinder_epipolar_localization.py')
        value['retry_of_protocol_sha256']=audit.fingerprint(FIRST/'protocol.json')
        value['retry_reason']='First 88-record forward pass failed only at aggregation because generic summary expected scale4/8/16 fields. Preserve first runner/protocol; summarize only computed default-temperature fields.'
    ORIGINAL_WRITE(path,value)


def main():
    if not (FIRST/'protocol.json').exists() or (FIRST/'report.json').exists():raise RuntimeError('Expected failed first audit only')
    ORIGINAL_WRITE(FIRST/'status.json',dict(state='failed',stage='aggregation',error="KeyError: scale_4_nme",training_affected=False))
    audit.OUT=OUT;audit.summarize=summarize_default_temperature;audit.write_json=write_with_provenance
    audit.main()


if __name__=='__main__':main()
