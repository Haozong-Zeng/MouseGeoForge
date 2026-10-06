"""Four predeclared crop readouts, same association diagnostics as ROI controls."""
import diagnose_prequery_roi_20261005 as audit
audit.SOURCE=audit.CYCLE/'predicted_crop_encoder_v1'
audit.OUT=audit.CYCLE/'predicted_crop_affinity_diagnostic_v1'
if __name__=='__main__':audit.main(['rgb_cls','rgb_patch_mean','gray_cls','gray_patch_mean'])
