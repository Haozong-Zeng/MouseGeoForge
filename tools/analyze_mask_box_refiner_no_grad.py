"""Analyze the versioned no_grad-freeze retry without modifying old analysis."""
import analyze_mask_box_refiner as analysis
from run_depth_pose_ablation import ROOT,write_json,fingerprint


if __name__=='__main__':
    analysis.OUT=ROOT/'output/mask_box_refiner_seed17_v1_no_grad'
    analysis.main()
    write_json(analysis.OUT/'analysis_entrypoint.json',dict(code_sha256=fingerprint(ROOT/'tools/analyze_mask_box_refiner_no_grad.py'),analysis_sha256=fingerprint(ROOT/'tools/analyze_mask_box_refiner.py')))
