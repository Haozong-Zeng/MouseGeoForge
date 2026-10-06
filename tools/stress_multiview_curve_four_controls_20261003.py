"""Reduced-DOF shared3Dcurve control using the same synthetic observations."""
import numpy as np
from scipy.interpolate import BSpline
import stress_multiview_surface_curve_20261003 as base
from stress_multiview_curve_consensus_20261003 import consensus

if __name__=='__main__':
    base.OUT=base.ROOT/'output/surface_curve_cycle_20261003/multiview_curve_four_controls_v1'
    base.KNOTS=np.r_[np.zeros(4),np.ones(4)]
    base.BASIS=BSpline(base.KNOTS,np.eye(4),3)
    base.SECOND=np.diff(np.eye(4),n=2,axis=0)
    base.FITTERS={'four_latent_500eval':lambda obs,cams:base.fit(obs,cams,True,max_nfev=500),
                  'four_consensus_500eval':consensus}
    base.EXTRA_PROTOCOL=dict(reference='multiview_curve_consensus_v1: six controls',
        hypothesis='Large detours under a corrupted camera may exploit six-control flexibility; four-control cubic curve trades shape capacity for stability.',
        same_observations_rng_seed=71,same_training_and_held_views=True,controls=4,
        candidate_max_nfev=150,final_max_nfev=500,threshold_px=6.,
        limitations='Not a learned head comparison. Synthetic virtual endpoints/amodal labels. Four controls can underfit valid complex poses; no real-label adoption.')
    try:base.main()
    except Exception:
        base.OUT.mkdir(exist_ok=True);base.write_json(base.OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
