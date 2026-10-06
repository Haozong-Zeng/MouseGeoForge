"""Same-sample objective comparison; source initial/free baseline retained."""
import numpy as np
import review_diverse_curve_features_20261003 as review
source=review.BASE
review.BASE=review.ROOT/'output/surface_curve_cycle_20261003/curve_objectives_v1'
with np.load(source/'predictions.npz') as z:review.REFERENCE_PREDICTIONS={k:z[k] for k in ['det_free','pose_free']}
review.DISPLAY_ARMS=['pose_free','pose_reduced_order','pose_material_huber']
review.main()
