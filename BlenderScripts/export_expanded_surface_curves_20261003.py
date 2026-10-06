"""Same strict read-only exporter, expanded TRAIN-only candidate protocol."""
import sys
from pathlib import Path
ROOT=Path(sys.argv[-1]);sys.path.insert(0,str(ROOT/'BlenderScripts'))
import export_diverse_surface_curves_20261003 as export
export.OUT=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_gt_v2'
export.PROTOCOL=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_expansion_protocol.json'
export.ROLE_FILTER='train'  # Reuse v1 validation sidecars unchanged; never reselect them.
export.main()
