# Third-party provenance

## VGGT-Omega

- Upstream: https://github.com/facebookresearch/vggt-omega
- Local upstream HEAD at initial source snapshot: `39a0cb8af88554f15ddcb5354cd52bde588fa014`
- License: [FAIR Noncommercial Research License](vggt-omega/LICENSE)
- Local additions/modifications: instance/keypoint/surface heads, model wiring,
  training and geometry losses, dataset loaders and tests.

The files are versioned as ordinary source in MouseGeoForge, not as an empty
submodule pointer. The original local nested Git repository is preserved on the
author's machine; its history, configuration and credentials are not published.

## Calibration GUI

`gui_based_vggt_o/` retains its existing [license](gui_based_vggt_o/LICENSE) and
source copyright notices. Its bundled model code must not be assumed to have a
different license merely because it is under the GUI directory.

## External dependencies and data

SAM2, original VGGT, 4D-GS, DeepLabCut, Anipose, Orbbec SDK and other downloaded
projects are not vendored wholesale. Install them separately where a workflow
requires them, following the relevant upstream documentation and terms.
DANNCE, SBeA, s-DANNCE, Lightning Pose/CRIM13, source videos, model weights,
Blender assets and literature PDFs are not included. Dataset availability does
not imply permission to redistribute it through this repository.

Before a future public release, review provenance and distribution rights for
all included code and obtain any required permissions. Private repository status
does not replace the applicable upstream license conditions.
