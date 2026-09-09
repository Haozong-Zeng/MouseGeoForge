# BlenderScripts

This directory contains the reusable Blender Python source code used to build,
annotate, render, and validate MouseGeoForge datasets. Large `.blend` projects,
rendered previews, experiment snapshots, vendored Python packages, and historical
reports are intentionally not stored here.

Production `.blend` files live in `../BlenderPyMouseProjects/`. Historical
artifacts live in `../backup/` and are not required for the active
BlenderPyMouse pipelines.

Legacy migration utilities may accept archived `.blend` files or render pairs as
explicit inputs, but the reusable Python source itself has no vendored dependency
on `backup/`.

Most scripts are intended to run through Blender:

```powershell
blender.exe project.blend --background --python BlenderScripts/script_name.py -- [arguments]
```

Scripts import shared helpers directly from this directory. Keep the directory
flat unless their imports are updated at the same time.

`convert_dannce_mouse_to_blenderpymouse.py` is the exception: it runs in the
project Python environment (OpenCV + SciPy), converts the official segmented
DANNCE videos and MATLAB calibration/labels, and writes distortion-corrected
BlenderPyMouse-style real multiview scenes.

`mjx_forelimb_prior.py` optionally uses `h5py`. Install it in Blender's Python
environment or set `MOUSEGEOFORGE_BLENDER_PYTHON_DEPS` to an external dependency
directory. Vendored binary packages are excluded from the source tree.
