# V5 canonical BlenderPyMouse export (2026-10-25)

The user requested that the rendered V5 data use the naming and file/directory
contract of `dataset/BlenderPyMouse`, rather than leaving the render-only v3 layout
in `output`. The 768-scene render is converted without another RGB render.

## Canonical datasets

All six datasets are direct children of `dataset/BlenderPyMouse/`. Each has direct
`frame_XXXXXX` children and the existing `blenderpymouse_core23_v1` metadata format.
Splits are separate dataset roots, so Train roots contain no DEV/sealed frames.

| Dataset folder | Scenes | Views |
| --- | ---: | ---: |
| `PyMouseStudio-BrownFur-ClimbWall-Core23-ComplexBackgroundV5-Train` | 288 | 864 |
| `PyMouseStudio-BrownFur-ClimbWall-Core23-ComplexBackgroundV5-Dev` | 72 | 216 |
| `PyMouseStudio-BrownFur-ClimbWall-Core23-ComplexBackgroundV5-Sealed` | 24 | 72 |
| `PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200-ComplexBackgroundV5-Train` | 288 | 864 |
| `PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200-ComplexBackgroundV5-Dev` | 72 | 216 |
| `PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200-ComplexBackgroundV5-Sealed` | 24 | 72 |

Names retain the corresponding source dataset prefix and add the version and
split suffixes. `Sample10-First200` identifies the source pose window; the exported
subset is frames49/97/145 crossed with fixed rigs/environments, not200 new poses.

```text
<dataset>/
  metadata.json
  scene_index.jsonl
  temporal_index.jsonl
  surface_keypoint_definition.json
  dataset_validation.json
  data_file_pins.json
  completion.json
  provenance/                        # source replay, scripts and split-filtered design
  frame_000001/
    blended_images/00000000.jpg       # views 00000000,00000001,00000002
    blended_images/00000000_masked.jpg
    masks/00000000.png
    rendered_depth_maps/00000000.pfm
    rendered_depth_maps/00000000_preview.png
    cams/00000000_cam.txt
    cams/pair.txt
    annotations/00000000.json
    scene_meta.json
    keypoints.json
    keypoints.npz
    keypoint_validity.npz
    keypoint_occlusion.npz
    provenance/                      # original lossless RGB/truth/record + scene parameters
```

`frame_XXXXXX` is a unique export scene ID within a dataset. The actual source
animation frame, rig, environment, split and DEV primary/attribution condition
remain explicit in both scene metadata and the indexes. These scene IDs are not
continuous video frames; `temporal_eligible=false`, every transition is false,
and no synthetic timing/synchronization claim is introduced. Train/val/sealed
spellings in machine metadata remain compatible with the original render design.

## Geometry and annotation semantics

- JPEG95/4:4:4 satisfies the established image layout. Every original PNG remains
  byte-identical in the frame's provenance directory; annotations expose its
  `image_lossless` path. There is no resizing, crop, flip or new RGB rendering.
- PFM is an exact float32 conversion of the primary-ray camera-Z array, little
  endian and bottom-row-first on disk. Body masks are exact binary0/255 copies of
  the visible skin mask. Hair and transparent glass are excluded from truth.
  These masks are not fur silhouettes, dilated masks, or amodal labels.
- Camera files use the existing extrinsic/intrinsic/depth-range text format.
  K/E are preserved exactly. This render's K already uses integer array-index
  pixel centres (`cx=cy=191.5` at384px); do not shift it by another half pixel.
- Core23 has the established23 channels, bones and semantic names, including the
  v3 fixed REST barycentric surface anchors at06/07/23. New world coordinates and
  support vertices are replayed from the exact pinned source files. Existing
  reference frames are checked to<.002mm; ClimbWall97/145 are newly evaluated from
  the same pinned source/definitions after verifying frame49, because those
  frames are absent from the old91-frame dataset.
- Every new camera receives newly projected UVs and new visibility, never copied
  old-camera UVs/visibility. Legacy support visibility uses the current depth and
  body mask; surface anchors use body centre rays plus scene-depth consistency.
  Conflicts remain unknown through `visibility_valid=false`. Occlusion causes
  are0 visible,1 self, or-1 unknown; apparatus is never called another mouse.
- Both the existing metadata completeness helper and Core23 reprojection audit,
  as well as the actual training validity/occlusion reader, validate the result.
  Depth, masks and K/E are independently compared to every original view;
  original PNG/truth copies are checked against their SHA256 pins.

The full render remains a frozen source archive at
`output/bottom_geometry_v5_complex_20261024/`. Its additive
`blenderpymouse_export.json` points to the canonical dataset roots. Old datasets
and original render payloads are not modified. No source `.blend` is saved.

The conversion receipt is under
`dataset/BlenderPyMouse/_annotation_audit/v5_complex_background_20261025/`.
Train provenance includes only that dataset's render design/records, not sealed
camera labels from the full manifest. The datasets are self-contained for loading
and training; the original output directory and source blends are not required.
Rendering/conversion authorization does not authorize sealed model scoring.

Implementation: [source replay](../BlenderScripts/extract_v5_core23_for_blenderpymouse_20261025.py),
[converter/validator](../tools/convert_v5_to_blenderpymouse_20261025.py),
[tests](../tests/test_v5_blenderpymouse_conversion_20261025.py).
