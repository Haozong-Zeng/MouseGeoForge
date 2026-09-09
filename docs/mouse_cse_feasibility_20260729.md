# Mouse Dense Surface Embedding Feasibility

Date: 2026-07-29

## Decision

Use a hybrid representation:

1. A category-level continuous surface embedding (CSE) head predicts a
   canonical mouse-mesh descriptor for every foreground pixel.
2. Dataset-specific sparse labels supervise points or uncertain surface
   regions in that same canonical space.
3. Keep a small task head for forelimb, palm, wrist, and finger joints.

Replacing all sparse keypoints with CSE is not recommended. Dense surface
correspondence is a better common interface for heterogeneous mouse datasets,
but paws, digits, ears, and other thin or frequently occluded structures still
benefit from explicit local landmark supervision.

## Literature Evidence

- Continuous Surface Embeddings (CSE) assign image pixels and canonical mesh
  vertices to a shared embedding space. The method was explicitly extended
  beyond humans to animal categories with 3D models:
  <https://papers.nips.cc/paper/2020/hash/c81e728d9d4c2f636f067f89cc14862c-Abstract.html>
- Animal Avatars uses CSE descriptors as stronger dense constraints than
  sparse keypoints for canonical animal-mesh fitting. It still retains a
  sparse keypoint loss because CSE predictions are less reliable for paws and
  thin structures:
  <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10197.pdf>
- Existing mouse datasets do not share one landmark ontology. Examples:
  - Mouse Pose Analysis Dataset has a comparatively rich body schema:
    <https://google.github.io/mouse-pose-analysis-dataset/>
  - MARS uses different top/front-view definitions:
    <https://data.caltech.edu/records/j1ww1-mdc55>
  - A large-scale mouse dataset uses only five landmarks:
    <https://www.mdpi.com/2073-8994/14/5/875>

These differences make a canonical mesh surface a useful intermediate
ontology. A paper's "forepaw" can map to a broad mesh region, while the current
Water Reaching "IndexFinger3_R" can map to a narrow vertex ring.

## Local Mesh Audit

- Canonical object: `MS_Mouse`
- Vertices: 9,984
- Edges: 19,982
- Triangles after triangulation: 19,964
- Existing task landmarks: 29
- Vertices currently covered by landmark support rings: 464

The existing sparse map therefore supervises only a small portion of the
available mouse surface.

## Pilot Test

Scenes:

`PyMouseStudio-BlackFur-WaterReaching-FrameToFrame-FGcamB-ForelimbFurFix`,
frames 159 and 197, CameraA/B/C.

Procedure:

1. Export the canonical and posed `MS_Mouse` topology from Blender.
2. Compute a 16D normalized graph-Laplacian descriptor for every canonical
   vertex.
3. Back-project body-mask pixels with rendered depth.
4. Associate each back-projected point with the nearest posed mesh vertex.
5. Recover all visible existing landmark groups by querying the one dense
   surface map.

Results:

| Frame | Representation | Mean px | Median px | P90 px |
| ---: | --- | ---: | ---: | ---: |
| 159 | 16D spectral CSE | 4.727 | 3.530 | 9.201 |
| 159 | Normalized canonical XYZ | 5.611 | 3.151 | 11.173 |
| 197 | 16D spectral CSE | 5.291 | 4.005 | 11.017 |
| 197 | Normalized canonical XYZ | 6.183 | 4.013 | 12.634 |

Across 154 visible landmark queries in both frames, the count-weighted mean is
5.005 px for spectral CSE and 5.893 px for canonical XYZ.

Additional observations:

- Mean three-view union mesh coverage: 46.71%.
- Fine finger landmark mean error: 4.85 px for spectral CSE.
- Coarse-anchor mean error is 9.60 px, but only seven visible samples entered
  that subset, so this split is not statistically useful yet.
- Left/right paws remain visibly separated in the descriptor rendering.

This is a representation and data-interface test, not a learned real-image
benchmark. The current correspondence labels use rendered depth plus a
nearest-vertex approximation. Final training labels should be rendered as
exact triangle IDs plus barycentric coordinates.

## Implemented Prototype

- Blender canonical/posed mesh export:
  `blender/export_mouse_cse_pilot_mesh.py`
- Three-view dense-correspondence pilot:
  `tools/test_mouse_cse_pilot.py`
- Optional VGGT surface head:
  `vggt-omega/vggt_omega/models/heads/surface_embedding_head.py`
- Dense and heterogeneous sparse-region losses:
  `vggt-omega/training/surface_embedding_losses.py`
- Head/loss tests:
  `vggt-omega/training/test_surface_embedding_head.py`

At 624x416 with three views, the 16D surface embedding plus foreground output
uses 3.16 MiB in FP32. A 29-channel keypoint heatmap at the same output stride
uses about 5.39 MiB. The prototype head has 2.90M parameters.

## Proposed Dataset Contract

Synthetic dense labels:

- `face_id`: int32 at output stride 4, `-1` for no mouse surface.
- `barycentric_uv`: two float16 values per valid pixel.
- `body_mask`: uint8.
- `canonical_mesh_id` and topology hash in metadata.

Do not store a 16D target tensor for every image. Reconstruct the target
descriptor during loading from face IDs, barycentric coordinates, and the
small canonical vertex codebook.

Real sparse labels:

- A registry maps each source dataset and landmark name to canonical support
  vertices, weights, uncertainty, and visibility semantics.
- Precise landmarks use narrow support rings.
- Ambiguous landmarks such as "paw" use a weighted surface region and a
  soft-min descriptor loss.
- Missing labels remain unsupervised; they are not treated as background.

## Training Plan

Stage A:

- Freeze the VGGT aggregator.
- Train the surface decoder and foreground branch on Blender dense labels.
- Retain the current camera/depth objectives.

Stage B:

- Mix real masks and heterogeneous sparse landmark datasets.
- Use sparse surface-region supervision on real frames.
- Add left/right part classification and multiview descriptor consistency to
  reduce symmetry swaps.

Stage C:

- Add or retain a compact Water Reaching hand head for palm, wrist, elbow,
  shoulder, and finger joints.
- Query CSE to initialize the task landmarks, then refine them with local
  heatmaps and fixed-rig probabilistic triangulation.

## Go/No-Go Criteria

Proceed to a short learned experiment if:

- Exact Blender raster labels pass visual checks in all five cameras.
- Surface query median error remains below 6 px on held-out synthetic frames.
- Left/right swap rate is below 1% on visible forepaws.
- Adding real sparse-region supervision does not degrade the existing palm
  fixed-rig 3D metric.

Do not replace the hand keypoint head unless a learned CSE model matches its
finger and palm 2D/3D errors under occlusion.
