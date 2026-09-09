# VGGT-Omega Model Intermediates

Each processed frame set is saved to:
```
<output-dir>/intermediates/frame_XXXXXX/
```

Enable with `--save-intermediates` when running `run_pipeline.py`.  
Add `--intermediates-full` to also save full token activations (~130 MB/frame extra).  
Add `--intermediates-no-grads` to skip the backward pass (no GradCAM, no patch importance).

---

## File Reference

### `images.npy`
**Shape:** `[F, 3, H, W]`  float32 in [0, 1]  
The preprocessed input images as fed to the model. F = number of cameras (3). H×W is the model resolution (e.g. 416×624 for 512/balanced). These are the images the model actually saw — not the source resolution.

```python
imgs = np.load("images.npy")           # [3, 3, H, W]
rgb  = np.transpose(imgs[0], (1,2,0))  # cam A as HxWx3
```

---

### `pose_enc.npy`
**Shape:** `[1, F, 9]`  float32  
The raw 9-dimensional pose encoding produced by CameraHead before decoding. Each of the 9 values encodes a mixture of rotation, translation, and focal length in a learned continuous space. Not directly interpretable — use `extrinsics.npy` and `intrinsics.npy` for human-readable matrices.

---

### `extrinsics.npy`
**Shape:** `[1, F, 3, 4]`  float32  
World-to-camera transformation matrices [R | t] decoded from `pose_enc`. Row = camera, columns 0:3 = rotation matrix, column 3 = translation vector. Translation scale is arbitrary (VGGT predicts up-to-scale poses); use `ATE_RMSE_scaled` for comparisons.

```python
ext = np.load("extrinsics.npy")[0]   # [F, 3, 4]
R_camA = ext[0, :3, :3]              # 3×3 rotation
t_camA = ext[0, :3,  3]              # 3-vector translation
```

---

### `intrinsics.npy`
**Shape:** `[1, F, 3, 3]`  float32  
Camera intrinsic matrices (K) at the model input resolution. Diagonal is [fx, fy, 1]; off-diagonal contains cx, cy. These are decoded from the focal-length part of `pose_enc` and are valid at the H×W resolution in `images.npy`, not the source video resolution.

---

### `patch_importance.npy`
**Shape:** `[F, patch_h, patch_w]`  float32  e.g. `[3, 26, 39]`

**What it is:** The L2 norm of `d(pose_enc) / d(patch_embed_output)` at each patch position. This is backpropagation attribution — not a model output. It answers: *"if this image patch were changed slightly, how much would the predicted camera pose change?"*

**How to read it:** Large value = that patch strongly influenced the pose prediction. Small value = model was insensitive to that patch.

**Bright patches vs dark patches:** We compute `quality_score` (in `quality_scores.csv`) as the fraction of total importance that falls on patches with mean brightness ≥ 15/255. Frames where most importance is on dark/featureless patches tend to produce worse calibrations.

```python
imp  = np.load("patch_importance.npy")   # [3, 26, 39]
# cam C importance grid:
plt.imshow(imp[2], cmap="inferno")
```

---

### `gradcam/camA_gradcam.png`, `camB_gradcam.png`, `camC_gradcam.png`
The `patch_importance` map upsampled to image resolution and blended over the original image using the "inferno" colormap (yellow/bright = high importance, dark = low). Power-0.4 gamma is applied to make mid-range values visible.

This is the most human-readable output: it shows *where in the image* the model's pose prediction came from.

---

### `camera_tokens/frame_{block}.npy`
**Shape:** `[F, 17, 1024]`  float32  
Camera and register token embeddings captured at the **output of frame_blocks[block]** (the intra-frame self-attention blocks). Contains the 17 non-patch tokens per camera: index 0 = camera token, indices 1–16 = register tokens.

- `frame_00` = embeddings just after block 0 (very early, mostly local DINOv2-style features)
- `frame_06`, `frame_11`, `frame_17`, `frame_23` = progressively deeper representations

**Camera token (index 0):** This is the single vector that accumulates pose information. Its L2 norm grows from ~1.5 (block 0) to ~160 (block 23) as pose uncertainty is resolved. After CameraHead processes it further, it becomes `pose_enc`.

**Register tokens (indices 1–16):** Act as "scratch space" — they absorb information from patches during self-attention, allowing the camera token to query them in later steps.

```python
tokens = np.load("camera_tokens/frame_23.npy")  # [3, 17, 1024]
cam_token    = tokens[:, 0,  :]   # [3, 1024]  camera tokens for all 3 views
reg_tokens   = tokens[:, 1:, :]   # [3, 16, 1024]  register tokens

# How similar are camA and camC at this depth?
from numpy.linalg import norm
cos = np.dot(cam_token[0], cam_token[2]) / (norm(cam_token[0]) * norm(cam_token[2]))
```

---

### `camera_tokens/inter_{block}.npy`
**Shape:** `[F, 17, 1024]`  float32  
Same as above but captured at the **output of inter_frame_blocks[block]** (the cross-frame attention blocks where cameras exchange information).

The inter-frame blocks come *after* frame_blocks at each depth. Comparing `frame_XX` vs `inter_XX` at the same block shows how much the cross-camera step changed each camera's representation.

**Block 6 is special:** It is a "register" type inter-frame block that only attends among camera+register tokens (not patches). It is the cheapest form of cross-camera communication.

**Cosine similarity of camera tokens across cameras decreases with depth** — from ~0.99 at block 0 (all cameras look similar since they share backbone weights) to ~0.25 at block 23 (cameras have differentiated based on their view). A good calibration frame typically shows *lower* final cosine similarity — the model differentiated the views more confidently.

---

### `activations/frame_{block}.npy`  *(optional, --intermediates-full)*
**Shape:** `[F, N, 1024]`  float32  where N = 17 + patch_h×patch_w (e.g. 1031)

Full token activations at the output of frame_blocks[block] — all tokens including patches. The first 17 are cam+register (same as `camera_tokens/frame_{block}.npy`); indices 17+ are the patch tokens laid out in raster order (row-major, top-left to bottom-right).

Use this to project patch representations to RGB via PCA:
```python
arr = np.load("activations/frame_23.npy")    # [3, 1031, 1024]
patches = arr[0, 17:, :]                     # [1014, 1024]  cam A patch tokens
from sklearn.decomposition import PCA
rgb = PCA(n_components=3, whiten=True).fit_transform(patches)  # [1014, 3]
rgb_grid = rgb.reshape(26, 39, 3)            # spatial layout
```

---

### `activations/inter_{block}.npy`  *(optional, --intermediates-full)*
**Shape:** `[F, N, 1024]`  for global-type blocks  
          `[F, 17, 1024]` for register-type blocks (indices 2, 6, 9, 14, 20)

Same as above but from inter_frame_blocks. Note that for global-type inter-frame blocks the full patch+camera sequence is attended jointly across all cameras, so these activations reflect cross-view information exchange.

---

## Quick Loading Snippet

```python
import numpy as np
from pathlib import Path

frame = Path("vggt_calibration_pipeline/runs/sample_videos/intermediates/frame_000348")

images       = np.load(frame / "images.npy")            # [3, 3, H, W]
extrinsics   = np.load(frame / "extrinsics.npy")[0]     # [3, 3, 4]
intrinsics   = np.load(frame / "intrinsics.npy")[0]     # [3, 3, 3]
importance   = np.load(frame / "patch_importance.npy")  # [3, 26, 39]

# Camera token evolution (norm should grow ~1.5 → ~160)
for blk in ["00", "06", "11", "17", "23"]:
    t = np.load(frame / f"camera_tokens/frame_{blk}.npy")  # [3, 17, 1024]
    norms = np.linalg.norm(t[:, 0, :], axis=-1)
    print(f"block {blk}: cam norms = {norms}")
```

---

## Storage

| Content | Size/frame |
|---------|-----------|
| Compact set (default) | ~11 MB |
| + `--intermediates-full` | ~140 MB |

For 405 frames: compact ≈ 4.5 GB, full ≈ 57 GB.
