"""Save VGGT-Omega model intermediates for each processed frame.

Compact set (default, ~3 MB/frame):
  pose_enc.npy                        raw pose encoding
  extrinsics.npy / intrinsics.npy     decoded camera matrices
  camera_tokens/block_{i:02d}.npy     camera+register token embeddings [F, 17, D]
                                      captured at every frame_block and inter_frame_block
  patch_importance.npy                d(pose)/d(patch_embed) norm [F, patch_h, patch_w]
  gradcam/                            GradCAM overlay PNGs

Full set (--intermediates-full, adds ~130 MB/frame):
  activations/frame_tokens_{i:02d}.npy    all tokens [F, N, D] at every frame block
  activations/inter_tokens_{i:02d}.npy    all tokens [F, N, D] at every inter block
"""
from __future__ import annotations

import types
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# Capture every block — index range determined at runtime from model architecture
CAPTURE_BLOCKS = None  # None = capture all available blocks
PATCH_TOKEN_START = 17


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _register_act_hook(block: nn.Module, store: dict, key: str) -> torch.utils.hooks.RemovableHook:
    def _hook(module, inp, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        store[key] = out.detach().float().cpu().numpy()
    return block.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_frame_intermediates(
    model: nn.Module,
    images_tensor: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    save_gradients: bool = True,
    save_full_activations: bool = False,
    capture_blocks: tuple[int, ...] | None = CAPTURE_BLOCKS,
) -> None:
    """Run one frame set through the model and save intermediates to output_dir.

    images_tensor: [F, 3, H, W] on device — one set of camera views.
    save_gradients: compute d(pose)/d(patch_embed) importance maps (adds ~1s/frame).
    save_full_activations: also save all-token activations [F, N, D] at every block
                           (large — ~130 MB/frame extra).
    capture_blocks: which block indices to capture. None = all available blocks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    F, _, H, W = images_tensor.shape
    patch_h, patch_w = H // 16, W // 16
    agg = model.aggregator

    n_frame_blocks = len(agg.frame_blocks)
    n_inter_blocks = len(agg.inter_frame_blocks)

    if capture_blocks is None:
        blocks_to_capture = tuple(range(max(n_frame_blocks, n_inter_blocks)))
    else:
        blocks_to_capture = capture_blocks

    act_store: dict = {}
    handles: list = []

    for i in blocks_to_capture:
        if i < n_frame_blocks:
            handles.append(_register_act_hook(agg.frame_blocks[i], act_store, f"frame_{i:02d}"))
        if i < n_inter_blocks:
            handles.append(_register_act_hook(agg.inter_frame_blocks[i], act_store, f"inter_{i:02d}"))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    # --- forward pass ---
    with amp_ctx, torch.inference_mode():
        predictions = model(images_tensor)

    for h in handles:
        h.remove()
    handles.clear()

    # --- save pose outputs ---
    from vggt_omega.utils.pose_enc import encoding_to_camera
    pose_enc = predictions["pose_enc"].detach().float().cpu().numpy()
    ext_t, intr_t = encoding_to_camera(predictions["pose_enc"], (H, W))
    np.save(output_dir / "pose_enc.npy", pose_enc)
    np.save(output_dir / "extrinsics.npy", ext_t.float().cpu().numpy())
    np.save(output_dir / "intrinsics.npy", intr_t.float().cpu().numpy())

    # --- save images ---
    np.save(output_dir / "images.npy", images_tensor.detach().float().cpu().numpy())

    # --- camera+register token embeddings at each captured block ---
    cam_dir = output_dir / "camera_tokens"
    cam_dir.mkdir(exist_ok=True)
    for key, arr in act_store.items():
        if key.startswith("frame_"):
            cam_tokens = arr[:, :PATCH_TOKEN_START, :]   # [F, 17, D]
        else:
            tokens_total = arr.shape[1]
            tokens_per_frame = tokens_total // F
            cam_tokens = np.stack(
                [arr[0, f * tokens_per_frame : f * tokens_per_frame + PATCH_TOKEN_START, :]
                 for f in range(F)]
            )  # [F, 17, D]
        np.save(cam_dir / f"{key}.npy", cam_tokens)

    # --- full token activations (optional, large) ---
    if save_full_activations:
        act_dir = output_dir / "activations"
        act_dir.mkdir(exist_ok=True)
        for key, arr in act_store.items():
            np.save(act_dir / f"{key}.npy", arr)

    del act_store

    # --- patch importance + GradCAM (requires gradient pass) ---
    if save_gradients:
        pe_store: dict = {}

        def _pe_hook(module, inp, output):
            t = output["x_norm_patchtokens"] if isinstance(output, dict) else output
            t.retain_grad()
            pe_store["out"] = t

        pe_handle = agg.patch_embed.register_forward_hook(_pe_hook)

        with amp_ctx, torch.enable_grad():
            preds2 = model(images_tensor)
            pose_enc2 = preds2["pose_enc"]
            img_hw2 = preds2["images"].shape[-2:]
            ext2, intr2 = encoding_to_camera(pose_enc2, img_hw2)
            scalar = ext2.float().pow(2).sum() + intr2.float().pow(2).sum()
            scalar.backward()

        pe_handle.remove()

        pe_out = pe_store.get("out")
        if pe_out is not None and pe_out.grad is not None:
            imp = pe_out.grad.detach().float().norm(dim=-1)  # [F, num_patches]
            imp_grid = imp.cpu().numpy().reshape(F, patch_h, patch_w)
            np.save(output_dir / "patch_importance.npy", imp_grid)
            _save_gradcam_pngs(imp_grid, images_tensor, patch_h, patch_w, F, output_dir)

        del pe_store, preds2, scalar
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GradCAM PNG export
# ---------------------------------------------------------------------------

def _save_gradcam_pngs(
    imp_grid: np.ndarray,
    images_tensor: torch.Tensor,
    patch_h: int,
    patch_w: int,
    num_frames: int,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
    except ImportError:
        return

    H, W = patch_h * 16, patch_w * 16
    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    imgs_np = images_tensor.detach().float().cpu().numpy()  # [F, 3, H, W], [0,1]
    imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))           # [F, H, W, 3]
    imgs_u8 = (np.clip(imgs_np, 0, 1) * 255).astype(np.uint8)

    def _upsample(hm: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(hm[None, None]).float()
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t[0, 0].numpy()

    cmap = plt.get_cmap("inferno")

    for f in range(num_frames):
        raw = imp_grid[f]
        vmin = float(np.percentile(raw, 5))
        vmax = float(np.percentile(raw, 99))
        norm = np.clip((raw - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0) ** 0.4
        heat = _upsample(norm)

        colour = (cmap(heat)[:, :, :3] * 255).astype(np.float32)
        alpha  = (heat * 0.80)[:, :, None]
        blend  = (alpha * colour + (1 - alpha) * imgs_u8[f].astype(np.float32)).clip(0, 255).astype(np.uint8)

        cam_label = chr(ord("A") + f)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        ax.imshow(blend)
        ax.axis("off")
        ax.set_title(f"Camera {cam_label} — pose GradCAM", fontsize=10)
        fig.tight_layout()
        fig.savefig(gradcam_dir / f"cam{cam_label}_gradcam.png", bbox_inches="tight")
        plt.close(fig)

    np.save(out_dir / "patch_importance.npy", imp_grid)
