#!/usr/bin/env python3
"""Build slide-ready figures from existing MouseGeoForge evaluation artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/haozong/projects/MouseGeoForge")
OUT = ROOT / "output" / "ppt_figures_vggt_keypoints_20260728"
SOURCE_OUT = OUT / "source_assets"

W, H = 2560, 1440
BG = "#F8FAFC"
INK = "#172033"
MUTED = "#5D6878"
GRID = "#DCE2EA"
BLUE = "#1677B8"
RED = "#D94841"
GREEN = "#1B9E77"
AMBER = "#D98B19"
PURPLE = "#7656A6"
LIGHT_BLUE = "#EAF4FB"
LIGHT_GREEN = "#EAF7F2"
LIGHT_RED = "#FBEFEE"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

KP_DIR = (
    ROOT
    / "output"
    / "blenderpymouse_vggt_omega_all"
    / "run_20260728_keypoint_head_all29_frozen_624x416_e5"
    / "evaluation_best"
)
GEOM_DIR = ROOT / "output" / "vggt_geometryaligned6000_visualizations"
REPORT_DIR = (
    ROOT
    / "output"
    / "pdf"
    / "vggt_geometryaligned6000_four_method_comparison"
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    return image, ImageDraw.Draw(image)


def draw_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    section: str,
) -> None:
    draw.text((120, 62), section.upper(), fill=BLUE, font=font(27, True))
    draw.rectangle((120, 104, 210, 112), fill=BLUE)
    draw.text((120, 138), title, fill=INK, font=font(56, True))
    draw.text((120, 215), subtitle, fill=MUTED, font=font(28))


def place_contain(
    base: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
    background: str = "#FFFFFF",
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    source = source.convert("RGB")
    scale = min(bw / source.width, bh / source.height)
    nw = max(1, int(source.width * scale))
    nh = max(1, int(source.height * scale))
    resized = source.resize((nw, nh), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (bw, bh), background)
    px, py = (bw - nw) // 2, (bh - nh) // 2
    panel.paste(resized, (px, py))
    base.paste(panel, (x0, y0))
    return x0 + px, y0 + py, x0 + px + nw, y0 + py + nh


def place_cover(
    base: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    source = source.convert("RGB")
    scale = max(bw / source.width, bh / source.height)
    nw = max(1, int(source.width * scale))
    nh = max(1, int(source.height * scale))
    resized = source.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - bw) // 2)
    top = max(0, (nh - bh) // 2)
    base.paste(resized.crop((left, top, left + bw, top + bh)), (x0, y0))


def draw_metric(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    value: str,
    label: str,
    color: str = BLUE,
) -> None:
    draw.text((x, y), value, fill=color, font=font(46, True))
    draw.text((x, y + 58), label, fill=MUTED, font=font(23))


def panel_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    color: str = INK,
) -> None:
    draw.text((x, y), label, fill=color, font=font(27, True))


def save(image: Image.Image, name: str) -> None:
    image.save(OUT / name, quality=96, dpi=(180, 180))


def make_real_keypoints() -> None:
    image, draw = canvas()
    draw_header(
        draw,
        "Multiview keypoint prediction on real video",
        "A frozen VGGT-Omega backbone with a 29-channel heatmap head",
        "Keypoint prediction",
    )

    left, gap, top = 120, 34, 310
    panel_w, panel_h = 750, 760
    for view in range(3):
        x0 = left + view * (panel_w + gap)
        draw.rectangle(
            (x0, top, x0 + panel_w, top + panel_h),
            fill="#050607",
            outline=GRID,
            width=2,
        )
        place_contain(
            image,
            Image.open(KP_DIR / f"real_batch000_view{view}.png"),
            (x0, top, x0 + panel_w, top + panel_h),
            "#050607",
        )
        draw.rectangle((x0, top, x0 + 132, top + 48), fill="#FFFFFF")
        panel_label(draw, x0 + 16, top + 8, f"Camera {chr(65 + view)}")

    draw.line((120, 1120, 2440, 1120), fill=GRID, width=2)
    draw_metric(draw, 150, 1160, "4.95 px", "overall 2D validation error")
    draw_metric(draw, 700, 1160, "6.07 px", "finger keypoints", RED)
    draw_metric(draw, 1190, 1160, "4.72 px", "palm centers", GREEN)
    draw_metric(draw, 1660, 1160, "4.37 px", "head keypoint", PURPLE)
    draw.ellipse((2185, 1190, 2205, 1210), outline=GREEN, width=5)
    draw.text((2220, 1182), "GT", fill=MUTED, font=font(23))
    draw.ellipse((2280, 1190, 2300, 1210), outline=RED, width=5)
    draw.text((2315, 1182), "Prediction", fill=MUTED, font=font(23))
    save(image, "01_keypoint_prediction_real.png")


def make_sim2real_keypoints() -> None:
    image, draw = canvas()
    draw_header(
        draw,
        "Synthetic supervision transfers to real imagery",
        "The same head localizes fine finger structure despite lighting and appearance shift",
        "Sim-to-real",
    )

    left, gap = 120, 32
    panel_w, panel_h = 750, 452
    for row, prefix in enumerate(("synthetic", "real")):
        y0 = 332 + row * 500
        draw.text(
            (18, y0 + 185),
            "SYNTH" if row == 0 else "REAL",
            fill=BLUE if row == 0 else INK,
            font=font(22, True),
        )
        for view in range(3):
            x0 = left + view * (panel_w + gap)
            draw.rectangle(
                (x0, y0, x0 + panel_w, y0 + panel_h),
                fill="#050607",
                outline=GRID,
                width=2,
            )
            place_cover(
                image,
                Image.open(KP_DIR / f"{prefix}_batch000_view{view}.png"),
                (x0, y0, x0 + panel_w, y0 + panel_h),
            )
            if row == 0:
                panel_label(draw, x0 + 12, y0 + 10, f"Camera {chr(65 + view)}", "#FFFFFF")

    draw.rectangle((120, 1315, 2440, 1322), fill=GRID)
    draw.text(
        (120, 1340),
        "Synthetic finger error: 1.33 px",
        fill=BLUE,
        font=font(25, True),
    )
    draw.text(
        (650, 1340),
        "Real finger error: 6.07 px",
        fill=RED,
        font=font(25, True),
    )
    draw.text(
        (1190, 1340),
        "All 29 landmarks trained jointly",
        fill=MUTED,
        font=font(25),
    )
    save(image, "02_keypoint_sim2real.png")


def make_dense_reconstruction() -> None:
    image, draw = canvas()
    draw_header(
        draw,
        "Dense 3D reconstruction in a shared world frame",
        "Five-view Blender control: literal depth and camera ground truth",
        "3D reconstruction",
    )
    source = Image.open(
        GEOM_DIR / "blender_frame_000144" / "pointcloud_comparison.png"
    )
    source = source.crop((0, 125, source.width, source.height))
    place_contain(image, source, (80, 280, 2480, 1190), "#FFFFFF")

    draw.rectangle((90, 1220, 2470, 1390), fill="#FFFFFF", outline=GRID, width=2)
    draw_metric(draw, 145, 1240, "0.216 -> 0.151", "depth AbsRel", BLUE)
    draw_metric(draw, 760, 1240, "10.64 -> 9.75", "camera-center RMSE", GREEN)
    draw_metric(draw, 1420, 1240, "5.27° -> 3.29°", "median rotation error", PURPLE)
    draw.text(
        (2110, 1252),
        "Original",
        fill=MUTED,
        font=font(24),
    )
    draw.text(
        (2110, 1295),
        "Finetuned",
        fill=BLUE,
        font=font(24, True),
    )
    draw.text(
        (2110, 1340),
        "lower is better",
        fill=MUTED,
        font=font(20),
    )
    save(image, "03_dense_3d_reconstruction.png")


def make_camera_pose() -> None:
    image, draw = canvas()
    draw_header(
        draw,
        "Camera pose estimation after shared-rig alignment",
        "Ground truth, original VGGT-Omega, and geometry-coupled finetuning",
        "Camera geometry",
    )
    source = Image.open(
        GEOM_DIR / "blender_frame_000144" / "camera_pose_comparison.png"
    )
    source = source.crop((0, 105, source.width, source.height))
    place_contain(image, source, (90, 290, 2470, 1135), "#FFFFFF")

    draw.rectangle((120, 1175, 2440, 1385), fill="#FFFFFF", outline=GRID, width=2)
    draw.text((160, 1200), "Synthetic 5-camera control", fill=INK, font=font(27, True))
    draw_metric(draw, 160, 1245, "-8.4%", "camera-center RMSE", GREEN)
    draw_metric(draw, 670, 1245, "-37.6%", "median rotation error", PURPLE)
    draw.line((1210, 1200, 1210, 1360), fill=GRID, width=2)
    draw.text((1260, 1200), "Real-video fixed-rig diagnostic", fill=INK, font=font(27, True))
    draw.text(
        (1260, 1252),
        "Relative camera geometry remains the dominant error source.",
        fill=RED,
        font=font(29, True),
    )
    draw.text(
        (1260, 1305),
        "Use calibrated Anipose cameras for metric 3D reconstruction.",
        fill=MUTED,
        font=font(24),
    )
    save(image, "04_camera_pose_estimation.png")


def load_overall_metrics() -> list[dict[str, str]]:
    with (REPORT_DIR / "overall_metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def make_scoring_chart() -> None:
    rows = load_overall_metrics()
    method_order = [
        "Calib-Aniposelib",
        "Calib-Aniposelib-fixedrel",
        "Depth-fixedrel-pairwise",
        "Depth-per-frame",
    ]
    short_names = [
        "Calibration ->\nAniposelib",
        "Calibration -> Aniposelib\n(fixed relative rig)",
        "Depth + fixed rig\n(pairwise)",
        "Depth\n(per frame)",
    ]
    original = []
    tuned = []
    for method in method_order:
        original.append(
            float(
                next(
                    r["median_mm"]
                    for r in rows
                    if r["method"] == method and r["model"] == "Original"
                )
            )
        )
        tuned.append(
            float(
                next(
                    r["median_mm"]
                    for r in rows
                    if r["method"] == method and r["model"] == "Aligned6000"
                )
            )
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=BG)
    ax = fig.add_axes([0.08, 0.19, 0.88, 0.62], facecolor=BG)
    x = np.arange(len(method_order))
    width = 0.32
    b1 = ax.bar(x - width / 2, original, width, color="#718091", label="Original")
    b2 = ax.bar(x + width / 2, tuned, width, color=BLUE, label="Finetuned")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Median 3D error (mm)", fontsize=18, fontweight="bold")
    ax.set_xticks(x, short_names, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=16, loc="upper left")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.07,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
    improved = [20, 20, 21, 22]
    for i, (o, t) in enumerate(zip(original, tuned)):
        pct = (o - t) / o * 100
        ax.text(
            i,
            max(o, t) + 0.42,
            f"-{pct:.1f}%  |  {improved[i]}/22 trials improved",
            ha="center",
            va="bottom",
            fontsize=13,
            color=GREEN,
            fontweight="bold",
        )

    fig.text(0.047, 0.945, "SCORING", fontsize=16, color=BLUE, fontweight="bold")
    fig.text(
        0.047,
        0.885,
        "Finetuning improves all four reconstruction routes",
        fontsize=30,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.047,
        0.842,
        "22 real trials, all frames, one shared Sim(3) alignment per video",
        fontsize=16,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.055,
        "Best: 1.27 mm median error with fixed relative calibration",
        fontsize=17,
        color=BLUE,
        fontweight="bold",
    )
    fig.text(
        0.95,
        0.055,
        "231,570 evaluated keypoint instances",
        fontsize=15,
        color=MUTED,
        ha="right",
    )
    fig.savefig(OUT / "05_scoring_four_methods.png", facecolor=BG)
    plt.close(fig)


def make_keypoint_3d_diagnostic() -> None:
    with (KP_DIR / "summary.json").open() as f:
        data = json.load(f)["real"]

    groups = ["finger", "palm", "head"]
    labels = ["Fingers", "Palms", "Head"]
    fixed = [data[g]["fixed3d"]["mean"] for g in groups]
    predicted = [data[g]["predicted3d"]["mean"] for g in groups]
    pixel = [data[g]["pixel"]["mean"] for g in groups]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=BG)
    ax = fig.add_axes([0.08, 0.19, 0.56, 0.62], facecolor=BG)
    x = np.arange(len(groups))
    width = 0.34
    b1 = ax.bar(x - width / 2, fixed, width, color=GREEN, label="Calibrated fixed rig")
    b2 = ax.bar(
        x + width / 2,
        predicted,
        width,
        color=RED,
        label="VGGT-predicted rig",
    )
    ax.set_ylabel("Mean 3D keypoint error (mm)", fontsize=18, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=16)
    ax.set_ylim(0, 13.2)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )

    ax2 = fig.add_axes([0.70, 0.51, 0.25, 0.27], facecolor="#FFFFFF")
    colors = [RED, GREEN, PURPLE]
    ax2.bar(np.arange(3), pixel, color=colors, width=0.58)
    ax2.set_title("2D localization", fontsize=17, fontweight="bold", pad=12)
    ax2.set_ylabel("Mean error (px)", fontsize=13)
    ax2.set_xticks(np.arange(3), labels, fontsize=12)
    ax2.set_ylim(0, 7.2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=GRID)
    for i, value in enumerate(pixel):
        ax2.text(i, value + 0.18, f"{value:.2f}", ha="center", fontweight="bold")

    fig.text(0.047, 0.945, "3D KEYPOINT FUSION", fontsize=16, color=BLUE, fontweight="bold")
    fig.text(
        0.047,
        0.885,
        "Accurate 2D keypoints expose the camera-rig bottleneck",
        fontsize=30,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.047,
        0.842,
        "Multiview Gaussian heatmaps are fused into 3D keypoint centers",
        fontsize=16,
        color=MUTED,
    )
    fig.text(
        0.70,
        0.36,
        "10x",
        fontsize=46,
        color=RED,
        fontweight="bold",
    )
    fig.text(
        0.79,
        0.37,
        "larger 3D error",
        fontsize=21,
        color=RED,
        fontweight="bold",
    )
    fig.text(
        0.70,
        0.30,
        "when model-predicted cameras replace",
        fontsize=15,
        color=MUTED,
    )
    fig.text(
        0.70,
        0.265,
        "the calibrated rig",
        fontsize=15,
        color=MUTED,
    )
    fig.text(
        0.70,
        0.185,
        "Operational choice",
        fontsize=18,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.70,
        0.14,
        "Use VGGT for dense geometry;",
        fontsize=15,
        color=BLUE,
        fontweight="bold",
    )
    fig.text(
        0.70,
        0.105,
        "use the fixed rig for metric 3D.",
        fontsize=15,
        color=BLUE,
        fontweight="bold",
    )
    fig.savefig(OUT / "06_keypoint_3d_camera_bottleneck.png", facecolor=BG)
    plt.close(fig)


def make_overview() -> None:
    image, draw = canvas()
    draw_header(
        draw,
        "VGGT-Omega for mouse geometry and fine-grained pose",
        "Synthetic supervision, real-video evaluation, and metric multiview reconstruction",
        "Results overview",
    )

    x0, y0 = 120, 330
    key_im = Image.open(KP_DIR / "real_batch000_view0.png")
    place_cover(image, key_im, (x0, y0, 940, 960))
    draw.rectangle((x0, y0, 940, 960), outline="#FFFFFF", width=3)
    draw.rectangle((x0, 900, 940, 960), fill="#FFFFFF")
    draw.text((x0 + 18, 912), "29-keypoint prediction", fill=INK, font=font(27, True))

    pc = Image.open(GEOM_DIR / "blender_frame_000144" / "pointcloud_comparison.png")
    # Emphasize the top-view row, where multi-view consistency is easiest to compare.
    pc_crop = pc.crop((0, int(pc.height * 0.49), pc.width, pc.height))
    place_contain(image, pc_crop, (1000, 330, 2440, 960), "#FFFFFF")
    draw.rectangle((1000, 900, 2440, 960), fill="#FFFFFF")
    draw.text((1018, 912), "Ground truth  |  Original  |  Finetuned", fill=INK, font=font(27, True))

    draw.rectangle((120, 1020, 2440, 1365), fill="#FFFFFF", outline=GRID, width=2)
    draw_metric(draw, 175, 1060, "4.95 px", "real 2D keypoint error", RED)
    draw_metric(draw, 690, 1060, "1.27 mm", "best median 3D error", BLUE)
    draw_metric(draw, 1210, 1060, "20-24%", "calibration-route improvement", GREEN)
    draw_metric(draw, 1850, 1060, "22 trials", "full-sequence real evaluation", PURPLE)
    draw.text(
        (175, 1270),
        "Main finding: dense geometry improves, while metric 3D still benefits strongly from the calibrated fixed rig.",
        fill=INK,
        font=font(27, True),
    )
    save(image, "00_results_overview.png")


def copy_sources() -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        KP_DIR / "real_batch000_view0.png",
        KP_DIR / "real_batch000_view1.png",
        KP_DIR / "real_batch000_view2.png",
        KP_DIR / "synthetic_batch000_view0.png",
        KP_DIR / "synthetic_batch000_view1.png",
        KP_DIR / "synthetic_batch000_view2.png",
        GEOM_DIR / "blender_frame_000144" / "pointcloud_comparison.png",
        GEOM_DIR / "blender_frame_000144" / "camera_pose_comparison.png",
        REPORT_DIR / "overall_method_comparison.png",
        REPORT_DIR / "trial_delta_heatmap.png",
    ]
    for src in sources:
        shutil.copy2(src, SOURCE_OUT / src.name)


def write_readme() -> None:
    text = """# PPT figure pack

All primary figures are 2560 x 1440 PNG files designed for a 16:9 slide.

1. `00_results_overview.png`
   One-slide overview of keypoint prediction, dense reconstruction, and headline metrics.
2. `01_keypoint_prediction_real.png`
   Three-view real-video keypoint overlays. Green circles are ground truth; red circles are predictions.
3. `02_keypoint_sim2real.png`
   Synthetic and real three-view overlays showing sim-to-real transfer.
4. `03_dense_3d_reconstruction.png`
   Blender dense-GT point-cloud comparison with depth and camera metrics.
5. `04_camera_pose_estimation.png`
   Shared-rig camera pose comparison and the residual real-camera caveat.
6. `05_scoring_four_methods.png`
   Original versus finetuned VGGT-Omega on 22 real trials and four reconstruction routes.
7. `06_keypoint_3d_camera_bottleneck.png`
   Fixed-rig versus predicted-rig 3D keypoint error, with the operational conclusion.

Suggested talk track:

- The synthetic dataset supports dense, fine-grained keypoint supervision.
- The keypoint head transfers to real water-reaching video with sub-7-pixel finger error.
- Geometry-coupled finetuning improves all four full-sequence scoring routes.
- A calibrated fixed rig remains preferable for metric 3D because predicted relative cameras are still the largest error source.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_sources()
    make_overview()
    make_real_keypoints()
    make_sim2real_keypoints()
    make_dense_reconstruction()
    make_camera_pose()
    make_scoring_chart()
    make_keypoint_3d_diagnostic()
    write_readme()
    print(f"Wrote figure pack to {OUT}")


if __name__ == "__main__":
    main()
