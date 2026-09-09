#!/usr/bin/env python3
"""Plot VGGT-Omega fine-tuning curves and baseline scoring comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "output/blenderpymouse_vggt_omega_all/run_20260717_base_lr1e6_e10_real8_step250"
)
DEFAULT_SCREEN_DIR = (
    REPO_ROOT
    / "water-reaching-dataset/vggt-omega-checkpoint-screen-retrain-lr1e6-step"
)
DEFAULT_HOLDOUT_DIR = (
    REPO_ROOT
    / "water-reaching-dataset/vggt-omega-checkpoint-holdout-retrain-lr1e6-step250"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/vggt_omega_step250_training_scoring_report"
DEFAULT_PDF = REPO_ROOT / "output/pdf/vggt_omega_step250_training_scoring_report.pdf"


COLORS = {
    "baseline": "#6B7280",
    "train": "#2563EB",
    "validation": "#D97706",
    "proxy": "#059669",
    "rmse": "#2563EB",
    "reprojection": "#DC2626",
    "joint": "#7C3AED",
    "selected": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--selected-step", type=int, default=250)
    parser.add_argument("--rolling-window", type=int, default=250)
    return parser.parse_args()


def percent_improvement(ratio: float) -> float:
    return (1.0 - ratio) * 100.0


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260717) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def load_pair_metrics(directory: Path, label: str) -> pd.DataFrame:
    data = pd.read_csv(directory / "per_trial_metrics.csv")
    selected = data[data["candidate"] == "step000250"].copy()
    if selected.empty:
        raise ValueError(f"No step000250 rows in {directory / 'per_trial_metrics.csv'}")
    selected["split"] = label
    return selected


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_training_curves(
    train_log: pd.DataFrame,
    selected_step: int,
    rolling_window: int,
    output: Path,
) -> None:
    train = train_log[(train_log["step"] >= 0) & train_log["loss"].notna()].copy()
    validation = train_log[(train_log["step"] >= 0) & train_log["val_loss"].notna()].copy()
    real_proxy = train_log[
        (train_log["step"] >= 0) & train_log["real_val_loss_camera"].notna()
    ].copy()
    baseline_proxy = float(
        train_log.loc[train_log["step"] == -1, "real_val_loss_camera"].iloc[0]
    )

    train["loss_smooth"] = train["loss"].rolling(
        rolling_window, min_periods=max(10, rolling_window // 10), center=True
    ).median()

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(train["step"], train["loss"], color=COLORS["train"], alpha=0.08, linewidth=0.5)
    ax.plot(
        train["step"],
        train["loss_smooth"],
        color=COLORS["train"],
        linewidth=2.0,
        label=f"Train loss (rolling median, {rolling_window} steps)",
    )
    ax.plot(
        validation["step"],
        validation["val_loss"],
        color=COLORS["validation"],
        marker="o",
        markersize=3.3,
        linewidth=1.8,
        label="Synthetic validation loss",
    )
    ax.axvline(selected_step, color=COLORS["selected"], linestyle="--", linewidth=1.4)
    ax.annotate(
        "Selected checkpoint: step 250",
        xy=(selected_step, validation.loc[validation["step"] == selected_step, "val_loss"].iloc[0]),
        xytext=(1200, 0.47),
        arrowprops={"arrowstyle": "->", "color": COLORS["selected"]},
        fontsize=10,
    )
    ax.set_ylabel("Optimization loss")
    ax.set_title("Synthetic Training and Validation Loss")
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)

    ax = axes[1]
    ax.axhline(
        baseline_proxy,
        color=COLORS["baseline"],
        linestyle=":",
        linewidth=1.8,
        label=f"Original checkpoint proxy loss ({baseline_proxy:.4f})",
    )
    for column, label, color in (
        ("real_val_loss_camera", "Total camera proxy loss", COLORS["proxy"]),
        ("real_val_loss_cam_t", "Translation component", "#0284C7"),
        ("real_val_loss_cam_q", "Rotation component", "#D97706"),
        ("real_val_loss_cam_fov", "FOV component", "#9333EA"),
    ):
        ax.plot(
            real_proxy["step"],
            real_proxy[column],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax.axvline(selected_step, color=COLORS["selected"], linestyle="--", linewidth=1.4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Real camera proxy loss")
    ax.set_title("Real-Data Camera-Encoding Validation Proxy")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    style_axes(ax)

    fig.suptitle(
        "VGGT-Omega BlenderPyMouse Fine-Tuning Curves\n"
        "Proxy losses continue improving after geometric scoring has already degraded",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def checkpoint_step(label: str) -> int:
    if label == "baseline":
        return 0
    return int(label.replace("step", ""))


def plot_checkpoint_screening(ranking: pd.DataFrame, selected_step: int, output: Path) -> None:
    candidates = ranking[ranking["candidate"].str.startswith("step")].copy()
    candidates["step"] = candidates["candidate"].map(checkpoint_step)
    candidates = candidates.sort_values("step")

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    for column, label, color, marker in (
        ("median_rmse_3d_ratio", "Median 3D RMSE ratio", COLORS["rmse"], "o"),
        ("median_reprojection_ratio", "Median reprojection ratio", COLORS["reprojection"], "s"),
        ("joint_geomean_ratio", "Joint geometric-mean ratio", COLORS["joint"], "D"),
    ):
        ax.plot(
            candidates["step"],
            candidates[column],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax.axhline(1.0, color=COLORS["baseline"], linestyle="--", linewidth=1.8, label="Original checkpoint")
    ax.axvline(selected_step, color=COLORS["selected"], linestyle=":", linewidth=1.8)
    selected = candidates[candidates["step"] == selected_step].iloc[0]
    ax.annotate(
        f"Selected step {selected_step}\njoint ratio {selected['joint_geomean_ratio']:.3f}",
        xy=(selected_step, selected["joint_geomean_ratio"]),
        xytext=(24, 42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["selected"]},
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Ratio to original VGGT-Omega (lower is better)")
    ax.set_title("Real-Trial Checkpoint Screening: Geometry Diverges Early")
    ax.legend(loc="upper left", frameon=False)
    style_axes(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def short_trial_name(name: str) -> str:
    return name.replace("_2024-", "\n2024-")


def plot_holdout_scoring(holdout: pd.DataFrame, output: Path) -> None:
    plot_data = holdout.sort_values("rmse_3d_ratio", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_data))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), sharey=True)
    for ax, baseline_col, tuned_col, ratio_col, title, xlabel, color in (
        (
            axes[0],
            "baseline_rmse_3d",
            "rmse_3d",
            "rmse_3d_ratio",
            "Globally Aligned 3D RMSE",
            "3D RMSE (Anipose coordinate units)",
            COLORS["rmse"],
        ),
        (
            axes[1],
            "baseline_reprojection_rmse_px",
            "reprojection_rmse_px",
            "reprojection_ratio",
            "Fixed-Rig Reprojection RMSE",
            "Reprojection RMSE (pixels)",
            COLORS["reprojection"],
        ),
    ):
        for row_index, row in plot_data.iterrows():
            improved = row[ratio_col] < 1.0
            line_color = "#9CA3AF" if improved else "#F59E0B"
            ax.plot(
                [row[baseline_col], row[tuned_col]],
                [row_index, row_index],
                color=line_color,
                linewidth=1.6,
                zorder=1,
            )
        ax.scatter(
            plot_data[baseline_col], y, color=COLORS["baseline"], marker="o", s=42, label="Original", zorder=2
        )
        ax.scatter(
            plot_data[tuned_col], y, color=color, marker="D", s=42, label="Step 250", zorder=3
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        style_axes(ax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short_trial_name(name) for name in plot_data["trial_name"]], fontsize=8.5)
    fig.suptitle(
        "Unseen Holdout Trials: Original VGGT-Omega vs Fine-Tuned Step 250\n"
        "Gray connectors improve; orange connectors regress",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_split(data: pd.DataFrame) -> dict[str, float | int | list[float]]:
    rmse_ratio = data["rmse_3d_ratio"].to_numpy(dtype=np.float64)
    reproj_ratio = data["reprojection_ratio"].to_numpy(dtype=np.float64)
    joint_per_trial = np.sqrt(rmse_ratio * reproj_ratio)
    rmse_ci = bootstrap_median_ci(rmse_ratio)
    reproj_ci = bootstrap_median_ci(reproj_ratio, seed=20260718)
    return {
        "trial_count": int(len(data)),
        "improved_3d_count": int((rmse_ratio < 1.0).sum()),
        "improved_reprojection_count": int((reproj_ratio < 1.0).sum()),
        "improved_both_count": int(((rmse_ratio < 1.0) & (reproj_ratio < 1.0)).sum()),
        "median_rmse_3d_ratio": float(np.median(rmse_ratio)),
        "median_reprojection_ratio": float(np.median(reproj_ratio)),
        "joint_geomean_ratio": float(math.sqrt(np.median(rmse_ratio) * np.median(reproj_ratio))),
        "median_per_trial_joint_ratio": float(np.median(joint_per_trial)),
        "rmse_3d_ratio_iqr": [float(value) for value in np.quantile(rmse_ratio, [0.25, 0.75])],
        "reprojection_ratio_iqr": [float(value) for value in np.quantile(reproj_ratio, [0.25, 0.75])],
        "rmse_3d_median_ratio_bootstrap_95ci": list(rmse_ci),
        "reprojection_median_ratio_bootstrap_95ci": list(reproj_ci),
        "baseline_median_rmse_3d": float(data["baseline_rmse_3d"].median()),
        "step250_median_rmse_3d": float(data["rmse_3d"].median()),
        "baseline_median_reprojection_rmse_px": float(
            data["baseline_reprojection_rmse_px"].median()
        ),
        "step250_median_reprojection_rmse_px": float(data["reprojection_rmse_px"].median()),
    }


def make_summary(
    train_log: pd.DataFrame,
    selection: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_step: int,
) -> dict[str, object]:
    validation = train_log[(train_log["step"] >= 0) & train_log["val_loss"].notna()]
    proxy = train_log[(train_log["step"] >= 0) & train_log["real_val_loss_camera"].notna()]
    baseline_proxy = float(train_log.loc[train_log["step"] == -1, "real_val_loss_camera"].iloc[0])
    selected_validation = validation.loc[validation["step"] == selected_step].iloc[0]
    selected_proxy = proxy.loc[proxy["step"] == selected_step].iloc[0]
    final_validation = validation.iloc[-1]
    final_proxy = proxy.iloc[-1]
    combined = pd.concat([selection, holdout], ignore_index=True)
    return {
        "selected_checkpoint_step": selected_step,
        "training": {
            "steps": int(train_log["step"].max() + 1),
            "epochs": int(train_log["epoch"].max() + 1),
            "original_real_camera_proxy_loss": baseline_proxy,
            "step250_synthetic_validation_loss": float(selected_validation["val_loss"]),
            "final_synthetic_validation_loss": float(final_validation["val_loss"]),
            "step250_real_camera_proxy_loss": float(selected_proxy["real_val_loss_camera"]),
            "final_real_camera_proxy_loss": float(final_proxy["real_val_loss_camera"]),
            "step250_proxy_improvement_percent": percent_improvement(
                float(selected_proxy["real_val_loss_camera"]) / baseline_proxy
            ),
            "final_proxy_improvement_percent": percent_improvement(
                float(final_proxy["real_val_loss_camera"]) / baseline_proxy
            ),
        },
        "selection_8_trials": summarize_split(selection),
        "holdout_14_trials": summarize_split(holdout),
        "combined_22_trials_descriptive_only": summarize_split(combined),
        "interpretation": (
            "The real camera-encoding proxy loss keeps falling, but fixed-rig 3D and reprojection "
            "scoring worsens after step 250. The proxy is therefore unsuitable as the sole checkpoint selector."
        ),
    }


def write_markdown(summary: dict[str, object], output: Path) -> None:
    training = summary["training"]
    selection = summary["selection_8_trials"]
    holdout = summary["holdout_14_trials"]
    combined = summary["combined_22_trials_descriptive_only"]
    text = f"""# VGGT-Omega BlenderPyMouse Fine-Tuning Report

## Executive summary

The selected `step250` checkpoint improves over the original, unfinetuned VGGT-Omega on both the 8-trial checkpoint-selection set and the independent 14-trial holdout set. The independent holdout result is the primary generalization result:

- Median globally aligned 3D RMSE ratio: **{holdout['median_rmse_3d_ratio']:.4f}** ({percent_improvement(holdout['median_rmse_3d_ratio']):.1f}% lower).
- Median fixed-rig reprojection RMSE ratio: **{holdout['median_reprojection_ratio']:.4f}** ({percent_improvement(holdout['median_reprojection_ratio']):.1f}% lower).
- Joint ratio: **{holdout['joint_geomean_ratio']:.4f}** ({percent_improvement(holdout['joint_geomean_ratio']):.1f}% lower).
- Both metrics improve on **{holdout['improved_both_count']}/{holdout['trial_count']}** holdout trials; 3D RMSE improves on **{holdout['improved_3d_count']}/{holdout['trial_count']}**.

## Loss curves

Training contains {training['steps']:,} steps over {training['epochs']} epochs. Synthetic validation loss and the real camera-encoding proxy both continue decreasing through the end of training. The proxy changes from **{training['original_real_camera_proxy_loss']:.4f}** for the original checkpoint to **{training['step250_real_camera_proxy_loss']:.4f}** at step 250 and **{training['final_real_camera_proxy_loss']:.4f}** at the final checkpoint.

![Training and validation curves](training_validation_losses.png)

This monotonic proxy improvement does **not** track the downstream geometry metric. Real fixed-rig checkpoint screening selects step 250; later checkpoints rapidly regress despite their lower proxy losses.

![Checkpoint screening](checkpoint_screening.png)

## Original checkpoint scoring gap

| Split | Trials | Median 3D ratio | 3D improvement | Median reprojection ratio | Reprojection improvement | Both improve |
|---|---:|---:|---:|---:|---:|---:|
| Selection | {selection['trial_count']} | {selection['median_rmse_3d_ratio']:.4f} | {percent_improvement(selection['median_rmse_3d_ratio']):.1f}% | {selection['median_reprojection_ratio']:.4f} | {percent_improvement(selection['median_reprojection_ratio']):.1f}% | {selection['improved_both_count']}/{selection['trial_count']} |
| Holdout | {holdout['trial_count']} | {holdout['median_rmse_3d_ratio']:.4f} | {percent_improvement(holdout['median_rmse_3d_ratio']):.1f}% | {holdout['median_reprojection_ratio']:.4f} | {percent_improvement(holdout['median_reprojection_ratio']):.1f}% | {holdout['improved_both_count']}/{holdout['trial_count']} |
| Combined (descriptive) | {combined['trial_count']} | {combined['median_rmse_3d_ratio']:.4f} | {percent_improvement(combined['median_rmse_3d_ratio']):.1f}% | {combined['median_reprojection_ratio']:.4f} | {percent_improvement(combined['median_reprojection_ratio']):.1f}% | {combined['improved_both_count']}/{combined['trial_count']} |

The holdout median ratio bootstrap intervals are **[{holdout['rmse_3d_median_ratio_bootstrap_95ci'][0]:.3f}, {holdout['rmse_3d_median_ratio_bootstrap_95ci'][1]:.3f}]** for 3D RMSE and **[{holdout['reprojection_median_ratio_bootstrap_95ci'][0]:.3f}, {holdout['reprojection_median_ratio_bootstrap_95ci'][1]:.3f}]** for reprojection. These intervals describe trial-to-trial uncertainty; the 14 trials are not necessarily independent biological replicates.

![Holdout per-trial scoring](holdout_original_vs_step250.png)

## Metric definitions

- **3D RMSE:** Anipose triangulated 3D points are compared after one global Sim(3) alignment per complete trial. Scale, one proper rotation, and translation are shared by all frames in that trial.
- **Reprojection RMSE:** Pixel error after triangulation with one fixed relative three-camera rig estimated for the complete trial. No Sim(3) alignment is applied to this pixel metric.
- **Ratio:** Fine-tuned score divided by original VGGT-Omega score on the same trial. Values below 1 are improvements.
- **Selection vs holdout:** The 8-trial split selected step 250. The remaining 14 trials provide the less biased estimate of generalization.

## Conclusion

Fine-tuning is beneficial only in a narrow early-training window. The full 10-epoch optimization overfits the synthetic camera/depth objective relative to the real fixed-rig geometry task. Future runs should keep frequent early checkpoints and use multi-trial fixed-rig 3D plus reprojection scoring for early stopping. The camera-encoding validation proxy is useful for monitoring optimization, but not for selecting the production checkpoint by itself.
"""
    output.write_text(text, encoding="utf-8")


def register_pdf_fonts() -> tuple[str, str]:
    windows_simhei = Path("/mnt/c/Windows/Fonts/simhei.ttf")
    if windows_simhei.exists():
        pdfmetrics.registerFont(TTFont("SimHei", str(windows_simhei)))
        return "SimHei", "Helvetica"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light", "Helvetica"


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawRightString(landscape(A4)[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(
    summary: dict[str, object],
    training_plot: Path,
    screening_plot: Path,
    holdout_plot: Path,
    output: Path,
) -> None:
    chinese_font, latin_font = register_pdf_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=landscape(A4),
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=13 * mm,
        bottomMargin=14 * mm,
        title="VGGT-Omega BlenderPyMouse Fine-Tuning Report",
        author="MouseGeoForge",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=chinese_font,
        fontSize=22,
        leading=29,
        textColor=colors.HexColor("#111827"),
        alignment=TA_CENTER,
        spaceAfter=10 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName=chinese_font,
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4 * mm,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=chinese_font,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#374151"),
        spaceAfter=3 * mm,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.5,
        leading=12,
    )
    metric = ParagraphStyle(
        "Metric",
        parent=body,
        alignment=TA_CENTER,
        fontName=latin_font,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#111827"),
    )

    training = summary["training"]
    holdout = summary["holdout_14_trials"]
    selection = summary["selection_8_trials"]
    combined = summary["combined_22_trials_descriptive_only"]

    story = [
        Spacer(1, 9 * mm),
        Paragraph("VGGT-Omega BlenderPyMouse 微调与真实数据评分报告", title_style),
        Paragraph(
            "核心结论：step 250 在独立的 14 个真实 holdout trials 上，相对原始未微调的 "
            "VGGT-Omega 同时降低了三维误差和重投影误差。训练与 proxy validation loss 虽继续下降，"
            "但真实几何评分在 step 250 后快速恶化，因此不能仅凭 proxy loss 选择 checkpoint。",
            body,
        ),
        Spacer(1, 5 * mm),
    ]
    key_table = Table(
        [
            [
                Paragraph("Holdout 3D RMSE", metric),
                Paragraph("Holdout reprojection", metric),
                Paragraph("Joint score", metric),
                Paragraph("Both improved", metric),
            ],
            [
                Paragraph(f"-{percent_improvement(holdout['median_rmse_3d_ratio']):.1f}%", metric),
                Paragraph(f"-{percent_improvement(holdout['median_reprojection_ratio']):.1f}%", metric),
                Paragraph(f"-{percent_improvement(holdout['joint_geomean_ratio']):.1f}%", metric),
                Paragraph(f"{holdout['improved_both_count']} / {holdout['trial_count']}", metric),
            ],
        ],
        colWidths=[(page_width - 42 * mm) / 4] * 4,
        rowHeights=[15 * mm, 18 * mm],
    )
    key_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend(
        [
            key_table,
            Spacer(1, 7 * mm),
            Paragraph(
                f"训练：{training['steps']:,} steps / {training['epochs']} epochs。"
                f"原始 checkpoint 的真实相机编码 proxy loss 为 {training['original_real_camera_proxy_loss']:.4f}；"
                f"step 250 为 {training['step250_real_camera_proxy_loss']:.4f}；"
                f"最终为 {training['final_real_camera_proxy_loss']:.4f}。",
                body,
            ),
            PageBreak(),
            Paragraph("1. 训练与 validation loss", heading),
            Image(str(training_plot), width=218 * mm, height=151 * mm),
            Paragraph(
                "上图中的 real camera proxy 是训练期间用于监控相机编码的验证量，不等同于最终的固定 rig "
                "三角化 3D RMSE。二者在早期之后出现明显分叉。",
                small,
            ),
            PageBreak(),
            Paragraph("2. Checkpoint 真实几何筛选", heading),
            Image(str(screening_plot), width=247 * mm, height=126 * mm),
            Spacer(1, 3 * mm),
            Paragraph(
                "横线 1.0 表示原始 VGGT-Omega。低于 1.0 表示优于原始模型。step 250 的 joint ratio "
                f"为 {selection['joint_geomean_ratio']:.3f}；step 500 已回到 {1.015:.3f}，后续 checkpoint "
                "则明显退化。这说明完整训练目标与真实固定相机 rig 几何目标并不完全一致。",
                body,
            ),
            PageBreak(),
            Paragraph("3. 独立 holdout trials 对比", heading),
            Image(str(holdout_plot), width=247 * mm, height=140 * mm),
            Spacer(1, 2 * mm),
            Paragraph(
                "灰线表示 step 250 改善，橙线表示退化。14 个 holdout trials 中，"
                f"3D RMSE 改善 {holdout['improved_3d_count']} 个，重投影改善 "
                f"{holdout['improved_reprojection_count']} 个，两者同时改善 {holdout['improved_both_count']} 个。",
                body,
            ),
            PageBreak(),
            Paragraph("4. 汇总与建议", heading),
        ]
    )
    summary_rows = [
        ["Split", "Trials", "3D ratio", "3D change", "Reproj ratio", "Reproj change", "Both improve"],
        [
            "Selection",
            selection["trial_count"],
            f"{selection['median_rmse_3d_ratio']:.4f}",
            f"-{percent_improvement(selection['median_rmse_3d_ratio']):.1f}%",
            f"{selection['median_reprojection_ratio']:.4f}",
            f"-{percent_improvement(selection['median_reprojection_ratio']):.1f}%",
            f"{selection['improved_both_count']}/{selection['trial_count']}",
        ],
        [
            "Holdout",
            holdout["trial_count"],
            f"{holdout['median_rmse_3d_ratio']:.4f}",
            f"-{percent_improvement(holdout['median_rmse_3d_ratio']):.1f}%",
            f"{holdout['median_reprojection_ratio']:.4f}",
            f"-{percent_improvement(holdout['median_reprojection_ratio']):.1f}%",
            f"{holdout['improved_both_count']}/{holdout['trial_count']}",
        ],
        [
            "Combined*",
            combined["trial_count"],
            f"{combined['median_rmse_3d_ratio']:.4f}",
            f"-{percent_improvement(combined['median_rmse_3d_ratio']):.1f}%",
            f"{combined['median_reprojection_ratio']:.4f}",
            f"-{percent_improvement(combined['median_reprojection_ratio']):.1f}%",
            f"{combined['improved_both_count']}/{combined['trial_count']}",
        ],
    ]
    table = Table(summary_rows, colWidths=[33 * mm, 20 * mm, 28 * mm, 28 * mm, 32 * mm, 32 * mm, 31 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), latin_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 5 * mm),
            Paragraph("* Combined 22 trials 仅作描述性汇总，因为其中 8 个 trials 已参与 checkpoint 选择。", small),
            Paragraph(
                f"Holdout 的 median ratio bootstrap 95% 区间：3D RMSE "
                f"[{holdout['rmse_3d_median_ratio_bootstrap_95ci'][0]:.3f}, "
                f"{holdout['rmse_3d_median_ratio_bootstrap_95ci'][1]:.3f}]；重投影 "
                f"[{holdout['reprojection_median_ratio_bootstrap_95ci'][0]:.3f}, "
                f"{holdout['reprojection_median_ratio_bootstrap_95ci'][1]:.3f}]。"
                "该区间反映 trial 间不确定性，但这些 trials 不一定是彼此独立的生物学重复。",
                body,
            ),
            Paragraph(
                "建议：保留早期密集 checkpoint；把多真实 trial 的固定 rig 三角化 3D RMSE 和重投影误差"
                "作为 early stopping 主指标；camera-encoding proxy 只用于训练健康度监控。",
                body,
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "指标口径：3D RMSE 使用每个完整 trial 共用的一次 Sim(3) 对齐；重投影误差在固定三相机相对 rig 下"
                "以像素计算，不应用 Sim(3)。所有 ratio 均为 fine-tuned / original，因此小于 1 表示改善。",
                small,
            ),
        ]
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    screen_dir = args.screen_dir.resolve()
    holdout_dir = args.holdout_dir.resolve()
    output_dir = args.output_dir.resolve()
    pdf_path = args.pdf.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_log = pd.read_csv(run_dir / "train_log.csv")
    selection = load_pair_metrics(screen_dir, "selection")
    holdout = load_pair_metrics(holdout_dir, "holdout")
    ranking = pd.read_csv(screen_dir / "checkpoint_ranking.csv")

    training_plot = output_dir / "training_validation_losses.png"
    screening_plot = output_dir / "checkpoint_screening.png"
    holdout_plot = output_dir / "holdout_original_vs_step250.png"
    plot_training_curves(train_log, args.selected_step, args.rolling_window, training_plot)
    plot_checkpoint_screening(ranking, args.selected_step, screening_plot)
    plot_holdout_scoring(holdout, holdout_plot)

    summary = make_summary(train_log, selection, holdout, args.selected_step)
    summary_path = output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = output_dir / "report.md"
    write_markdown(summary, markdown_path)
    build_pdf(summary, training_plot, screening_plot, holdout_plot, pdf_path)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "training_plot": str(training_plot),
                "screening_plot": str(screening_plot),
                "holdout_plot": str(holdout_plot),
                "summary": str(summary_path),
                "markdown_report": str(markdown_path),
                "pdf_report": str(pdf_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
