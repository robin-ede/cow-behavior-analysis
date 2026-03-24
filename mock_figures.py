# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.10.19)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mock Figures for Research Meeting
#
# Standalone script that generates all 6 publication-style figures using realistic
# fake data. Run this to preview the visual layout before real training artifacts
# are available.
#
# Output: `artifacts/figures/mock/`
#

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Global style — clean minimal, print-ready
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

# %%
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEHAVIOR_CLASSES = [
    "drinking water",
    "foraging",
    "lying down",
    "ruminating",
    "standing",
]

# ColorBrewer Set2 — 5 colors, colorblind-safe
BEHAVIOR_COLORS = {
    "drinking water": "#66c2a5",
    "foraging": "#fc8d62",
    "lying down": "#8da0cb",
    "ruminating": "#e78ac3",
    "standing": "#a6d854",
}

OUTPUT_DIR = Path("artifacts/figures/mock")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# Fake data — all mock numbers in one place for easy swapping
# ---------------------------------------------------------------------------

# Fig 1 — class distribution (train / val / test counts)
CLASS_COUNTS = {
    #                   train   val    test
    "drinking water": (412, 88, 89),
    "foraging": (3_801, 814, 815),
    "lying down": (2_943, 630, 631),
    "ruminating": (4_217, 903, 904),
    "standing": (6_152, 1_318, 1_318),
}

# Fig 2 — YOLO detection summary metrics (these are the real reported numbers)
DETECTION_METRICS = {
    "mAP50": 0.901,
    "Precision": 0.870,
    "Recall": 0.849,
}

# Fig 3 — ViT training history (10 epochs)
rng = np.random.default_rng(42)
_epochs = np.arange(1, 11)

# Train loss: sharp initial drop, then slow decay
_train_loss_base = 0.85 * np.exp(-0.55 * (_epochs - 1)) + 0.10
TRAIN_LOSS = _train_loss_base + rng.normal(0, 0.008, size=10)

# Val loss: similar but slightly higher, small uptick after epoch ~7 (realistic)
_val_loss_base = 0.90 * np.exp(-0.50 * (_epochs - 1)) + 0.13
_val_loss_base[7:] += np.array([0.010, 0.018, 0.022])  # slight overfit tail
VAL_LOSS = _val_loss_base + rng.normal(0, 0.010, size=10)

# Val accuracy: jumps to ~91.6% after epoch 1, plateaus near 92.8%
_acc_base = 0.928 - 0.012 * np.exp(-0.8 * (_epochs - 1))
_acc_base[0] = 0.916
VAL_ACCURACY = np.clip(_acc_base + rng.normal(0, 0.003, size=10), 0.88, 0.935)
VAL_ACCURACY[-1] = 0.9281  # pin updated held-out test number

# Fig 4 — confusion matrix (5×5, row = true, col = predicted)
# Designed to reproduce reported error patterns:
#   lying↔rumination ~14%, drinking↔standing ~7%
_cm_raw = np.array(
    [
        # drnk  forg  lying  rumi  stand
        [78, 2, 1, 0, 8],  # drinking water  (true)
        [1, 790, 3, 6, 4],  # foraging
        [0, 4, 610, 88, 2],  # lying down      (14% → rumination)
        [0, 8, 52, 848, 3],  # ruminating
        [7, 5, 2, 3, 1_270],  # standing
    ],
    dtype=float,
)

CONFUSION_MATRIX = _cm_raw

# Fig 5 — precision & recall (derived from the confusion matrix above)
_tp = np.diag(_cm_raw)
_fp = _cm_raw.sum(axis=0) - _tp
_fn = _cm_raw.sum(axis=1) - _tp
PRECISION_PER_CLASS = _tp / (_tp + _fp + 1e-9)
RECALL_PER_CLASS = _tp / (_tp + _fn + 1e-9)

# Fig 6 — herd behavior timeline (48 half-hour slots across a 24-hour period)
# Each row is one time slot; columns are fraction of herd in each behavior.
# Encodes realistic cattle rhythms:
#   - lying down peaks overnight and in early morning
#   - foraging peaks around 3 feeding windows (~06:00, ~12:00, ~18:00)
#   - rumination follows foraging with a ~1-hour lag
#   - standing fills the remaining share
#   - drinking water stays low (~2–5%) throughout
_hours = np.linspace(0, 24, 48, endpoint=False)  # 48 half-hour slots


def _feeding_pulse(hours, center, width=1.8, height=0.22):
    """Gaussian bump centered on a feeding time."""
    return height * np.exp(-0.5 * ((hours - center) / width) ** 2)


def _rumination_pulse(hours, center, width=2.2, height=0.18):
    """Rumination follows feeding with a lag."""
    return height * np.exp(-0.5 * ((hours - (center + 1.2)) / width) ** 2)


rng6 = np.random.default_rng(7)

# Drinking: low baseline with tiny random bumps near water access times
_drinking = np.clip(
    0.025 + 0.015 * np.sin(2 * np.pi * _hours / 8) + rng6.normal(0, 0.008, 48),
    0.01,
    0.07,
)

# Foraging: three daily meals
_foraging = (
    _feeding_pulse(_hours, 6.0)
    + _feeding_pulse(_hours, 12.0, height=0.20)
    + _feeding_pulse(_hours, 18.0, height=0.24)
    + 0.10
    + rng6.normal(0, 0.015, 48)
)

# Rumination: follows each feeding bout
_rumination = (
    _rumination_pulse(_hours, 6.0)
    + _rumination_pulse(_hours, 12.0)
    + _rumination_pulse(_hours, 18.0, height=0.20)
    + 0.08
    + rng6.normal(0, 0.012, 48)
)

# Lying down: higher overnight (22–24, 0–5) and a midday rest dip
_night_mask = np.where((_hours < 5) | (_hours > 21), 0.28, 0.08)
_lying = (
    _night_mask
    + 0.06 * np.cos(2 * np.pi * (_hours - 13) / 24)
    + rng6.normal(0, 0.015, 48)
)

# Clip everything non-negative before normalising
_foraging = np.clip(_foraging, 0.04, None)
_rumination = np.clip(_rumination, 0.03, None)
_lying = np.clip(_lying, 0.02, None)
_drinking = np.clip(_drinking, 0.01, None)

# Standing is the residual — computed after normalisation
_stack_raw = np.column_stack([_lying, _foraging, _rumination, _drinking])
_stack_sum = _stack_raw.sum(axis=1, keepdims=True)
# Cap the four components so standing always has at least 15%
_scale = np.where(_stack_sum > 0.85, 0.85 / _stack_sum, 1.0)
_stack_raw *= _scale
_standing = 1.0 - _stack_raw.sum(axis=1)


# Apply a light rolling average (window=3) to smooth for print
def _smooth(arr, w=3):
    return np.convolve(arr, np.ones(w) / w, mode="same")


_lying = _smooth(_stack_raw[:, 0])
_foraging = _smooth(_stack_raw[:, 1])
_rumination = _smooth(_stack_raw[:, 2])
_drinking = _smooth(_stack_raw[:, 3])
_standing = _smooth(_standing)

# Final normalisation to ensure columns sum exactly to 1 at every time step
HERD_TIMELINE = np.column_stack([_lying, _foraging, _rumination, _drinking, _standing])
HERD_TIMELINE = np.clip(HERD_TIMELINE, 0, None)
HERD_TIMELINE /= HERD_TIMELINE.sum(axis=1, keepdims=True)
TIMELINE_HOURS = _hours

# %%
# ---------------------------------------------------------------------------
# Figure 1 — Class Distribution
# ---------------------------------------------------------------------------


def save_class_distribution(output_path: Path) -> None:
    splits = ["Train", "Val", "Test"]
    split_colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    x = np.arange(len(BEHAVIOR_CLASSES))
    n_splits = len(splits)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, (split, color) in enumerate(zip(splits, split_colors)):
        counts = [CLASS_COUNTS[cls][i] for cls in BEHAVIOR_CLASSES]
        bars = ax.bar(
            x + (i - 1) * bar_w,
            counts,
            bar_w,
            label=split,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in BEHAVIOR_CLASSES])
    ax.set_ylabel("Sample count")
    ax.set_title("Dataset Class Distribution by Split")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Figure 2 — YOLO Detection Metrics
# ---------------------------------------------------------------------------


def save_detection_metrics(output_path: Path) -> None:
    labels = list(DETECTION_METRICS.keys())
    values = list(DETECTION_METRICS.values())
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.barh(
        labels, values, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5
    )

    for bar, val in zip(bars, values):
        ax.text(
            val - 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="right",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("YOLOv11 Detection Performance")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Figure 3 — ViT Training Curves
# ---------------------------------------------------------------------------


def save_training_curves(output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(3.5, 3.0))
    ax2 = ax1.twinx()

    color_loss = "#4e79a7"
    color_acc = "#e15759"

    (l1,) = ax1.plot(
        _epochs,
        TRAIN_LOSS,
        color=color_loss,
        lw=1.5,
        linestyle="-",
        marker="o",
        markersize=3,
        label="Train loss",
    )
    (l2,) = ax1.plot(
        _epochs,
        VAL_LOSS,
        color=color_loss,
        lw=1.5,
        linestyle="--",
        marker="s",
        markersize=3,
        label="Val loss",
    )
    (l3,) = ax2.plot(
        _epochs,
        VAL_ACCURACY * 100,
        color=color_acc,
        lw=1.5,
        linestyle="-",
        marker="^",
        markersize=3,
        label="Val accuracy",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax2.set_ylabel("Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(_epochs)
    ax2.set_ylim(85, 97)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, loc="center right")
    ax1.set_title("ViT Training Curves")

    # Turn off twin-axis top/right spines that rcParams doesn't catch
    ax2.spines["top"].set_visible(False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Figure 4 — Confusion Matrix
# ---------------------------------------------------------------------------


def save_confusion_matrix(output_path: Path) -> None:
    cm = CONFUSION_MATRIX
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    short_labels = ["drinking", "foraging", "lying\ndown", "ruminating", "standing"]

    # Custom Blues colormap with white at 0
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_yticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_xticklabels(short_labels)
    ax.set_yticklabels(short_labels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if val > 0.55 else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix (Normalized)")
    ax.grid(visible=False)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Figure 5 — Precision & Recall by Behavior Class
# ---------------------------------------------------------------------------


def save_precision_recall(output_path: Path) -> None:
    x = np.arange(len(BEHAVIOR_CLASSES))
    bar_w = 0.35
    short_labels = [
        "drinking\nwater",
        "foraging",
        "lying\ndown",
        "ruminating",
        "standing",
    ]

    fig, ax = plt.subplots(figsize=(7, 3.2))

    bars_p = ax.bar(
        x - bar_w / 2,
        PRECISION_PER_CLASS,
        bar_w,
        label="Precision",
        color="#4e79a7",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x + bar_w / 2,
        RECALL_PER_CLASS,
        bar_w,
        label="Recall",
        color="#f28e2b",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_p, bars_r]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall by Behavior Class")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Figure 6 — Herd Behavior Timeline (Stacked Area)
# ---------------------------------------------------------------------------

# Column order in HERD_TIMELINE matches this list
_TIMELINE_CLASSES = [
    "lying down",
    "foraging",
    "ruminating",
    "drinking water",
    "standing",
]


def save_herd_behavior_timeline(output_path: Path) -> None:
    colors = [BEHAVIOR_COLORS[cls] for cls in _TIMELINE_CLASSES]
    hours = TIMELINE_HOURS

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Build cumulative stack bottom-up
    bottoms = np.zeros(len(hours))
    for j, (cls, color) in enumerate(zip(_TIMELINE_CLASSES, colors)):
        vals = HERD_TIMELINE[:, j]
        ax.fill_between(
            hours,
            bottoms,
            bottoms + vals,
            color=color,
            alpha=0.88,
            linewidth=0,
            label=cls,
        )
        bottoms = bottoms + vals

    # Thin boundary lines between layers for print clarity
    bottoms2 = np.zeros(len(hours))
    for j in range(len(_TIMELINE_CLASSES) - 1):
        bottoms2 += HERD_TIMELINE[:, j]
        ax.plot(hours, bottoms2, color="white", linewidth=0.4, alpha=0.6)

    # Annotate feeding peaks
    for feed_hour, label in [
        (6, "morning\nfeed"),
        (12, "midday\nfeed"),
        (18, "evening\nfeed"),
    ]:
        ax.axvline(feed_hour, color="#555555", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(
            feed_hour + 0.2,
            0.97,
            label,
            va="top",
            ha="left",
            fontsize=6.5,
            color="#444444",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Share of Herd")
    ax.set_title("Herd Behavior Distribution Over 24 Hours")
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        fontsize=7.5,
        ncol=1,
    )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# ---------------------------------------------------------------------------
# Main — generate all figures
# ---------------------------------------------------------------------------

print(f"Writing mock figures to: {OUTPUT_DIR.resolve()}\n")

save_class_distribution(OUTPUT_DIR / "fig1_class_distribution.png")
save_detection_metrics(OUTPUT_DIR / "fig2_detection_metrics.png")
save_training_curves(OUTPUT_DIR / "fig3_training_curves.png")
save_confusion_matrix(OUTPUT_DIR / "fig4_confusion_matrix.png")
save_precision_recall(OUTPUT_DIR / "fig5_precision_recall.png")
save_herd_behavior_timeline(OUTPUT_DIR / "fig6_herd_behavior_timeline.png")

print(f"\nAll 6 figures saved to {OUTPUT_DIR.resolve()}")
