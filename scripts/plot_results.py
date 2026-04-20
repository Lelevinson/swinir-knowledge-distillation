"""Regenerate current paper-facing result figures.

The values here are the locally verified Set5 x4 and latency numbers documented
in docs/reproducibility_set5.md. This script does not run inference or retrain;
it only redraws summary figures from verified metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

STUDENT_PARAMS_M = 988_959 / 1_000_000
TEACHER_PARAMS_M = 11_900_199 / 1_000_000
PARAM_REDUCTION = (1.0 - STUDENT_PARAMS_M / TEACHER_PARAMS_M) * 100.0

PSNR_RGB = {
    "Baseline Student": 30.4532,
    "FAKD Student": 30.5133,
    "SwinIR-M Teacher": 30.9883,
}

LATENCY_MS = {
    "SwinIR-M Teacher": 40.13,
    "FAKD Student": 19.70,
    "Baseline Student": 20.28,
}


def save_params_figure() -> None:
    labels = ["Student", "Teacher"]
    values = [STUDENT_PARAMS_M, TEACHER_PARAMS_M]
    colors = ["#4C9F70", "#7A7F87"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}M",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Model Size Comparison")
    ax.set_ylabel("Parameters (Millions)")
    ax.text(
        0.5,
        max(values) * 0.62,
        f"Student is {PARAM_REDUCTION:.1f}% smaller",
        ha="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999"},
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_params.png", dpi=300)
    plt.close(fig)


def save_efficiency_figure() -> None:
    points = [
        ("Baseline Student", STUDENT_PARAMS_M, PSNR_RGB["Baseline Student"], "#7A7F87"),
        ("FAKD Student", STUDENT_PARAMS_M, PSNR_RGB["FAKD Student"], "#4C9F70"),
        ("SwinIR-M Teacher", TEACHER_PARAMS_M, PSNR_RGB["SwinIR-M Teacher"], "#2F6B9A"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, params, psnr, color in points:
        ax.scatter(params, psnr, s=130, color=color, edgecolors="#222222", linewidth=0.8, label=label)
        y_offset = 0.045 if label != "Baseline Student" else -0.075
        ax.text(params, psnr + y_offset, f"{psnr:.2f} dB", ha="center", fontsize=10)

    ax.annotate(
        "+0.06 dB same-backbone gain",
        xy=(STUDENT_PARAMS_M, PSNR_RGB["FAKD Student"]),
        xytext=(2.3, 30.72),
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.1},
        fontsize=10,
    )
    ax.set_title("Set5 x4 Performance vs Model Size")
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("RGB PSNR (dB)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_efficiency.png", dpi=300)
    plt.close(fig)


def save_latency_figure() -> None:
    labels = list(LATENCY_MS.keys())
    values = list(LATENCY_MS.values())
    colors = ["#2F6B9A", "#4C9F70", "#7A7F87"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_title("Latency on 64x64 LR Input")
    ax.set_ylabel("Mean latency (ms)")
    ax.text(
        0.5,
        max(values) * 0.78,
        "FAKD student: 2.04x faster than teacher",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999"},
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_latency.png", dpi=300)
    plt.close(fig)


def main() -> None:
    save_params_figure()
    save_efficiency_figure()
    save_latency_figure()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
