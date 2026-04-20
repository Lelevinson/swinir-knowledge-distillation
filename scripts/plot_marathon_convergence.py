"""Plot final 500k Set5 validation curves from tracked training logs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

LOGS = {
    "Baseline Student": ROOT / "traininglogs" / "Model_A_500k_train.log",
    "FAKD Student": ROOT / "traininglogs" / "Model_C_500k_train.log",
}

MAX_ITER = 500_000


def parse_psnr_log(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    iterations: list[int] = []
    psnrs: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Average PSNR" not in line:
            continue
        iter_match = re.search(r"iter:\s*([\d,]+)", line)
        psnr_match = re.search(r"Average PSNR\s*:\s*([\d.]+)", line)
        if not iter_match or not psnr_match:
            continue

        iteration = int(iter_match.group(1).replace(",", ""))
        if iteration > MAX_ITER:
            continue
        iterations.append(iteration)
        psnrs.append(float(psnr_match.group(1)))

    if not iterations:
        raise ValueError(f"No Average PSNR entries found in {path}")
    return iterations, psnrs


def main() -> None:
    styles = {
        "Baseline Student": {"color": "#7A7F87", "linestyle": "--", "marker": "o"},
        "FAKD Student": {"color": "#4C9F70", "linestyle": "-", "marker": "*"},
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    final_points = {}

    for label, path in LOGS.items():
        iterations, psnrs = parse_psnr_log(path)
        ax.plot(iterations, psnrs, linewidth=2, label=label, **styles[label])
        final_points[label] = (iterations[-1], psnrs[-1])
        ax.scatter([iterations[-1]], [psnrs[-1]], s=90, color=styles[label]["color"], edgecolors="#222222", zorder=3)

    base_iter, base_psnr = final_points["Baseline Student"]
    fakd_iter, fakd_psnr = final_points["FAKD Student"]
    gain = fakd_psnr - base_psnr

    ax.text(base_iter, base_psnr - 0.08, f"{base_psnr:.2f} dB", ha="right", color=styles["Baseline Student"]["color"])
    ax.text(
        fakd_iter,
        fakd_psnr + 0.05,
        f"{fakd_psnr:.2f} dB ({gain:+.2f})",
        ha="right",
        color=styles["FAKD Student"]["color"],
        fontweight="bold",
    )

    ax.set_title("Set5 x4 Validation PSNR Through 500k Iterations")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("RGB PSNR (dB)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output = FIG_DIR / "figure_marathon_convergence.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
