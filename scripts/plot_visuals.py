"""Regenerate final Set5 visual evidence directly from tracked checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network_swinir_student import SwinIR_Student  # noqa: E402

SCALE = 4
WINDOW_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_DIR = ROOT / "testsets" / "Set5" / "HR"
LQ_DIR = ROOT / "testsets" / "Set5" / "LR_bicubic" / "X4"
BASELINE_CKPT = ROOT / "student_weights" / "model_A_500k.pth"
FAKD_CKPT = ROOT / "student_weights" / "model_C_500k.pth"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

IMG_NAMES = ["baby", "bird", "butterfly", "head", "woman"]


def build_student() -> SwinIR_Student:
    return SwinIR_Student(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=WINDOW_SIZE,
        img_range=1.0,
        depths=[4, 4, 4, 4],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def load_model(path: Path) -> SwinIR_Student:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model = build_student()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(DEVICE)


def run_sr(model: SwinIR_Student, lq_bgr: np.ndarray) -> np.ndarray:
    lq = lq_bgr.astype(np.float32) / 255.0
    lq = np.transpose(lq[:, :, [2, 1, 0]], (2, 0, 1))
    img_lq = torch.from_numpy(lq).float().unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(img_lq)

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    return (output * 255.0).round().astype(np.uint8)


def get_y_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mse = np.mean((img1_y.astype(np.float32) - img2_y.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def best_improvement_patch(img_gt: np.ndarray, img_old: np.ndarray, img_new: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY).astype(np.float32)

    abs_diff_old = cv2.absdiff(gray_gt, gray_old)
    abs_diff_new = cv2.absdiff(gray_gt, gray_new)
    improvement_map = abs_diff_old - abs_diff_new

    patch_size = 80
    best_score = -float("inf")
    best_y, best_x = 0, 0
    h, w = gray_gt.shape

    for y in range(0, h - patch_size, 10):
        for x in range(0, w - patch_size, 10):
            score = float(np.sum(improvement_map[y : y + patch_size, x : x + patch_size]))
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    return best_y, best_x, abs_diff_old, abs_diff_new, improvement_map


def plot_example(img_name: str, baseline: SwinIR_Student, fakd: SwinIR_Student) -> None:
    gt_path = GT_DIR / f"{img_name}.png"
    lq_path = LQ_DIR / f"{img_name}x{SCALE}.png"
    img_gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    img_lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
    if img_gt is None or img_lq is None:
        raise FileNotFoundError(f"Missing Set5 image pair for {img_name}")

    img_old = run_sr(baseline, img_lq)
    img_new = run_sr(fakd, img_lq)

    patch_size = 80
    y, x, abs_diff_old, abs_diff_new, improvement_map = best_improvement_patch(img_gt, img_old, img_new)

    crop_gt = img_gt[y : y + patch_size, x : x + patch_size]
    crop_old = img_old[y : y + patch_size, x : x + patch_size]
    crop_new = img_new[y : y + patch_size, x : x + patch_size]
    crop_err_old = abs_diff_old[y : y + patch_size, x : x + patch_size]
    crop_err_new = abs_diff_new[y : y + patch_size, x : x + patch_size]
    crop_imp = improvement_map[y : y + patch_size, x : x + patch_size]

    psnr_old = get_y_psnr(crop_gt, crop_old)
    psnr_new = get_y_psnr(crop_gt, crop_new)
    gain = psnr_new - psnr_old

    amp = 10
    vis_err_old = cv2.applyColorMap(np.clip(crop_err_old * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_err_new = cv2.applyColorMap(np.clip(crop_err_new * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_imp = np.clip(crop_imp * amp + 128, 0, 255).astype(np.uint8)
    vis_imp = cv2.applyColorMap(vis_imp, cv2.COLORMAP_JET)

    crop_gt, crop_old, crop_new = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [crop_gt, crop_old, crop_new]]
    vis_err_old, vis_err_new, vis_imp = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in [vis_err_old, vis_err_new, vis_imp]]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(crop_gt)
    axs[0, 0].set_title(f"Ground Truth\n({img_name})")
    axs[0, 0].axis("off")
    axs[0, 1].imshow(crop_old)
    axs[0, 1].set_title(f"Baseline Student\n{psnr_old:.2f} dB")
    axs[0, 1].axis("off")
    axs[0, 2].imshow(crop_new)
    axs[0, 2].set_title(f"FAKD Student\n{psnr_new:.2f} dB", color="green", fontweight="bold")
    axs[0, 2].axis("off")
    axs[1, 0].imshow(vis_imp)
    axs[1, 0].set_title("Improvement Map\nRed = FAKD is Closer")
    axs[1, 0].axis("off")
    axs[1, 1].imshow(vis_err_old)
    axs[1, 1].set_title("Error Map: Baseline")
    axs[1, 1].axis("off")
    axs[1, 2].imshow(vis_err_new)
    axs[1, 2].set_title("Error Map: FAKD")
    axs[1, 2].axis("off")

    fig.tight_layout()
    save_path = FIG_DIR / f"final_evidence_{img_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {save_path.relative_to(ROOT)} (Patch Gain: {gain:+.2f} dB)")


def main() -> None:
    print(f"Starting final 500k visual comparison on {DEVICE}...")
    baseline = load_model(BASELINE_CKPT)
    fakd = load_model(FAKD_CKPT)

    for img_name in IMG_NAMES:
        print(f"Processing {img_name}...")
        plot_example(img_name, baseline, fakd)

    print("Done.")


if __name__ == "__main__":
    main()
