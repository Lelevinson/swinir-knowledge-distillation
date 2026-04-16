"""No-save reproducibility helpers for the verified Set5 x4 results.

The main evaluation scripts in this repository save result images. This helper
keeps the checked paper metrics reproducible without touching checkpoints or
existing result folders.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network_swinir import SwinIR  # noqa: E402
from models.network_swinir_student import SwinIR_Student  # noqa: E402
from utils import util_calculate_psnr_ssim as util_metrics  # noqa: E402
from utils import utils_image as util_image  # noqa: E402

SCALE = 4
WINDOW_SIZE = 8

SET5_GT = ROOT / "testsets" / "Set5" / "HR"
SET5_LQ = ROOT / "testsets" / "Set5" / "LR_bicubic" / "X4"

BASELINE_CKPT = ROOT / "student_weights" / "model_A_500k.pth"
FAKD_CKPT = ROOT / "student_weights" / "model_C_500k.pth"
TEACHER_CKPT = ROOT / "model_zoo" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"

BASELINE_RESULTS = ROOT / "results" / "swinir_classical_sr_x4_model_A_500k"
FAKD_RESULTS = ROOT / "results" / "swinir_classical_sr_x4_model_C_500k"

STUDENTS = OrderedDict(
    [
        ("Baseline Student", BASELINE_CKPT),
        ("FAKD Student", FAKD_CKPT),
    ]
)


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_state(path: Path):
    require_path(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    return checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint


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


def build_teacher() -> SwinIR:
    return SwinIR(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=WINDOW_SIZE,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def load_model(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(load_state(path), strict=True)
    model.eval()
    return model.to(device)


def image_pairs():
    require_path(SET5_GT)
    require_path(SET5_LQ)
    for gt_path in sorted(SET5_GT.glob("*.png")):
        lq_candidates = [
            SET5_LQ / gt_path.name,
            SET5_LQ / f"{gt_path.stem}x{SCALE}{gt_path.suffix}",
            SET5_LQ / f"{gt_path.stem}_x{SCALE}{gt_path.suffix}",
        ]
        lq_path = next((path for path in lq_candidates if path.exists()), lq_candidates[0])
        require_path(lq_path)
        yield gt_path, lq_path


def result_image_path(result_dir: Path, gt_path: Path) -> Path:
    candidates = [
        result_dir / gt_path.name,
        result_dir / f"{gt_path.stem}_SwinIR{gt_path.suffix}",
        result_dir / f"{gt_path.stem}x{SCALE}_SwinIR{gt_path.suffix}",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def tensor_from_uint_rgb(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return util_image.uint2tensor4(img).to(device)


def eval_training_rgb(device: torch.device) -> OrderedDict[str, float]:
    """Match the RGB validation protocol used for the reported 30.45/30.51."""
    results = OrderedDict()
    for label, ckpt in STUDENTS.items():
        model = load_model(build_student(), ckpt, device)
        scores = []
        print(f"\n{label}")
        for gt_path, lq_path in image_pairs():
            gt = util_image.imread_uint(str(gt_path), 3)
            gt = util_image.modcrop(gt, SCALE)
            lq = util_image.imread_uint(str(lq_path), 3)

            img_lq = tensor_from_uint_rgb(lq, device)
            with torch.no_grad():
                output = model(img_lq)

            sr = util_image.tensor2uint(output)
            psnr = util_image.calculate_psnr(sr, gt, border=SCALE)
            scores.append(psnr)
            print(f"  {gt_path.name}: RGB PSNR {psnr:.4f}")

        average = float(np.mean(scores))
        results[label] = average
        print(f"  Average RGB PSNR: {average:.4f}")
    gain = results["FAKD Student"] - results["Baseline Student"]
    print(f"\nFAKD gain over baseline: {gain:+.4f} dB")
    return results


def calc_full_metrics(sr_bgr: np.ndarray, gt_bgr: np.ndarray) -> tuple[float, float, float, float]:
    rgb_psnr = util_metrics.calculate_psnr(sr_bgr, gt_bgr, crop_border=SCALE, test_y_channel=False)
    rgb_ssim = util_metrics.calculate_ssim(sr_bgr, gt_bgr, crop_border=SCALE, test_y_channel=False)
    y_psnr = util_metrics.calculate_psnr(sr_bgr, gt_bgr, crop_border=SCALE, test_y_channel=True)
    y_ssim = util_metrics.calculate_ssim(sr_bgr, gt_bgr, crop_border=SCALE, test_y_channel=True)
    return rgb_psnr, rgb_ssim, y_psnr, y_ssim


def print_metric_average(label: str, rows: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    average = tuple(float(np.mean(values)) for values in zip(*rows))
    print(
        f"  Average: RGB PSNR {average[0]:.4f}, RGB SSIM {average[1]:.6f}, "
        f"Y PSNR {average[2]:.4f}, Y SSIM {average[3]:.6f}"
    )
    return average


def eval_saved_results() -> OrderedDict[str, tuple[float, float, float, float]]:
    results = OrderedDict()
    folders = OrderedDict(
        [
            ("Baseline Student", BASELINE_RESULTS),
            ("FAKD Student", FAKD_RESULTS),
        ]
    )
    for label, result_dir in folders.items():
        require_path(result_dir)
        rows = []
        print(f"\n{label}: {result_dir.relative_to(ROOT)}")
        for gt_path, _ in image_pairs():
            sr_path = result_image_path(result_dir, gt_path)
            require_path(sr_path)
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            sr = cv2.imread(str(sr_path), cv2.IMREAD_UNCHANGED)
            row = calc_full_metrics(sr, gt)
            rows.append(row)
            print(
                f"  {gt_path.name}: RGB PSNR {row[0]:.4f}, RGB SSIM {row[1]:.6f}, "
                f"Y PSNR {row[2]:.4f}, Y SSIM {row[3]:.6f}"
            )
        results[label] = print_metric_average(label, rows)

    rgb_gain = results["FAKD Student"][0] - results["Baseline Student"][0]
    y_gain = results["FAKD Student"][2] - results["Baseline Student"][2]
    print(f"\nFAKD gain over baseline: RGB {rgb_gain:+.4f} dB, Y {y_gain:+.4f} dB")
    return results


def pad_for_swinir(img_lq: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
    w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
    padded = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, : h_old + h_pad, :]
    padded = torch.cat([padded, torch.flip(padded, [3])], 3)[:, :, :, : w_old + w_pad]
    return padded, h_old, w_old


def eval_teacher(device: torch.device) -> tuple[float, float, float, float]:
    """Evaluate the local SwinIR-M teacher with img_size/training_patch_size 64."""
    model = load_model(build_teacher(), TEACHER_CKPT, device)
    rows = []
    print(f"\nSwinIR-M Teacher: {TEACHER_CKPT.relative_to(ROOT)}")
    for gt_path, lq_path in image_pairs():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        lq = cv2.imread(str(lq_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        img_lq = torch.from_numpy(np.transpose(lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lq = img_lq.unsqueeze(0).to(device)
        img_lq, h_old, w_old = pad_for_swinir(img_lq)

        with torch.no_grad():
            output = model(img_lq)
        output = output[..., : h_old * SCALE, : w_old * SCALE]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        sr = (output * 255.0).round().astype(np.uint8)

        row = calc_full_metrics(sr, gt)
        rows.append(row)
        print(
            f"  {gt_path.name}: RGB PSNR {row[0]:.4f}, RGB SSIM {row[1]:.6f}, "
            f"Y PSNR {row[2]:.4f}, Y SSIM {row[3]:.6f}"
        )

    return print_metric_average("SwinIR-M Teacher", rows)


def count_params() -> None:
    student = build_student()
    teacher = build_teacher()
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Baseline/FAKD student architecture: {student_params:,} parameters ({student_params / 1e6:.3f}M)")
    print(f"SwinIR-M teacher architecture: {teacher_params:,} parameters ({teacher_params / 1e6:.3f}M)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce verified no-save Set5 x4 metrics.")
    parser.add_argument(
        "command",
        choices=["students-rgb", "saved-results", "teacher", "params", "all"],
        help="Metric/count group to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model evaluation: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.set_grad_enabled(False)

    if args.command in {"students-rgb", "all"}:
        eval_training_rgb(device)
    if args.command in {"saved-results", "all"}:
        eval_saved_results()
    if args.command in {"teacher", "all"}:
        eval_teacher(device)
    if args.command in {"params", "all"}:
        count_params()


if __name__ == "__main__":
    main()
