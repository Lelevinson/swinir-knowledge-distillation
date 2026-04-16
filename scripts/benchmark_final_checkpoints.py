"""Benchmark final paper checkpoints without saving outputs.

This script times the local SwinIR-M teacher and final 500k student checkpoints
on the same synthetic 64x64 LR input. It does not train, save images, or modify
checkpoints.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network_swinir import SwinIR  # noqa: E402
from models.network_swinir_student import SwinIR_Student  # noqa: E402

SCALE = 4
INPUT_SIZE = (1, 3, 64, 64)

TEACHER_CKPT = ROOT / "model_zoo" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
FAKD_CKPT = ROOT / "student_weights" / "model_C_500k.pth"
BASELINE_CKPT = ROOT / "student_weights" / "model_A_500k.pth"


def build_teacher() -> SwinIR:
    return SwinIR(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def build_student() -> SwinIR_Student:
    return SwinIR_Student(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[4, 4, 4, 4],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def load_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    return checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint


def load_model(label: str, model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"Loading {label}")
    print(f"  checkpoint: {checkpoint_path.relative_to(ROOT)}")
    model.load_state_dict(load_state(checkpoint_path), strict=True)
    model.eval()
    model.to(device)
    print(f"  parameters: {parameter_count(model):,}")
    return model


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    trials: int,
) -> tuple[list[float], float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)
        synchronize(device)

        trial_times = []
        for _ in range(trials):
            synchronize(device)
            start = time.perf_counter()
            for _ in range(repeats):
                model(input_tensor)
            synchronize(device)
            elapsed = time.perf_counter() - start
            trial_times.append((elapsed / repeats) * 1000.0)

    mean_ms = statistics.fmean(trial_times)
    median_ms = statistics.median(trial_times)
    return trial_times, mean_ms, median_ms


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark final SwinIR teacher/student checkpoints.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc. Default: auto.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup forwards per model. Default: 20.")
    parser.add_argument("--repeats", type=int, default=100, help="Timed forwards per trial. Default: 100.")
    parser.add_argument("--trials", type=int, default=3, help="Timed trials per model. Default: 3.")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only benchmark teacher and FAKD final student.",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.trials <= 0:
        raise ValueError("Require warmup >= 0, repeats > 0, and trials > 0.")

    device = resolve_device(args.device)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True

    print("Final checkpoint latency benchmark")
    print(f"Repository: {ROOT}")
    print(f"Device: {device_description(device)}")
    print(f"Input tensor: {INPUT_SIZE} LR patch")
    print(f"Warmup forwards: {args.warmup}")
    print(f"Repeats per trial: {args.repeats}")
    print(f"Trials: {args.trials}")
    print(f"CUDA synchronization: {'yes' if device.type == 'cuda' else 'not applicable'}")
    print()

    specs = OrderedDict(
        [
            ("Teacher SwinIR-M", (build_teacher, TEACHER_CKPT)),
            ("FAKD Student final 500k", (build_student, FAKD_CKPT)),
        ]
    )
    if not args.skip_baseline:
        specs["Baseline Student final 500k"] = (build_student, BASELINE_CKPT)

    input_tensor = torch.randn(INPUT_SIZE, device=device)
    results = OrderedDict()

    for label, (builder, checkpoint_path) in specs.items():
        model = load_model(label, builder(), checkpoint_path, device)
        trials_ms, mean_ms, median_ms = benchmark_model(
            model=model,
            input_tensor=input_tensor,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
        results[label] = {
            "params": parameter_count(model),
            "trials": trials_ms,
            "mean": mean_ms,
            "median": median_ms,
        }
        print(f"  trial latencies ms: {', '.join(f'{value:.2f}' for value in trials_ms)}")
        print(f"  mean latency: {mean_ms:.2f} ms")
        print(f"  median latency: {median_ms:.2f} ms")
        print()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    teacher_ms = results["Teacher SwinIR-M"]["mean"]
    fakd_ms = results["FAKD Student final 500k"]["mean"]
    speedup = teacher_ms / fakd_ms
    reduction = ((teacher_ms - fakd_ms) / teacher_ms) * 100.0

    print("Summary using mean latency")
    print("| Model | Checkpoint | Parameters | Mean latency (ms) | Median latency (ms) |")
    print("| --- | --- | ---: | ---: | ---: |")
    for label, data in results.items():
        checkpoint = specs[label][1].relative_to(ROOT)
        print(f"| {label} | {checkpoint} | {data['params']:,} | {data['mean']:.2f} | {data['median']:.2f} |")
    print()
    print(f"Teacher latency: {teacher_ms:.2f} ms")
    print(f"FAKD student latency: {fakd_ms:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Latency reduction: {reduction:.2f}%")

    if "Baseline Student final 500k" in results:
        baseline_ms = results["Baseline Student final 500k"]["mean"]
        baseline_speedup = teacher_ms / baseline_ms
        baseline_reduction = ((teacher_ms - baseline_ms) / teacher_ms) * 100.0
        print(f"Baseline student latency: {baseline_ms:.2f} ms")
        print(f"Baseline speedup: {baseline_speedup:.2f}x")
        print(f"Baseline latency reduction: {baseline_reduction:.2f}%")


if __name__ == "__main__":
    main()
