"""Count checkpoint-compatible SwinIR teacher/student parameters.

This utility mirrors the architecture used by the verified final checkpoints.
For the complete reproducibility workflow, prefer:

    python scripts/reproduce_set5_metrics.py params
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network_swinir import SwinIR as TeacherModel
from models.network_swinir_student import SwinIR_Student as StudentModel


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def build_teacher():
    return TeacherModel(
        upscale=4,
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


def build_student():
    return StudentModel(
        upscale=4,
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


def main():
    teacher_params = count_parameters(build_teacher())
    student_params = count_parameters(build_student())
    reduction = (1.0 - student_params / teacher_params) * 100.0

    print("--- Checkpoint-Compatible Model Size Comparison ---")
    print(f"Teacher Model Parameters: {teacher_params:,} ({teacher_params / 1e6:.3f}M)")
    print(f"Student Model Parameters: {student_params:,} ({student_params / 1e6:.3f}M)")
    print(f"Student parameter reduction vs teacher: {reduction:.2f}%")


if __name__ == "__main__":
    main()
