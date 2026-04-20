# AGENTS.md

This repository is a course-project fork of SwinIR/KAIR for Feature-Aware
Knowledge Distillation (FAKD) on x4 single-image super-resolution. Treat this
file as the short handoff for coding agents.

## Current Project Claim

The supported claim is narrow and controlled:

- FAKD improves a same-capacity lightweight SwinIR-style student over a
  pixel-loss-only baseline on Set5 x4.
- The deployed/inference student architecture is unchanged by distillation.
- The project does not claim broad SOTA performance.

## Verified Local Artifacts

Checkpoints:

- Baseline student: `student_weights/model_A_500k.pth`
- FAKD student: `student_weights/model_C_500k.pth`
- SwinIR-M teacher: `model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`

Set5 data:

- HR: `testsets/Set5/HR`
- LR x4: `testsets/Set5/LR_bicubic/X4`

Source-of-truth reproducibility note:

- `docs/reproducibility_set5.md`

## Verified Numbers

Set5 x4 RGB validation-style PSNR:

- Baseline student: 30.4532 dB
- FAKD student: 30.5133 dB
- Gain: +0.0601 dB

Saved-output Set5 x4 Y-channel PSNR:

- Baseline student: 32.3705 dB
- FAKD student: 32.4083 dB
- Gain: +0.0378 dB

Parameter counts:

- Student architecture: 988,959 parameters
- Teacher architecture: 11,900,199 parameters

Latency, local RTX 4060 Laptop GPU, synthetic `1x3x64x64` LR input:

- Teacher: 40.13 ms mean
- FAKD student: 19.70 ms mean
- Speedup: 2.04x
- Latency reduction: 50.91%

## Preferred Commands

Use no-save helpers when possible:

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py students-rgb
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py saved-results
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py teacher
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py params
D:\Conda_Envs\swinir\python.exe scripts\benchmark_final_checkpoints.py
```

Regenerate current paper-facing figures. `plot_visuals.py` loads the final
student checkpoints directly and does not require existing `results/` folders:

```powershell
D:\Conda_Envs\swinir\python.exe scripts\plot_results.py
D:\Conda_Envs\swinir\python.exe scripts\plot_marathon_convergence.py
D:\Conda_Envs\swinir\python.exe scripts\plot_visuals.py
```

## Caveats

- The paper's 30.45/30.51 numbers are RGB validation-style PSNR.
- Y-channel Set5 evaluation also improves, but by a smaller +0.0378 dB.
- Set14, B100/BSD100, Urban100, and Manga109 are not locally available.
- Do not invent cross-dataset results.
- Latency is hardware-dependent. Cite the device and benchmark protocol.
- Do not retrain unless the user explicitly asks.
- Do not overwrite or delete checkpoints.

## Code Hygiene Notes

- Prefer `scripts/reproduce_set5_metrics.py` over ad hoc evaluation for
  paper numbers.
- Prefer `scripts/benchmark_final_checkpoints.py` for latency.
- `main_test_student.py` and `main_test_swinir.py` save generated images into
  `results/`; use helpers when metrics only are needed.
- For the local SwinIR-M x4 teacher checkpoint, use training patch size 64.
