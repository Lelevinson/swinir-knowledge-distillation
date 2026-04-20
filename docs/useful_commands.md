# Project Command Reference

Run commands from the repository root.

## Environment

Windows/Conda environment used for verified local runs:

```powershell
conda activate swinir
```

Most verified commands below use the explicit interpreter path:

```powershell
D:\Conda_Envs\swinir\python.exe
```

## Reproduce Verified Set5 Metrics

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py students-rgb
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py saved-results
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py teacher
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py params
```

Expected headline result:

- Baseline Student: 30.4532 RGB PSNR
- FAKD Student: 30.5133 RGB PSNR
- Gain: +0.0601 dB

## Benchmark Final Checkpoints

```powershell
D:\Conda_Envs\swinir\python.exe scripts\benchmark_final_checkpoints.py
```

Expected local RTX 4060 Laptop GPU headline result:

- Teacher: 40.13 ms
- FAKD student: 19.70 ms
- Speedup: 2.04x
- Latency reduction: 50.91%

## Regenerate Current Figures

```powershell
D:\Conda_Envs\swinir\python.exe scripts\plot_results.py
D:\Conda_Envs\swinir\python.exe scripts\plot_marathon_convergence.py
D:\Conda_Envs\swinir\python.exe scripts\plot_visuals.py
```

These update the current paper-facing figures in `figs/`. The visual comparison
script loads the final student checkpoints directly and does not require
pre-existing `results/` folders.

## Original Save-Image Evaluation

These scripts write result images into `results/`. Use the no-save helpers above
when metrics only are needed.

Baseline student:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_student.py --task classical_sr --scale 4 --model_path student_weights\model_A_500k.pth --folder_lq testsets\Set5\LR_bicubic\X4 --folder_gt testsets\Set5\HR
```

FAKD student:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_student.py --task classical_sr --scale 4 --model_path student_weights\model_C_500k.pth --folder_lq testsets\Set5\LR_bicubic\X4 --folder_gt testsets\Set5\HR
```

Teacher:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 64 --model_path model_zoo\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth --folder_lq testsets\Set5\LR_bicubic\X4 --folder_gt testsets\Set5\HR
```

## Training Commands

Do not retrain unless intentionally starting a new experiment. Final configs are
kept for reproducibility:

```powershell
D:\Conda_Envs\swinir\python.exe main_train_student.py --opt options\swinir\train_swinir_student_500k_A.json
D:\Conda_Envs\swinir\python.exe main_train_student.py --opt options\swinir\train_swinir_student_500k.json
```

The first command is the final L1-only baseline config. The second is the final
FAKD config.
