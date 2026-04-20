# Setup Notes

This repository is a course-project fork for lightweight SwinIR x4
super-resolution with Feature-Aware Knowledge Distillation. The current
reproducible result is documented in `docs/reproducibility_set5.md`.

## Environment

The verified local runs used a Conda environment named `swinir` on Windows with
CUDA available. If you are on the same machine, activate it with:

```powershell
conda activate swinir
```

or call the interpreter directly:

```powershell
D:\Conda_Envs\swinir\python.exe
```

The checked-in `environment.yml` is a Windows-oriented environment export. On a
Linux lab machine, use it as a package reference rather than assuming it will be
portable without adjustment.

## Required Local Artifacts

Final student checkpoints:

- `student_weights/model_A_500k.pth`
- `student_weights/model_C_500k.pth`

Teacher checkpoint:

- `model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`

Set5 x4 evaluation data:

- `testsets/Set5/HR`
- `testsets/Set5/LR_bicubic/X4`

Training data, only needed if retraining:

- `trainsets/trainH/DIV2K/DIV2K_train_HR`
- `trainsets/trainH/DIV2K/DIV2K_train_LR_bicubic/X4`

## Downloading Teacher Models

The teacher model can be downloaded with the commands in:

- `docs/download_models.md`

At minimum, this project needs:

```text
001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
```

## Reproduce Results Without Saving Images

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py students-rgb
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py saved-results
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py teacher
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py params
D:\Conda_Envs\swinir\python.exe scripts\benchmark_final_checkpoints.py
```

## Project Structure

- `docs/reproducibility_set5.md`: source-of-truth result protocol.
- `student_weights/`: final packaged student checkpoints.
- `traininglogs/`: tracked logs for the reported training runs.
- `scripts/reproduce_set5_metrics.py`: no-save metric helper.
- `scripts/benchmark_final_checkpoints.py`: no-save latency helper.
- `scripts/plot_results.py`: redraws summary figures from verified values.
- `scripts/plot_marathon_convergence.py`: redraws the 500k convergence figure.
- `scripts/plot_visuals.py`: redraws final visual comparison examples directly
  from tracked final student checkpoints and Set5 images.
- `options/swinir/train_swinir_student_500k_A.json`: final baseline config.
- `options/swinir/train_swinir_student_500k.json`: final FAKD config.

## Notes On Generated Outputs

`results/`, `superresolution/`, `model_zoo/`, and `trainsets/` are ignored for
new generated files. Some final course-project artifacts, such as
`student_weights/`, `traininglogs/`, `testsets/Set5`, and selected `figs/`, are
tracked intentionally for reproducibility.
