# Reproducibility Note: Verified Set5 x4 Results

This note documents the local evidence for the Set5 x4 numbers used in the
paper. It is intentionally narrow: no retraining is required, and the preferred
helper commands below do not overwrite checkpoints or result images.

## Local Artifacts

Checkpoints:

- Baseline Student: `student_weights/model_A_500k.pth`
- FAKD Student: `student_weights/model_C_500k.pth`
- SwinIR-M teacher: `model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`

Set5 x4 data:

- HR images: `testsets/Set5/HR`
- Bicubic LR images: `testsets/Set5/LR_bicubic/X4`

Existing student result images:

- Baseline Student: `results/swinir_classical_sr_x4_model_A_500k`
- FAKD Student: `results/swinir_classical_sr_x4_model_C_500k`

Additional benchmark datasets were not found locally during the audit:
Set14, BSD100/B100, Urban100, and Manga109.

## Preferred No-Save Commands

Use these commands from the repository root:

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py students-rgb
```

Expected output:

| Model | Set5 RGB PSNR |
| --- | ---: |
| Baseline Student | 30.4532 |
| FAKD Student | 30.5133 |
| Gain | +0.0601 |

These are the training-validation-equivalent RGB PSNR values supporting the
paper's rounded Set5 claim: 30.45 dB to 30.51 dB, a +0.06 dB gain.

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py saved-results
```

Expected output from the already generated student result images:

| Model | RGB PSNR | RGB SSIM | Y PSNR | Y SSIM |
| --- | ---: | ---: | ---: | ---: |
| Baseline Student | 30.4560 | 0.867104 | 32.3705 | 0.896912 |
| FAKD Student | 30.5026 | 0.868199 | 32.4083 | 0.897660 |
| Gain | +0.0466 | +0.001095 | +0.0378 | +0.000748 |

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py teacher
```

Expected output:

| Model | RGB PSNR | RGB SSIM | Y PSNR | Y SSIM |
| --- | ---: | ---: | ---: | ---: |
| SwinIR-M teacher | 30.9883 | 0.875844 | 32.9158 | 0.904364 |

```powershell
D:\Conda_Envs\swinir\python.exe scripts\reproduce_set5_metrics.py params
```

Expected output:

| Model | Parameters |
| --- | ---: |
| Baseline/FAKD student architecture | 988,959 |
| SwinIR-M teacher architecture | 11,900,199 |

The helper instantiates the checkpoint-compatible model definitions. The
standalone `utils/count_parameters.py` utility has been updated to use the same
constructors.

## Original Evaluation Commands

The original evaluation scripts write output images into `results/`. Running
them with the same checkpoint names can overwrite existing generated images.
Use the no-save helper above when only metrics are needed.

Baseline Student:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_student.py --model_path student_weights\model_A_500k.pth --folder_gt testsets\Set5\HR --folder_lq testsets\Set5\LR_bicubic\X4 --scale 4
```

FAKD Student:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_student.py --model_path student_weights\model_C_500k.pth --folder_gt testsets\Set5\HR --folder_lq testsets\Set5\LR_bicubic\X4 --scale 4
```

SwinIR-M teacher:

```powershell
D:\Conda_Envs\swinir\python.exe main_test_swinir.py --model_path model_zoo\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth --folder_gt testsets\Set5\HR --folder_lq testsets\Set5\LR_bicubic\X4 --scale 4 --training_patch_size 64
```

Important teacher caveat: the local SwinIR-M x4 checkpoint name and
checkpoint-compatible architecture use `s64`. Current project scripts default to
`--training_patch_size 64`; keep that value when evaluating this teacher.

## Latency Benchmark

Use this no-save command to benchmark the exact final paper checkpoints:

```powershell
D:\Conda_Envs\swinir\python.exe scripts\benchmark_final_checkpoints.py
```

Output from the local verification run:

- Device: CUDA, NVIDIA GeForce RTX 4060 Laptop GPU
- Input tensor: `(1, 3, 64, 64)` LR patch
- Warmup forwards per model: 20
- Timed repeats per trial: 100
- Trials per model: 3
- CUDA synchronization: yes

| Model | Latency |
| --- | ---: |
| SwinIR-M teacher | 40.13 ms mean, 40.12 ms median |
| FAKD Student final 500k | 19.70 ms mean, 19.65 ms median |
| Baseline Student final 500k | 20.28 ms mean, 20.10 ms median |
| Teacher vs FAKD speedup | 2.04x |
| Teacher vs FAKD latency reduction | 50.91% |

The benchmark also prints parameter counts:

| Model | Parameters |
| --- | ---: |
| SwinIR-M teacher | 11,900,199 |
| FAKD Student final 500k | 988,959 |
| Baseline Student final 500k | 988,959 |

Latency depends on hardware, CUDA/cuDNN state, driver state, input size, warmup
settings, repeat count, and power/thermal behavior. These numbers are safe to
cite as local RTX 4060 Laptop GPU timings for a synthetic 64x64 LR input patch,
not as hardware-independent latency. The FAKD and baseline students use the same
architecture, so their small latency difference should be treated as measurement
noise rather than a FAKD-specific speed claim.

## Protocol Caveats

- The paper's 30.45/30.51 dB Set5 numbers are supported by the
  training-validation-equivalent RGB PSNR protocol.
- The stronger comparison convention for SISR papers is usually Y-channel PSNR
  after border shaving. The saved student outputs also improve under Y-channel
  evaluation, but the local Y-channel gain is smaller: +0.0378 dB.
- Set14, BSD100/B100, Urban100, and Manga109 were not locally available, so no
  expanded benchmark table should be inserted unless those datasets are added
  and evaluated under the same protocol.
- The verified result supports a controlled same-backbone improvement from FAKD
  without adding inference-time model capacity. It does not support broad SOTA
  or cross-dataset superiority claims.
