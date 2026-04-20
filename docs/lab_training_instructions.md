# Lab Machine Notes

These notes are for intentionally running or reproducing the project on a lab
machine. Do not retrain the models unless a new experiment is required.

## Quick Verification

From the repository root:

```bash
conda activate swinir
python scripts/reproduce_set5_metrics.py students-rgb
python scripts/reproduce_set5_metrics.py params
python scripts/benchmark_final_checkpoints.py
```

Expected Set5 RGB headline:

- Baseline Student: 30.4532 dB
- FAKD Student: 30.5133 dB
- Gain: +0.0601 dB

## Linux Environment Sketch

If the Windows `environment.yml` is not portable to the lab machine, create a
fresh environment and install the required packages manually:

```bash
conda create -n swinir python=3.8 -y
conda activate swinir
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install opencv-python matplotlib tqdm timm requests
```

## Required Files

Teacher checkpoint:

```bash
mkdir -p model_zoo/swinir
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth -P model_zoo/swinir/
```

DIV2K training data is only needed for retraining:

```bash
mkdir -p trainsets/trainH/DIV2K
cd trainsets/trainH/DIV2K
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
unzip -q DIV2K_train_HR.zip
unzip -q DIV2K_train_LR_bicubic_X4.zip
rm *.zip
cd ../../..
```

## Optional Retraining Commands

Use `tmux` for long training runs:

```bash
tmux new -s swinir_training
conda activate swinir
```

Baseline 500k config:

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_500k_A.json
```

FAKD 500k config:

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_500k.json
```

Detach from `tmux` with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t swinir_training
```
