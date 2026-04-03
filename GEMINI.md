# Project Context: SwinIR Knowledge Distillation

## Project Overview
This project focuses on **Feature-Aware Knowledge Distillation** to compress the massive Transformer-based **SwinIR** model for classical image super-resolution (x4). It trains a lightweight "Student" model by aligning intermediate feature spaces with a pre-trained "Teacher" model, reducing model parameters by 92% (11.8M to 0.89M) and significantly improving latency for edge device deployment without sacrificing structural fidelity.

### Technologies
- **Python 3.8**
- **PyTorch** (CUDA 12.1)
- **Conda** for environment management

## Building and Running

### Environment Setup
1. Create the Conda environment from the provided configuration:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate swinir
   ```

### Training Models
Training runs are configured via JSON files in the `options/swinir/` directory.

- **Model A (Baseline - L1 Loss):**
  ```bash
  python main_train_student.py --opt options/swinir/train_swinir_student.json
  ```
- **Model B (Response-Based Distillation):**
  ```bash
  python main_train_student.py --opt options/swir/train_swinir_student_distill_response.json
  ```
- **Model C (Feature-Based Distillation - Proposed):**
  ```bash
  python main_train_student.py --opt options/swinir/train_swinir_student_distill_feature.json
  ```

### Testing Models
Use `main_test_student.py` to evaluate trained checkpoints.
```bash
python main_test_student.py \
  --model_path student_weights/model_C_500k.pth \
  --folder_gt testsets/Set5/HR \
  --folder_lq testsets/Set5/LR_bicubic/X4
```
Generate visual comparisons with: `python scripts/run_demo.py`

## Development Conventions & Guidelines
- **Experimental Logging:** All experimental plans, results, and observations should be documented in `research_log.md`.
- **Dataset Management:** Datasets (e.g., DIV2K) and pre-trained teacher models are not tracked by Git and must be downloaded manually into `trainsets/` and `model_zoo/` respectively (refer to `DOWNLOAD_MODELS.md` and `docs/setup.md`).
- **Code Organization:** 
  - `main_train_student.py` / `main_test_student.py`: Primary entry points.
  - `models/network_swinir.py`: Teacher blueprint.
  - `models/network_swinir_student.py`: Student blueprint.
  - `models/model_plain.py`: Core logic including distillation losses.
- **Git Workflow:** Follow standard Git workflow (`pull` -> `add` -> `commit` -> `push`) to maintain sync and avoid conflicts. Use descriptive commit messages.