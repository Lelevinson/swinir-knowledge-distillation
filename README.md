# Accelerating Lightweight Image Restoration via Feature-Aware Knowledge Distillation

**A Comparative Study on Compressing Swin Transformer Models for Classical Super-Resolution (x4)**

![Architecture Diagram](figs/figure_architecture.jpg)

## Overview

Transformer-based models, such as SwinIR, have achieved state-of-the-art performance in image restoration but suffer from high computational costs, limiting their deployment on edge devices. This project proposes a **Feature-Aware Knowledge Distillation** framework to compress the massive SwinIR model into a lightweight "Student" version without sacrificing structural fidelity.

Unlike conventional distillation methods that only align the final output, our approach aligns the intermediate feature spaces of the student and teacher using learnable projectors. This enables the effective transfer of intermediate structural knowledge, forcing the student to learn the feature extraction process rather than solely mimicking the result.

### Key Contributions

- **Massive Compression:** Successfully reduced model parameters by **92%** (11.8M to 0.89M). Network depth was also reduced from 6 to 4 Residual Swin Transformer Blocks.
- **Real-Time Efficiency:** The lightweight student model achieves a **2.14x speedup** and a **53.37% reduction in latency** (17.55 ms per forward pass compared to the teacher's 37.63 ms).
- **Superior Performance:** The Feature-Aware student achieved a PSNR of **30.51 dB** after 500,000 iterations, outperforming the baseline on the Set5 benchmark.

---

## Methodology

We investigated two training strategies to determine the most effective method for training lightweight Transformers:

1.  **Model A (Baseline):** Trained using standard L1 pixel loss.
2.  **Model C (Feature-Aware Distillation - Ours):** Trained using pixel loss + feature loss on intermediate layers. To address the channel dimensionality mismatch between the teacher (C=180) and student (C=60), we employ learnable linear projectors at each RSTB stage.

---

## Experimental Results

### 1. Quantitative Comparison

We evaluated the models on the Set5 benchmark dataset for x4 Super-Resolution after a full 500,000 iterations.

| Model       | Method                 | Parameters | Final PSNR (dB) | Improvement  |
| :---------- | :--------------------- | :--------- | :-------------- | :----------- |
| **Model A** | Baseline (L1)          | 0.89M      | 30.45           | -            |
| **Model C** | **Feature KD (Ours)**  | **0.89M**  | **30.51**       | **+0.06 dB** |
| _Teacher_   | _SwinIR-M (Reference)_ | _11.8M_    | _32.40_         | _Reference_  |

### 2. Training Dynamics & Stability

Our Feature-Aware method demonstrates superior learning efficiency over a long training cycle. The model maintains a consistent lead throughout the 500,000-iteration marathon. The crucial crossover occurs early in training (approx. 100k iterations), after which Model C sustains its advantage, validating the stability of the distillation strategy.

![Convergence Plot](figs/figure_marathon_convergence.png)

### 3. Model Efficiency

The primary goal of this research was to achieve high performance within a strict parameter budget for edge applications. The chart below visualizes the massive scale difference between the Teacher and the Student.

We successfully compressed the 11.8 Million parameter Teacher into a **0.89 Million parameter Student**, achieving a **92% reduction** in model size while maintaining the same hierarchical design.

![Efficiency Plot](figs/figure_efficiency.png)

---

## Visual Quality Analysis

Feature-aware distillation effectively guides the student to focus on structural fidelity. To demonstrate this, we performed a differential error analysis on the 'butterfly' image from Set5.

- **Top Row:** Visual reconstruction comparison.
- **Bottom Row (Error Maps):** The **Improvement Map (Bottom Left)** highlights exactly where Model C outperformed the Baseline. The **Red and Yellow** pixels indicate areas where the Feature-Aware model significantly reduced the reconstruction error. These improvements are concentrated along the structural edges of the wing.

![Visual Evidence](figs/final_evidence_butterfly.png)

---

## Installation and Usage

For detailed instructions on setting up the environment, downloading datasets, and running training or testing scripts, please refer to the **[Installation Guide](docs/setup.md)**.

### Quick Start

To see the results immediately, you can test our pre-trained models on the Set5 dataset.

**1. Test Model C (Our Best Model)**

```bash
python main_test_student.py \
  --model_path student_weights/model_C_500k.pth \
  --folder_gt testsets/Set5/HR \
  --folder_lq testsets/Set5/LR_bicubic/X4
```

**2. Generate Visual Comparison (Model A vs Model C)**
We provide a demo script that loads both models, runs inference on the 'butterfly' image, and generates the error analysis maps.

```bash
python scripts/run_demo.py
```

---

## Acknowledgements

This work is built upon the official [SwinIR](https://github.com/JingyunLiang/SwinIR) and [KAIR](https://github.com/cszn/KAIR) repositories. We thank the original authors for their open-source contributions to the image restoration community.
