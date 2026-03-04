import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the folder exists
os.makedirs('figs', exist_ok=True)

# ==========================================
# DATA CONFIGURATION
# ==========================================

# 1. The Training Log Data (Iter 2k to 20k)
iterations = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

# Model A (Baseline) - Corrected first value
psnr_a = [26.97, 27.55, 27.91, 28.09, 28.37, 28.51, 28.58, 28.70, 28.78, 28.80]

# Model B (Response Distillation)
psnr_b = [27.20, 27.64, 27.94, 28.21, 28.40, 28.58, 28.64, 28.75, 28.80, 28.82]

# Model C (Feature Distillation)
psnr_c = [27.00, 27.65, 28.09, 28.43, 28.58, 28.79, 28.94, 29.05, 29.13, 29.17]

# 2. Efficiency Data
model_names = ['Bicubic', 'Student (Ours)', 'Teacher (SwinIR)']
params = [0, 0.89, 11.8] # Millions
# UPDATED: Student PSNR changed from 29.17 to 30.51 to match final results
psnr_scores = [28.42, 30.51, 32.40] 

# ==========================================
# PLOT 1: PARAMETER COMPARISON (For Section 3.1)
# ==========================================
plt.figure(figsize=(7, 5))
# Only compare Student vs Teacher for this one
bar_labels = ['Student', 'Teacher']
bar_values = [0.89, 11.8]
colors = ['#90EE90', '#D3D3D3'] # Light Green and Light Gray

bars = plt.bar(bar_labels, bar_values, color=colors, edgecolor='black')

# Add text on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height} M', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Model Size Comparison', fontsize=14)
plt.ylabel('Parameters (Millions)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figs/figure_params.png', dpi=300)
print("Saved figure_params.png (For Methodology)")

# ==========================================
# PLOT 2: THE RACE (Convergence Analysis)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(iterations, psnr_a, marker='o', linestyle=(0, (5, 10)), color='gray', label='Model A (Baseline)')
plt.plot(iterations, psnr_b, marker='^', linestyle=(0, (5, 5)), color='blue', label='Model B (Response KD)')
plt.plot(iterations, psnr_c, marker='*', linestyle='-', color='red', linewidth=2, label='Model C (Feature KD)')

plt.title('Training Convergence (Set5 x4)', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/figure_convergence.png', dpi=300)
print("Saved figure_convergence.png (For Results)")

# ==========================================
# PLOT 3: THE GOLDEN CHART (Efficiency)
# ==========================================
plt.figure(figsize=(8, 6))

colors_scatter = ['gray', 'red', 'blue']
for i in range(len(model_names)):
    plt.scatter(params[i], psnr_scores[i], s=150, color=colors_scatter[i], label=model_names[i], edgecolors='black')
    offset = 0.5 if i == 1 else -0.5
    plt.text(params[i], psnr_scores[i] + 0.1, f"{psnr_scores[i]} dB", ha='center', fontsize=10)

plt.title('Performance vs. Model Size', fontsize=14)
plt.xlabel('Parameters (Millions)', fontsize=12)
plt.ylabel('PSNR (dB) on Set5', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# UPDATED: Arrow target (xy) and text location (xytext) shifted up to match the 30.51 dB position
plt.annotate('92% Smaller!', xy=(0.89, 30.51), xytext=(4, 31.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig('figs/figure_efficiency.png', dpi=300)
print("Saved figure_efficiency.png (For Results)")