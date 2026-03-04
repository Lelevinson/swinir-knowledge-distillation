import matplotlib.pyplot as plt
import re
import os

# Resolve paths relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

# --- CONFIGURATION ---
# Path to Model C Log
path_c = r'superresolution/student_C_500k_marathon/train.log'

# Path to Model A Log
path_a = r'superresolution/student_A_500k_marathon/train.log' 

# CUTOFF LIMIT (The Finish Line)
# We limit the graph to 550k so Model A's 3M run doesn't ruin the scale
MAX_ITER = 550000 

# Output Folder
os.makedirs('figs', exist_ok=True)

def parse_psnr_log(file_path):
    iters = []
    psnrs = []
    
    if not os.path.exists(file_path) or file_path == '':
        print(f"Log file not found or empty: {file_path}")
        return [], []

    print(f"Parsing {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            if 'Average PSNR' in line:
                try:
                    # Extract Iteration
                    iter_match = re.search(r'iter:\s*([\d,]+)', line)
                    if iter_match:
                        current_iter = int(iter_match.group(1).replace(',', ''))
                    
                    # Extract PSNR
                    psnr_match = re.search(r'Average PSNR\s*:\s*([\d\.]+)', line)
                    if psnr_match:
                        current_psnr = float(psnr_match.group(1))
                        
                    # DATA FILTERING: Stop if we go past the limit
                    if current_iter > MAX_ITER:
                        break # Stop reading this file
                        
                    iters.append(current_iter)
                    psnrs.append(current_psnr)
                except Exception as e:
                    continue
    return iters, psnrs

# --- 1. LOAD DATA ---
iters_c, psnrs_c = parse_psnr_log(path_c)
iters_a, psnrs_a = parse_psnr_log(path_a)

# --- 2. PLOT ---
plt.figure(figsize=(10, 6))

# Plot Model A (Gray) - Baseline
if len(iters_a) > 0:
    plt.plot(iters_a, psnrs_a, color='gray', linestyle='--', linewidth=2, label='Model A (Baseline)')
    # Label the last point
    last_iter_a = iters_a[-1]
    last_psnr_a = psnrs_a[-1]
    plt.plot(last_iter_a, last_psnr_a, 'o', color='gray')
    plt.text(last_iter_a, last_psnr_a - 0.15, f"{last_psnr_a:.2f} dB", ha='right', color='gray')

# Plot Model C (Red) - Ours
if len(iters_c) > 0:
    plt.plot(iters_c, psnrs_c, color='red', linewidth=2, label='Model C (Feature KD)')
    # Label the last point
    last_iter_c = iters_c[-1]
    last_psnr_c = psnrs_c[-1]
    plt.plot(last_iter_c, last_psnr_c, 'r*', markersize=12)
    plt.text(last_iter_c, last_psnr_c + 0.05, f"{last_psnr_c:.2f} dB", ha='right', color='red', fontweight='bold')

# Formatting
plt.title('Marathon Training Convergence (Set5 x4)', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
output_path = 'figs/figure_marathon_convergence.png'
plt.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path}")
