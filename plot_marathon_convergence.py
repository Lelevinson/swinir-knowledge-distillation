import matplotlib.pyplot as plt
import re
import os

# --- CONFIGURATION ---
# Path to Model C Log (500k Marathon)
path_c = r'superresolution/student_C_500k_marathon/train.log'

# Path to Model A Log (500k Marathon) - LEAVE EMPTY FOR NOW
path_a = r'' 

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
            # Look for lines like: <epoch:..., iter: 165,000, Average PSNR : 30.33dB
            if 'Average PSNR' in line:
                try:
                    # Extract Iteration (Remove commas)
                    iter_match = re.search(r'iter:\s*([\d,]+)', line)
                    if iter_match:
                        current_iter = int(iter_match.group(1).replace(',', ''))
                    
                    # Extract PSNR
                    psnr_match = re.search(r'Average PSNR\s*:\s*([\d\.]+)', line)
                    if psnr_match:
                        current_psnr = float(psnr_match.group(1))
                        
                    # Append
                    iters.append(current_iter)
                    psnrs.append(current_psnr)
                except Exception as e:
                    print(f"Skipping line due to error: {line.strip()}")
                    continue
    return iters, psnrs

# --- 1. LOAD DATA ---
iters_c, psnrs_c = parse_psnr_log(path_c)
iters_a, psnrs_a = parse_psnr_log(path_a)

# --- 2. PLOT ---
plt.figure(figsize=(10, 6))

# Plot Model C (Red)
if len(iters_c) > 0:
    plt.plot(iters_c, psnrs_c, color='red', linewidth=2, label='Model C (Feature KD)')
    # Annotate the final point
    plt.plot(iters_c[-1], psnrs_c[-1], 'r*', markersize=10)
    plt.text(iters_c[-1], psnrs_c[-1] + 0.05, f"{psnrs_c[-1]:.2f} dB", ha='right', color='red', fontweight='bold')

# Plot Model A (Gray) - Will run later when you have the file
if len(iters_a) > 0:
    plt.plot(iters_a, psnrs_a, color='gray', linestyle='--', linewidth=2, label='Model A (Baseline)')
    plt.plot(iters_a[-1], psnrs_a[-1], 'o', color='gray')
    plt.text(iters_a[-1], psnrs_a[-1] - 0.1, f"{psnrs_a[-1]:.2f} dB", ha='right', color='gray')

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