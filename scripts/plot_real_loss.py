import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Resolve paths relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)

# --- CONFIGURATION ---
# Path to your Sprint Log (Model C v1)
# Adjust this if your folder name is slightly different
log_path = r'superresolution/student_C_v1_delayed_x4/train.log'

# Output
os.makedirs('figs', exist_ok=True)

def parse_log(file_path):
    iters = []
    losses = []
    
    if not os.path.exists(file_path):
        print(f"Error: Log file not found at {file_path}")
        return [], []

    with open(file_path, 'r') as f:
        for line in f:
            # We look for lines containing "G_loss"
            if 'G_loss' in line and 'iter' in line:
                try:
                    # 1. Extract Iteration
                    # Pattern: iter:   200  OR iter: 10,000
                    iter_match = re.search(r'iter:\s*([\d,]+)', line)
                    if iter_match:
                        current_iter = int(iter_match.group(1).replace(',', ''))
                        iters.append(current_iter)
                    
                    # 2. Extract G_loss
                    # Pattern: G_loss: 1.234e-02
                    loss_match = re.search(r'G_loss:\s*([\d\.e\+\-]+)', line)
                    if loss_match:
                        current_loss = float(loss_match.group(1))
                        losses.append(current_loss)
                except Exception as e:
                    continue
                    
    return iters, losses

# --- MAIN EXECUTION ---
iters, losses = parse_log(log_path)

if len(iters) > 0:
    print(f"Found {len(iters)} data points.")
    
    # Optional: Moving Average Smoothing (Makes the graph look scientific, less noisy)
    def smooth(scalars, weight=0.8):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    losses_smooth = smooth(losses, weight=0.6) # Adjust weight (0-1) for smoothness

    # PLOT
    plt.figure(figsize=(8, 5))
    
    # Plot raw data faintly
    plt.plot(iters, losses, color='lightcoral', alpha=0.3, label='Raw Loss')
    
    # Plot smoothed data strongly
    plt.plot(iters, losses_smooth, color='red', linewidth=2, label='Smoothed Loss (Model C)')
    
    plt.title('Generator Loss Convergence (20k Sprint)', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Total Loss (G_loss)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/figure_real_loss.png', dpi=300)
    print("Saved figs/figure_real_loss.png")
    
else:
    print("No data found! Check the log_path variable.")
