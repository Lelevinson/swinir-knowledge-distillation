import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION: MARATHON BATCH HUNTER
# ==========================================
# 1. Ground Truth Path
dir_gt = r'testsets\Set5\HR'

# 2. Old Model (165k) - "Baseline" for this comparison
dir_old = r'results\swinir_classical_sr_x4_165000_E'

# 3. New Model (540k) - "Ours" for this comparison
dir_new = r'results\swinir_classical_sr_x4_540000_E'

# The complete Set5 list
img_names = ['baby', 'bird', 'butterfly', 'head', 'woman']

# Output folder
os.makedirs('figs', exist_ok=True)

# Helper: Calculate Y-Channel PSNR (Scientific Standard)
def get_y_psnr(img1, img2):
    # Convert BGR to YCbCr and take Y channel (0)
    img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mse = np.mean((img1_y.astype(np.float32) - img2_y.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("Starting Comparison (Batch)...")

for img_name in img_names:
    print(f"Processing {img_name}...")
    
    # --- 1. LOAD IMAGES ---
    p_gt = os.path.join(dir_gt, f"{img_name}.png")
    p_old = os.path.join(dir_old, f"{img_name}_SwinIR.png")
    p_new = os.path.join(dir_new, f"{img_name}_SwinIR.png")

    img_gt = cv2.imread(p_gt)
    img_old = cv2.imread(p_old)
    img_new = cv2.imread(p_new)

    if img_gt is None or img_old is None or img_new is None:
        print(f"  -> Skipping {img_name}: Image not found. Check paths.")
        continue

    # --- 2. CALCULATE ERROR MAPS (FULL IMAGE) ---
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Absolute Error (|GT - Pred|)
    # This answers: "How wrong is the model?"
    abs_diff_old = cv2.absdiff(gray_gt, gray_old)
    abs_diff_new = cv2.absdiff(gray_gt, gray_new)

    # Improvement Map: (Error Old) - (Error New)
    # Positive Value = Old error was bigger = New is BETTER
    improvement_map = abs_diff_old - abs_diff_new

    # --- 3. FIND BEST PATCH ---
    # We hunt for the spot where 540k fixed the most errors
    patch_size = 80
    best_score = -999999
    best_y, best_x = 0, 0

    h, w = gray_gt.shape
    # Scan with step 10 for speed
    for y in range(0, h - patch_size, 10):
        for x in range(0, w - patch_size, 10):
            # Sum up the improvement pixels in this patch
            score = np.sum(improvement_map[y:y+patch_size, x:x+patch_size])
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    # --- 4. EXTRACT CROPS ---
    y, x = best_y, best_x
    crop_gt = img_gt[y:y+patch_size, x:x+patch_size]
    crop_old = img_old[y:y+patch_size, x:x+patch_size]
    crop_new = img_new[y:y+patch_size, x:x+patch_size]
    
    # Extract the error patches
    crop_err_old = abs_diff_old[y:y+patch_size, x:x+patch_size]
    crop_err_new = abs_diff_new[y:y+patch_size, x:x+patch_size]
    crop_imp = improvement_map[y:y+patch_size, x:x+patch_size]

    # Calculate Local PSNR for titles
    psnr_old = get_y_psnr(crop_gt, crop_old)
    psnr_new = get_y_psnr(crop_gt, crop_new)
    
    # Calculate GAIN here so it is defined for the print statement
    gain = psnr_new - psnr_old

    # --- 5. VISUALIZE MAPS ---
    amp = 10
    
    # Error Maps (Blue=Good, Red=Bad Error)
    vis_err_old = cv2.applyColorMap(np.clip(crop_err_old * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_err_new = cv2.applyColorMap(np.clip(crop_err_new * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Improvement Map (Red=Improvement, Blue=Degradation)
    vis_imp = np.clip(crop_imp * 10 + 128, 0, 255).astype(np.uint8)
    vis_imp = cv2.applyColorMap(vis_imp, cv2.COLORMAP_JET)

    # Convert all to RGB for Matplotlib
    crop_gt = cv2.cvtColor(crop_gt, cv2.COLOR_BGR2RGB)
    crop_old = cv2.cvtColor(crop_old, cv2.COLOR_BGR2RGB)
    crop_new = cv2.cvtColor(crop_new, cv2.COLOR_BGR2RGB)
    vis_err_old = cv2.cvtColor(vis_err_old, cv2.COLOR_BGR2RGB)
    vis_err_new = cv2.cvtColor(vis_err_new, cv2.COLOR_BGR2RGB)
    vis_imp = cv2.cvtColor(vis_imp, cv2.COLOR_BGR2RGB)

    # --- 6. PLOT ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # ROW 1: The Images
    axs[0, 0].imshow(crop_gt)
    axs[0, 0].set_title(f"Ground Truth\n({img_name})", fontsize=14)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(crop_old)
    axs[0, 1].set_title(f"165k Iterations\n{psnr_old:.2f} dB", fontsize=14)
    axs[0, 1].axis('off')

    # Color title Green if gain is positive
    t_color = 'green' if gain > 0 else 'red'
    axs[0, 2].imshow(crop_new)
    axs[0, 2].set_title(f"540k Iterations\n{psnr_new:.2f} dB", fontsize=14, fontweight='bold', color=t_color)
    axs[0, 2].axis('off')

    # ROW 2: The Error Analysis
    # Left: The Difference of Errors (Improvement)
    axs[1, 0].imshow(vis_imp)
    axs[1, 0].set_title(f"Improvement (Old Error - New Error)\nRed = 540k is Closer to GT", fontsize=12, fontweight='bold')
    axs[1, 0].axis('off')

    # Middle: Old Error
    axs[1, 1].imshow(vis_err_old)
    axs[1, 1].set_title(f"Error Map: 165k\n(Distance from GT)", fontsize=12)
    axs[1, 1].axis('off')

    # Right: New Error
    axs[1, 2].imshow(vis_err_new)
    axs[1, 2].set_title(f"Error Map: 540k\n(Distance from GT)", fontsize=12)
    axs[1, 2].axis('off')

    plt.tight_layout()
    save_path = f'figs/marathon_evidence_{img_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {save_path} (Patch Gain: {gain:+.2f} dB)")

print("Done! Check the 'figs/' folder.")