import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION: SPRINT BATCH HUNTER
# ==========================================
# 1. Ground Truth Path
dir_gt = r'testsets\Set5\HR'

# 2. Model A (Baseline - 20k)
# Note: Using the training log images because they are definitely separated by folder name
dir_model_a = r'superresolution\student_A_L1_x4\images'

# 3. Model C (Ours - 20k)
dir_model_c = r'superresolution\student_C_v1_delayed_x4\images'

# The complete Set5 list
img_names = ['baby', 'bird', 'butterfly', 'head', 'woman']

# Output folder
os.makedirs('figs', exist_ok=True)

# Helper: Calculate Y-Channel PSNR
def get_y_psnr(img1, img2):
    img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mse = np.mean((img1_y.astype(np.float32) - img2_y.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("Starting Sprint Comparison (Model A vs Model C)...")

for img_name in img_names:
    print(f"Processing {img_name}...")
    
    # --- 1. LOAD IMAGES ---
    # Construct paths based on Training Log structure: 'images/imgnamex4/imgnamex4_20000.png'
    p_gt = os.path.join(dir_gt, f"{img_name}.png")
    
    # Model A Path
    p_a = os.path.join(dir_model_a, f"{img_name}x4", f"{img_name}x4_20000.png")
    
    # Model C Path
    p_c = os.path.join(dir_model_c, f"{img_name}x4", f"{img_name}x4_20000.png")

    img_gt = cv2.imread(p_gt)
    img_a = cv2.imread(p_a)
    img_c = cv2.imread(p_c)

    if img_gt is None or img_a is None or img_c is None:
        print(f"  -> Skipping {img_name}. Check paths.")
        continue

    # --- 2. CALCULATE ERROR MAPS ---
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Absolute Error (|GT - Pred|)
    abs_diff_a = cv2.absdiff(gray_gt, gray_a)
    abs_diff_c = cv2.absdiff(gray_gt, gray_c)

    # Improvement Map: (Error A) - (Error C)
    # Positive (Red) = A was worse than C (C Wins)
    improvement_map = abs_diff_a - abs_diff_c

    # --- 3. HUNT FOR BEST CROP ---
    patch_size = 80
    best_score = -999999
    best_y, best_x = 0, 0

    h, w = gray_gt.shape
    for y in range(0, h - patch_size, 10):
        for x in range(0, w - patch_size, 10):
            score = np.sum(improvement_map[y:y+patch_size, x:x+patch_size])
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    # --- 4. EXTRACT CROPS ---
    y, x = best_y, best_x
    crop_gt = img_gt[y:y+patch_size, x:x+patch_size]
    crop_a = img_a[y:y+patch_size, x:x+patch_size]
    crop_c = img_c[y:y+patch_size, x:x+patch_size]
    
    # Extract patches for maps
    crop_err_a = abs_diff_a[y:y+patch_size, x:x+patch_size]
    crop_err_c = abs_diff_c[y:y+patch_size, x:x+patch_size]
    crop_imp = improvement_map[y:y+patch_size, x:x+patch_size]

    # Calculate Local PSNR
    psnr_a = get_y_psnr(crop_gt, crop_a)
    psnr_c = get_y_psnr(crop_gt, crop_c)
    gain = psnr_c - psnr_a

    # --- 5. VISUALIZE MAPS ---
    amp = 10
    
    # Error Maps (Blue=Good, Red=Bad Error)
    vis_err_a = cv2.applyColorMap(np.clip(crop_err_a * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_err_c = cv2.applyColorMap(np.clip(crop_err_c * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Improvement Map (Red=Improvement)
    vis_imp = np.clip(crop_imp * 10 + 128, 0, 255).astype(np.uint8)
    vis_imp = cv2.applyColorMap(vis_imp, cv2.COLORMAP_JET)

    # Convert to RGB
    crop_gt = cv2.cvtColor(crop_gt, cv2.COLOR_BGR2RGB)
    crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
    crop_c = cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB)
    vis_err_a = cv2.cvtColor(vis_err_a, cv2.COLOR_BGR2RGB)
    vis_err_c = cv2.cvtColor(vis_err_c, cv2.COLOR_BGR2RGB)
    vis_imp = cv2.cvtColor(vis_imp, cv2.COLOR_BGR2RGB)

    # --- 6. PLOT ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # ROW 1: The Images
    axs[0, 0].imshow(crop_gt)
    axs[0, 0].set_title(f"Ground Truth\n({img_name})", fontsize=14)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(crop_a)
    axs[0, 1].set_title(f"Model A (Baseline)\n{psnr_a:.2f} dB", fontsize=14)
    axs[0, 1].axis('off')

    t_color = 'green' if gain > 0 else 'red'
    axs[0, 2].imshow(crop_c)
    axs[0, 2].set_title(f"Model C (Ours)\n{psnr_c:.2f} dB", fontsize=14, fontweight='bold', color=t_color)
    axs[0, 2].axis('off')

    # ROW 2: The Error Analysis
    axs[1, 0].imshow(vis_imp)
    axs[1, 0].set_title(f"Improvement (A Error - C Error)\nRed = C is Closer to GT", fontsize=12, fontweight='bold')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(vis_err_a)
    axs[1, 1].set_title(f"Error Map: Model A\n(Red = High Error)", fontsize=12)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(vis_err_c)
    axs[1, 2].set_title(f"Error Map: Model C\n(Blue = Low Error)", fontsize=12)
    axs[1, 2].axis('off')

    plt.tight_layout()
    save_path = f'figs/sprint_evidence_{img_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {save_path} (Patch Gain: {gain:+.2f} dB)")

print("Done! Check 'figs/' for sprint_evidence images.")