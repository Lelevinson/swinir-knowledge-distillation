import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION: MARATHON BATCH HUNTER
# ==========================================
# 1. Ground Truth Path
dir_gt = r'testsets\Set5\HR'

# 2. Model A (Baseline - 500k)
dir_old = r'results\swinir_classical_sr_x4_model_A_500k' # (Assuming this is your folder name)

# 3. Model C (Ours - 500k)
dir_new = r'results\swinir_classical_sr_x4_model_C_500k' # (Assuming this is your folder name)

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

print("Starting FINAL Comparison (500k Models)...")

for img_name in img_names:
    print(f"Processing {img_name}...")
    
    # --- LOAD IMAGES ---
    # Note: Using the filenames from the TEST script, not the training logs
    p_gt = os.path.join(dir_gt, f"{img_name}.png")
    p_old = os.path.join(dir_old, f"{img_name}_SwinIR.png")
    p_new = os.path.join(dir_new, f"{img_name}_SwinIR.png")

    img_gt = cv2.imread(p_gt)
    img_old = cv2.imread(p_old)
    img_new = cv2.imread(p_new)

    if img_gt is None or img_old is None or img_new is None:
        print(f"  -> Skipping {img_name}: Image not found. Check paths.")
        continue

    # --- CALCULATE & FIND BEST PATCH ---
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY).astype(np.float32)

    abs_diff_old = cv2.absdiff(gray_gt, gray_old)
    abs_diff_new = cv2.absdiff(gray_gt, gray_new)
    improvement_map = abs_diff_old - abs_diff_new

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

    # --- EXTRACT CROPS & VISUALIZE ---
    y, x = best_y, best_x
    crop_gt = img_gt[y:y+patch_size, x:x+patch_size]
    crop_old = img_old[y:y+patch_size, x:x+patch_size]
    crop_new = img_new[y:y+patch_size, x:x+patch_size]
    
    crop_err_old = abs_diff_old[y:y+patch_size, x:x+patch_size]
    crop_err_new = abs_diff_new[y:y+patch_size, x:x+patch_size]
    crop_imp = improvement_map[y:y+patch_size, x:x+patch_size]

    psnr_old = get_y_psnr(crop_gt, crop_old)
    psnr_new = get_y_psnr(crop_gt, crop_new)
    gain = psnr_new - psnr_old

    amp = 10
    vis_err_old = cv2.applyColorMap(np.clip(crop_err_old * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_err_new = cv2.applyColorMap(np.clip(crop_err_new * amp, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis_imp = np.clip(crop_imp * 10 + 128, 0, 255).astype(np.uint8)
    vis_imp = cv2.applyColorMap(vis_imp, cv2.COLORMAP_JET)

    # Convert to RGB
    crop_gt, crop_old, crop_new = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in [crop_gt, crop_old, crop_new]]
    vis_err_old, vis_err_new, vis_imp = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in [vis_err_old, vis_err_new, vis_imp]]

    # --- PLOT ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(crop_gt); axs[0, 0].set_title(f"Ground Truth\n({img_name})"); axs[0, 0].axis('off')
    axs[0, 1].imshow(crop_old); axs[0, 1].set_title(f"Model A (500k)\n{psnr_old:.2f} dB"); axs[0, 1].axis('off')
    axs[0, 2].imshow(crop_new); axs[0, 2].set_title(f"Model C (500k)\n{psnr_new:.2f} dB", color='green', fontweight='bold'); axs[0, 2].axis('off')
    axs[1, 0].imshow(vis_imp); axs[1, 0].set_title(f"Improvement Map\nRed = C is Better"); axs[1, 0].axis('off')
    axs[1, 1].imshow(vis_err_old); axs[1, 1].set_title(f"Error Map: Model A"); axs[1, 1].axis('off')
    axs[1, 2].imshow(vis_err_new); axs[1, 2].set_title(f"Error Map: Model C"); axs[1, 2].axis('off')

    plt.tight_layout()
    save_path = f'figs/final_evidence_{img_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {save_path} (Patch Gain: {gain:+.2f} dB)")

print("Done.")