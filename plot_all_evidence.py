import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Use raw strings for Windows paths
dir_gt = r'testsets\Set5\HR'
dir_a = r'superresolution\student_A_L1_x4\images'
dir_c = r'superresolution\student_C_v1_delayed_x4\images'

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

print("Starting Batch Evidence Generation...")

for img_name in img_names:
    print(f"Processing {img_name}...")
    
    # 1. LOAD
    # Note: Folder structure might vary slightly based on previous runs. 
    # Adjusting to standard: 'images/namex4/namex4_20000.png'
    p_gt = os.path.join(dir_gt, f"{img_name}.png")
    p_a = os.path.join(dir_a, f"{img_name}x4", f"{img_name}x4_20000.png")
    p_c = os.path.join(dir_c, f"{img_name}x4", f"{img_name}x4_20000.png")

    img_gt = cv2.imread(p_gt)
    img_a = cv2.imread(p_a)
    img_c = cv2.imread(p_c)

    if img_gt is None or img_a is None or img_c is None:
        print(f"  -> Error: Could not find images for {img_name}. Skipping.")
        continue

    # 2. FIND BEST PATCH (Where C beats A the most)
    # Error = Squared difference from GT
    err_a_map = np.sum((img_gt.astype(np.float32) - img_a.astype(np.float32)) ** 2, axis=2)
    err_c_map = np.sum((img_gt.astype(np.float32) - img_c.astype(np.float32)) ** 2, axis=2)
    
    # Improvement Map: Positive means A had more error than C
    improvement_map = err_a_map - err_c_map

    patch_size = 80
    best_score = -999999
    best_y, best_x = 0, 0

    h, w, _ = img_gt.shape
    # Step by 10 for speed
    for y in range(0, h - patch_size, 10):
        for x in range(0, w - patch_size, 10):
            score = np.sum(improvement_map[y:y+patch_size, x:x+patch_size])
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    # 3. EXTRACT CROPS
    crop_gt = img_gt[best_y:best_y+patch_size, best_x:best_x+patch_size]
    crop_a = img_a[best_y:best_y+patch_size, best_x:best_x+patch_size]
    crop_c = img_c[best_y:best_y+patch_size, best_x:best_x+patch_size]

    # Calculate Local PSNR
    psnr_a = get_y_psnr(crop_gt, crop_a)
    psnr_c = get_y_psnr(crop_gt, crop_c)
    gain = psnr_c - psnr_a

    # 4. MAKE DIFFERENCE MAP (Visual Evidence)
    diff_img = cv2.absdiff(crop_a, crop_c)
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
    diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_VIRIDIS)

    # Convert all to RGB
    crop_gt = cv2.cvtColor(crop_gt, cv2.COLOR_BGR2RGB)
    crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
    crop_c = cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB)
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)

    # 5. PLOT
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    
    axs[0].imshow(crop_gt)
    axs[0].set_title(f"Ground Truth\n({img_name})", fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(crop_a)
    axs[1].set_title(f"Model A (Baseline)\n{psnr_a:.2f} dB", fontsize=14)
    axs[1].axis('off')

    # Color the title Green if C won, Red if not
    title_color = 'green' if gain > 0 else 'red'
    axs[2].imshow(crop_c)
    axs[2].set_title(f"Model C (Ours)\n{psnr_c:.2f} dB", fontsize=14, fontweight='bold', color=title_color)
    axs[2].axis('off')

    axs[3].imshow(diff_img)
    axs[3].set_title(f"Difference Map\nGain: {gain:+.2f} dB", fontsize=14)
    axs[3].axis('off')

    plt.tight_layout()
    save_path = f'figs/evidence_{img_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print(f"  -> Saved {save_path}")

print("Done! Check the 'figs' folder.")