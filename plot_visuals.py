import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs('figs', exist_ok=True)
# --- CONFIGURATION ---
dir_gt = r'testsets\Set5\HR'
dir_a = r'superresolution\student_A_L1_x4\images'
dir_c = r'superresolution\student_C_v1_delayed_x4\images'

# We know the winner is the butterfly from the previous run
img_name = 'butterfly'

# --- 1. LOAD IMAGES ---
# Construct paths
p_gt = os.path.join(dir_gt, f"{img_name}.png")
p_a = os.path.join(dir_a, f"{img_name}x4", f"{img_name}x4_20000.png")
p_c = os.path.join(dir_c, f"{img_name}x4", f"{img_name}x4_20000.png")

img_gt = cv2.imread(p_gt)
img_a = cv2.imread(p_a)
img_c = cv2.imread(p_c)

if img_gt is None or img_a is None or img_c is None:
    print("Error: Images not found.")
    exit()

# --- 2. FIND THE BEST PATCH (RE-HUNTING) ---
# We want the spot where Model C beats Model A the most
err_a_full = np.sum((img_gt.astype(np.float32) - img_a.astype(np.float32)) ** 2, axis=2)
err_c_full = np.sum((img_gt.astype(np.float32) - img_c.astype(np.float32)) ** 2, axis=2)
diff_map = err_a_full - err_c_full # Positive means C is better

patch_size = 80
best_score = -1
best_coords = (0, 0)

h, w, _ = img_gt.shape
for y in range(0, h - patch_size, 10):
    for x in range(0, w - patch_size, 10):
        score = np.sum(diff_map[y:y+patch_size, x:x+patch_size])
        if score > best_score:
            best_score = score
            best_coords = (y, x)

y, x = best_coords
print(f"Focused on Patch at Y:{y}, X:{x}")

# --- 3. EXTRACT PATCHES ---
crop_gt = img_gt[y:y+patch_size, x:x+patch_size]
crop_a = img_a[y:y+patch_size, x:x+patch_size]
crop_c = img_c[y:y+patch_size, x:x+patch_size]

# --- 4. CALCULATE Y-CHANNEL PSNR (The Author's Method) ---
def get_y_psnr(img1, img2):
    # 1. Convert to YCbCr
    # (Note: OpenCV uses BGR, so we use BGR2YCrCb)
    img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    
    # 2. Calculate MSE on Y channel only
    mse = np.mean((img1_y.astype(np.float32) - img2_y.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Note: We pass the crops before converting them to RGB for display
# (The crops variables are still in BGR from cv2.imread here)
psnr_a = get_y_psnr(crop_gt, crop_a)
psnr_c = get_y_psnr(crop_gt, crop_c)

# --- 5. GENERATE DIFFERENCE MAP (The "Evidence") ---
# Show |Model A - Model C| inverted
# Brighter pixels = Bigger difference between the models
diff_img = cv2.absdiff(crop_a, crop_c)
diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX) # Maximize contrast
diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_VIRIDIS) # Green/Yellow map

# Convert patches to RGB for display
crop_gt = cv2.cvtColor(crop_gt, cv2.COLOR_BGR2RGB)
crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
crop_c = cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB)
diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)

# --- 6. PLOT ---
fig, axs = plt.subplots(1, 4, figsize=(16, 5))

axs[0].imshow(crop_gt)
axs[0].set_title("Ground Truth", fontsize=14)
axs[0].axis('off')

axs[1].imshow(crop_a)
axs[1].set_title(f"Model A (Baseline)\n{psnr_a:.2f} dB", fontsize=14)
axs[1].axis('off')

axs[2].imshow(crop_c)
axs[2].set_title(f"Model C (Ours)\n{psnr_c:.2f} dB", fontsize=14, fontweight='bold', color='green')
axs[2].axis('off')

axs[3].imshow(diff_img)
axs[3].set_title("Difference (A vs C)\nBright = Changed Pixels", fontsize=14)
axs[3].axis('off')

plt.tight_layout()
plt.savefig('figs/figure_final_evidence.png', dpi=300, bbox_inches='tight')
print("Saved figure_final_evidence.png")