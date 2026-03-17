import sys
import os

# Resolve paths relative to project root and add to Python path for model imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.network_swinir_student import SwinIR_Student

# ==========================================
# CONFIGURATION
# ==========================================
# Paths to the uploaded weights
path_model_a = r'student_weights/model_A_500k.pth'
path_model_c = r'student_weights/model_C_500k.pth'

# Test Image
path_gt = r'testsets/Set5/HR/butterfly.png'
path_lq = r'testsets/Set5/LR_bicubic/X4/butterflyx4.png'

# Output
output_file = 'figs/demo_result.png'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. HELPER: LOAD MODEL & RUN INFERENCE
# ==========================================
def get_prediction(model_path, lq_tensor):
    # Initialize Student Architecture (4 Blocks, 60 Dim)
    model = SwinIR_Student(upscale=4, in_chans=3, img_size=64, window_size=8,
                           img_range=1., depths=[4, 4, 4, 4], embed_dim=60, 
                           num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                           upsampler='pixelshuffle', resi_connection='1conv')
    
    # Load Weights
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
        
    checkpoint = torch.load(model_path, map_location=device)
    param_key = 'params' if 'params' in checkpoint else 'params_ema'
    if param_key in checkpoint:
        model.load_state_dict(checkpoint[param_key], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
        
    model.eval().to(device)
    
    # Inference
    with torch.no_grad():
        output = model(lq_tensor)
        
    return output

# ==========================================
# 2. HELPER: PROCESS IMAGES
# ==========================================
def tensor2img(tensor):
    img = tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0)) # CHW to HWC, RGB to BGR
    return (img * 255.0).round().astype(np.uint8)

# ==========================================
# MAIN EXECUTION
# ==========================================
print("Running Demo...")
os.makedirs('figs', exist_ok=True)

# 1. Prepare Data
if not os.path.exists(path_gt) or not os.path.exists(path_lq):
    print("Error: Test images not found. Please download datasets first (see docs/setup.md).")
    exit()

img_lq = cv2.imread(path_lq, cv2.IMREAD_COLOR).astype(np.float32) / 255.
img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1)) # BGR to RGB, HWC to CHW
img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

# 2. Run Models
print("Running Model A (Baseline)...")
out_a_tensor = get_prediction(path_model_a, img_lq)
print("Running Model C (Ours)...")
out_c_tensor = get_prediction(path_model_c, img_lq)

if out_a_tensor is None or out_c_tensor is None:
    exit()

# 3. Convert to Images
img_gt = cv2.imread(path_gt) # Load GT directly for comparison
img_a = tensor2img(out_a_tensor)
img_c = tensor2img(out_c_tensor)

# 4. Generate Error Maps (The Logic from plot_visuals.py)
print("Generating Visualization...")

# Crop to the wing detail
crop_y, crop_x, size = 150, 140, 80
crop_gt = img_gt[crop_y:crop_y+size, crop_x:crop_x+size]
crop_a = img_a[crop_y:crop_y+size, crop_x:crop_x+size]
crop_c = img_c[crop_y:crop_y+size, crop_x:crop_x+size]

# Calculate Grayscale Errors
gray_gt = cv2.cvtColor(crop_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_c = cv2.cvtColor(crop_c, cv2.COLOR_BGR2GRAY).astype(np.float32)

err_a = cv2.absdiff(gray_gt, gray_a)
err_c = cv2.absdiff(gray_gt, gray_c)
improvement = err_a - err_c # Positive = A is worse = C is Better

# Visualize
vis_imp = cv2.applyColorMap(np.clip(improvement * 10 + 128, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
vis_err_a = cv2.applyColorMap(np.clip(err_a * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
vis_err_c = cv2.applyColorMap(np.clip(err_c * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

# Plot
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Convert BGR to RGB for Matplotlib
def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

axs[0, 0].imshow(bgr2rgb(crop_gt)); axs[0, 0].set_title("Ground Truth"); axs[0, 0].axis('off')
axs[0, 1].imshow(bgr2rgb(crop_a)); axs[0, 1].set_title("Model A (Baseline)"); axs[0, 1].axis('off')
axs[0, 2].imshow(bgr2rgb(crop_c)); axs[0, 2].set_title("Model C (Ours)"); axs[0, 2].axis('off')

axs[1, 0].imshow(bgr2rgb(vis_imp)); axs[1, 0].set_title("Improvement Map\nRed = Ours is Better"); axs[1, 0].axis('off')
axs[1, 1].imshow(bgr2rgb(vis_err_a)); axs[1, 1].set_title("Error: Baseline"); axs[1, 1].axis('off')
axs[1, 2].imshow(bgr2rgb(vis_err_c)); axs[1, 2].set_title("Error: Ours"); axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Success! Demo result saved to {output_file}")
