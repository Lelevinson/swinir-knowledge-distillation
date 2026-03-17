import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Resolve paths relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Import your models
from models.network_swinir_student import SwinIR_Student
from models.network_swinir import SwinIR 

# ==========================================
# CONFIGURATION
# ==========================================
# Paths to the weights
path_model_student = r'student_weights/model_C_500k.pth'
path_model_teacher = r'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'

# Test Image (Low Resolution input)
path_lq = r'testsets/Set5/LR_bicubic/X4/butterflyx4.png'

# Output
output_file = 'figs/feature_activations.png'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dictionary to hold the extracted features
features = {}

# ==========================================
# 1. HELPER: HOOK FUNCTION
# ==========================================
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()
    return hook

# ==========================================
# MAIN EXECUTION
# ==========================================
print("Generating Feature Map Activations...")
os.makedirs('figs', exist_ok=True)

if not os.path.exists(path_lq):
    print(f"Error: Test image not found at {path_lq}")
    sys.exit()

# 1. Load the Image
img_lq = cv2.imread(path_lq, cv2.IMREAD_COLOR).astype(np.float32) / 255.
img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1)) # BGR to RGB, HWC to CHW
img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

# Get dynamic height and width to reshape the transformer tokens later
_, _, H, W = img_lq.shape 

# 2. Initialize Models
teacher = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                 img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                 num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                 upsampler='pixelshuffle', resi_connection='1conv')

student = SwinIR_Student(upscale=4, in_chans=3, img_size=64, window_size=8,
                         img_range=1., depths=[4, 4, 4, 4], embed_dim=60, 
                         num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                         upsampler='pixelshuffle', resi_connection='1conv')

# Load Teacher Weights Safely
print("Loading Teacher weights...")
ckpt_teacher = torch.load(path_model_teacher, map_location=device, weights_only=True)
if 'params' in ckpt_teacher:
    teacher.load_state_dict(ckpt_teacher['params'], strict=True)
elif 'params_ema' in ckpt_teacher:
    teacher.load_state_dict(ckpt_teacher['params_ema'], strict=True)
else:
    teacher.load_state_dict(ckpt_teacher, strict=True)

# Load Student Weights Safely
print("Loading Student weights...")
ckpt_student = torch.load(path_model_student, map_location=device, weights_only=True)
param_key_student = 'params' if 'params' in ckpt_student else 'params_ema'
if param_key_student in ckpt_student:
    student.load_state_dict(ckpt_student[param_key_student], strict=True)
else:
    student.load_state_dict(ckpt_student, strict=True)

teacher.eval().to(device)
student.eval().to(device)

# 3. Register Hooks to the 4th RSTB Block (Index 3)
teacher.layers[3].register_forward_hook(get_features('teacher_block_4'))
student.layers[3].register_forward_hook(get_features('student_block_4'))

# 4. Run Forward Pass
print("Running inference to extract features...")
with torch.no_grad():
    _ = teacher(img_lq)
    _ = student(img_lq)

# 5. Process Heatmaps
# Transformer output shape is [Batch, H*W, Channels]
# We average across the Channels (dim=2), which leaves us with shape [Batch, H*W]
teacher_mean = torch.mean(features['teacher_block_4'], dim=2).squeeze()
student_mean = torch.mean(features['student_block_4'], dim=2).squeeze()

# Reshape the 1D sequence back into a 2D [H, W] image grid
teacher_heatmap = teacher_mean.view(H, W).numpy()
student_heatmap = student_mean.view(H, W).numpy()

# 6. Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Teacher Plot
im1 = axes[0].imshow(teacher_heatmap, cmap='viridis')
axes[0].set_title('Teacher RSTB 4 Activations\n(Mean of 180 Channels)', fontsize=14)
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Student Plot
im2 = axes[1].imshow(student_heatmap, cmap='viridis')
axes[1].set_title('Student RSTB 4 Activations\n(Mean of 60 Channels)', fontsize=14)
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Success! Feature activations saved to {output_file}")