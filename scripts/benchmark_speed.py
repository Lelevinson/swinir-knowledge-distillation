import sys
import os

# Resolve paths relative to project root and add to Python path for model imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
import time
import numpy as np

# Import Architectures
from models.network_swinir import SwinIR as TeacherNet
from models.network_swinir_student import SwinIR_Student as StudentNet

# --- CONFIGURATION ---
# Path to Teacher (Official SwinIR-M)
teacher_path = r'model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'

# Path to Student (Your 165k Checkpoint)
# Adjust this path if your folder name is different!
student_path = r'superresolution/student_C_500k_marathon/models/165000_E.pth'

# Image Settings
# We use a 64x64 input (which becomes 256x256 output at x4)
# This simulates a typical patch processing size.
INPUT_SIZE = (1, 3, 64, 64) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOPS = 50  # How many times to run to get an average

def load_teacher():
    print("Loading Teacher...")
    model = TeacherNet(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                       num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                       upsampler='pixelshuffle', resi_connection='1conv')
    
    if os.path.exists(teacher_path):
        state = torch.load(teacher_path, map_location=DEVICE)
        # Handle 'params' key if it exists
        if 'params' in state: state = state['params']
        model.load_state_dict(state, strict=True)
    else:
        print(f"Warning: Teacher path not found at {teacher_path}. Testing with random weights.")
    
    return model.to(DEVICE).eval()

def load_student():
    print("Loading Student...")
    model = StudentNet(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[4, 4, 4, 4], embed_dim=60, 
                       num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                       upsampler='pixelshuffle', resi_connection='1conv')
    
    if os.path.exists(student_path):
        state = torch.load(student_path, map_location=DEVICE)
        if 'params' in state: state = state['params']
        model.load_state_dict(state, strict=True)
    else:
        print(f"Warning: Student path not found at {student_path}. Testing with random weights.")

    return model.to(DEVICE).eval()

def measure_speed(model, name):
    # Create dummy input
    input_tensor = torch.randn(INPUT_SIZE).to(DEVICE)
    
    # Warmup (Get the GPU ready)
    print(f"Warming up {name}...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measurement
    print(f"Benchmarking {name} ({LOOPS} runs)...")
    
    # Synchronize for GPU accuracy
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(LOOPS):
            _ = model(input_tensor)
            
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = (total_time / LOOPS) * 1000 # Convert to milliseconds
    
    print(f"  -> Average Inference Time: {avg_time:.2f} ms")
    return avg_time

if __name__ == '__main__':
    print(f"Benchmarking on device: {DEVICE}")
    print("-" * 30)
    
    teacher = load_teacher()
    t_time = measure_speed(teacher, "Teacher (SwinIR-M)")
    
    print("-" * 30)
    
    student = load_student()
    s_time = measure_speed(student, "Student (Ours)")
    
    print("-" * 30)
    print("RESULTS:")
    print(f"Teacher: {t_time:.2f} ms")
    print(f"Student: {s_time:.2f} ms")
    
    speedup = t_time / s_time
    print(f"Speedup: {speedup:.2f}x FASTER")
    
    # Reduction in time percentage
    reduction = ((t_time - s_time) / t_time) * 100
    print(f"Latency Reduction: {reduction:.2f}%")
