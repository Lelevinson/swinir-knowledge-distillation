Here is the **Complete, Zero-Assumption Guide** for your Lab Computer session. This covers every keystroke from the moment you sit down to the moment you walk away.

You can print this or open it on your phone/laptop to follow along.

---

### **Phase 1: Getting the System Ready**

**1. Open the Terminal**

- Find the terminal icon on the Ubuntu desktop and click it.

**2. Check for Miniconda (The Environment Manager)**
Type this command:

```bash
conda --version
```

- **If it shows a version number (e.g., `conda 4.10.3`):** Skip to Step 4.
- **If it says `command not found`:** Proceed to Step 3.

**3. Install Miniconda (Only if Step 2 failed)**
Copy and paste these lines one by one:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
```

_(Now close the Terminal window and open a new one to refresh it)._

**4. Install `tmux` (The Safety Net)**
Type this to see if it is installed:

```bash
tmux -V
```

- **If it shows a version:** Great.
- **If not found:** Try installing it (you might need a password):
  ```bash
  sudo apt install tmux -y
  ```
  _(If you don't have the password, ask the lab admin "Please install tmux," or proceed without it, but be very careful not to close the window)._

---

### **Phase 2: Getting Your Project**

**5. Clone your Repository**

```bash
cd ~
git clone https://github.com/Lelevinson/SwinIR-Knowledge-Distillation.git
cd SwinIR-Knowledge-Distillation
```

**6. Create the "swinir" Environment**

```bash
conda create -n swinir python=3.8 -y
conda activate swinir
```

**7. Install Libraries (For RTX 4080)**

```bash
# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other required tools
pip install opencv-python matplotlib tqdm timm requests
```

---

### **Phase 3: Downloading the Heavy Files**

**8. Download the Teacher Model**

```bash
# Create the folder
mkdir -p model_zoo/swinir

# Download the file
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth -P model_zoo/swinir/
```

**9. Download the DIV2K Dataset**
This block creates the specific folder structure your JSON needs. Copy and run this exact block:

```bash
# Create directories
mkdir -p trainsets/trainH/DIV2K

# Enter directory
cd trainsets/trainH/DIV2K

# Download and Unzip High-Res Images
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip -q DIV2K_train_HR.zip

# Download and Unzip Low-Res Images
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
unzip -q DIV2K_train_LR_bicubic_X4.zip

# Delete zip files to save space
rm *.zip

# Return to the project main folder
cd ../../..
```

---

### **Phase 4: Running the Training (The Marathon)**

**10. Enter the `tmux` Safety Bubble**

```bash
tmux new -s training_session
```

_(Your terminal will clear. You are now inside the protected session)._

**11. Reactivate Conda (Just to be safe)**

```bash
conda activate swinir
```

**12. START THE TRAINING**

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_500k.json
```

**13. Verify it is working**

- Wait 1-2 minutes.
- Look for text saying `Epoch: 1, Iter: 1, Average PSNR...`.
- If the numbers are moving, it works.

---

### **Phase 5: Leaving the Lab**

**14. Detach from Tmux**

- Press and Hold **`Ctrl`**.
- Press **`B`**.
- Release both keys.
- Press **`D`**.
- _(You will exit the "Bubble". The training text disappears, but it is still running)._

**15. Log out**
You can now close the terminal window and log out of the computer.

---

### **Phase 6: Checking Progress (Next Day)**

**16. Log in and Open Terminal**

**17. Re-enter the Bubble**

```bash
tmux attach -t training_session
```

_(You will see the training log scrolling exactly where you left it)._

**18. To Leave Again:**
Repeat Step 14 (`Ctrl+B`, then `D`).

**19. To Stop Training (When finished):**

- Inside the tmux session, press `Ctrl + C` to stop the python script.
- Type `exit` to close the tmux session.
