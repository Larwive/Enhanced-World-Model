# 🚀 Google Colab Testing Guide

This guide explains how to test the Enhanced World Model improvements using Google Colab.

## Quick Start

### Option 1: Upload to GitHub (Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add improved world model with A2C"
   git push origin main
   ```

2. **Open in Colab:**
   - Go to [Google Colab](https://colab.research.google.com)
   - File → Open Notebook → GitHub
   - Enter your repository URL
   - Select `test_world_model.ipynb`

3. **Enable GPU:**
   - Runtime → Change runtime type → GPU → Save

4. **Run the notebook:**
   - Runtime → Run all
   - Or run cells one by one (Shift+Enter)

### Option 2: Upload Notebook Directly

1. **Download the notebook:**
   - The notebook is at `test_world_model.ipynb`

2. **Upload to Colab:**
   - Go to [Google Colab](https://colab.research.google.com)
   - File → Upload notebook
   - Select `test_world_model.ipynb`

3. **Upload your code:**
   Since the code won't be in GitHub, you'll need to either:
   - Zip the entire `src/` directory and upload it
   - Or modify the notebook to clone from your GitHub repo

4. **Enable GPU and run**

## What the Notebook Does

### 1. Setup (Cells 1-4)
- ✅ Checks for GPU availability
- ✅ Installs dependencies (gymnasium, tensorboard, etc.)
- ✅ Clones repository (if using GitHub)
- ✅ Imports all necessary modules

### 2. Quick Sanity Check (Cell 5)
- ✅ Verifies model can be created
- ✅ Tests forward pass
- ✅ Confirms planning is enabled

### 3. Train with A2C (Cell 6)
- 🏋️ Trains the improved world model
- 📊 Shows progress during training
- 💾 Saves checkpoints

**Default settings:**
- Environment: CartPole-v1 (change to CarRacing-v3 for visual tasks)
- Epochs: 50 (increase to 200+ for full training)
- Parallel envs: 4
- Learning rate: 3e-4

### 4. Evaluate (Cell 7)
- 📈 Runs 10 evaluation episodes
- 📊 Shows statistics (mean, max, min rewards)
- 📉 Plots reward distribution

### 5. Visualize (Cell 8)
- 🎮 Renders the trained agent
- 🖼️ Displays frames from an episode
- 📹 Shows agent behavior

### 6. Analysis (Cell 9)
- 🔍 Compares old vs new controller architectures
- 📐 Shows parameter counts
- 🏗️ Displays model structure

## Configuration

You can modify these variables in the notebook (Cell 4):

```python
ENV_NAME = "CartPole-v1"  # Environment to use
NUM_ENVS = 4              # Parallel environments
MAX_EPOCHS = 50           # Training epochs (increase for better results)
LEARNING_RATE = 3e-4      # Learning rate
N_STEPS = 128             # Steps per A2C update
PLANNING_HORIZON = 5      # Planning lookahead
```

### For Different Environments

**CartPole-v1** (Simple, discrete):
```python
ENV_NAME = "CartPole-v1"
MAX_EPOCHS = 100
LEARNING_RATE = 3e-4
N_STEPS = 128
```

**CarRacing-v3** (Complex, continuous, visual):
```python
ENV_NAME = "CarRacing-v3"
MAX_EPOCHS = 500
LEARNING_RATE = 1e-4
N_STEPS = 64
PLANNING_HORIZON = 3
```

**MountainCar-v0** (Sparse rewards):
```python
ENV_NAME = "MountainCar-v0"
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
N_STEPS = 256
```

## Expected Training Time

**On Colab GPU (T4):**

| Environment | Epochs | Time | Expected Reward |
|-------------|--------|------|-----------------|
| CartPole-v1 | 50 | ~5 min | 300-500 |
| CartPole-v1 | 200 | ~15 min | 400-500 (solved) |
| CarRacing-v3 | 100 | ~30 min | 50-200 |
| CarRacing-v3 | 500 | ~2.5 hrs | 300-600 |

**On Colab CPU:**
- 3-5x slower than GPU
- Not recommended for CarRacing (too slow)

## Downloading Results

### Save Checkpoints

```python
# In a notebook cell
from google.colab import files

# Download best model
files.download('./checkpoints/a2c_notebook_CartPole-v1_best.pt')

# Download final model
files.download('./checkpoints/a2c_notebook_CartPole-v1_final.pt')
```

### Save Plots

Right-click on any plot → Save image as...

## Troubleshooting

### Issue: "No module named 'WorldModel'"

**Solution:**
```python
import sys
sys.path.insert(0, './src')
```

Or make sure you're in the correct directory:
```python
import os
os.chdir('Enhanced-World-Model')
```

### Issue: CUDA out of memory

**Solution:**
Reduce batch size and parallel environments:
```python
NUM_ENVS = 2  # Instead of 4
N_STEPS = 64  # Instead of 128
```

### Issue: Training is too slow

**Solutions:**
1. Make sure GPU is enabled (Runtime → Change runtime type → GPU)
2. Reduce MAX_EPOCHS for quick testing
3. Use simpler environment (CartPole instead of CarRacing)

### Issue: Rewards stay low

**Solutions:**
1. Train for more epochs (increase MAX_EPOCHS)
2. Check the environment is solvable
3. Try pre-training vision/memory first (see next section)

## Pre-training (Advanced)

For complex environments like CarRacing, pre-training helps:

### Step 1: Pre-train Vision

```python
from pretrain import pretrain

# Freeze controller and memory
for param in model.controller.parameters():
    param.requires_grad = False
for param in model.memory.parameters():
    param.requires_grad = False

# Pre-train vision
pretrain(model, envs, max_iter=100, device=device,
         pretrain_vision=True, pretrain_memory=False)

# Save
model.save('./checkpoints/pretrained_vision.pt',
          envs.single_observation_space,
          envs.single_action_space)
```

### Step 2: Pre-train Memory

```python
# Unfreeze memory, keep controller frozen
for param in model.memory.parameters():
    param.requires_grad = True
for param in model.controller.parameters():
    param.requires_grad = False

pretrain(model, envs, max_iter=100, device=device,
         pretrain_vision=False, pretrain_memory=True)

model.save('./checkpoints/pretrained_vision_memory.pt',
          envs.single_observation_space,
          envs.single_action_space)
```

### Step 3: End-to-End A2C

```python
# Unfreeze all
for param in model.parameters():
    param.requires_grad = True

# Load pre-trained checkpoint
model.load('./checkpoints/pretrained_vision_memory.pt',
          obs_space=envs.single_observation_space,
          action_space=envs.single_action_space)

# Train with A2C
train_a2c(model, envs, max_epochs=500, ...)
```

## Tips for Best Results

### 1. Start Simple
- Begin with CartPole-v1 to verify everything works
- Then move to more complex environments

### 2. Monitor Training
Watch these metrics in the output:
- **Mean Reward**: Should increase over time
- **Policy Loss**: Should stabilize (not necessarily decrease)
- **Value Loss**: Should decrease
- **Memory Loss**: Should decrease (good dynamics learning)

### 3. Patience
- CartPole: 50-100 epochs
- CarRacing: 500+ epochs
- Don't expect instant results!

### 4. Use GPU
- Always enable GPU for faster training
- Runtime → Change runtime type → GPU

### 5. Save Frequently
- Notebook saves checkpoints automatically
- Download important checkpoints to avoid losing them

## Comparing with Baseline

To see the improvement, you can:

1. **Train old system** (without improvements):
   ```python
   model_old, envs_old = create_world_model(ENV_NAME, NUM_ENVS,
                                           use_improved_controller=False)
   # Train with legacy method
   ```

2. **Train new system** (with improvements):
   ```python
   model_new, envs_new = create_world_model(ENV_NAME, NUM_ENVS,
                                           use_improved_controller=True)
   # Train with A2C
   train_a2c(model_new, envs_new, ...)
   ```

3. **Compare results**:
   - Old system: ~20-50 reward on CartPole
   - New system: ~400-500 reward on CartPole (solves it!)

## Additional Resources

- **IMPROVEMENTS.md**: Detailed documentation of all changes
- **src/train_a2c.py**: A2C training algorithm implementation
- **src/controller/ImprovedDiscreteController.py**: New controller architecture

## Example: Full Training Run

Here's a complete example for CartPole:

```python
# 1. Create model
model, envs = create_world_model("CartPole-v1", num_envs=4,
                                 use_improved_controller=True)

# 2. Train with A2C
train_a2c(model, envs, max_epochs=200, n_steps=128,
         learning_rate=3e-4, device=device,
         save_path='./checkpoints/')

# 3. Evaluate
# ... run evaluation cells ...

# 4. Download best checkpoint
from google.colab import files
files.download('./checkpoints/a2c_notebook_CartPole-v1_best.pt')
```

Expected output after 200 epochs:
- Mean reward: 450-500
- Solves CartPole consistently!

## Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Review IMPROVEMENTS.md for detailed explanations
3. Ensure all dependencies are installed
4. Verify GPU is enabled for faster training

---

**Happy Training! 🚀**
