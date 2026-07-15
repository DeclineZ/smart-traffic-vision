# Windows Development & Jetson Deployment Guide

## Quick Start

### On Windows (Development/Debugging)
```bash
# Install Windows-compatible dependencies
pip install -r requirements_windows.txt

# Develop and test your code
python main.py
```

### On Jetson (Deployment)
```bash
# Install Jetson dependencies
pip install -r requirements2.txt

# Run your code
python run_pipeline.py
```

## File Structure

- `requirements_windows.txt` → Use on Windows for development
- `requirements2.txt` → Use on Jetson (already configured)
- `platform_config.py` → Helper module for cross-platform code

## How to Use `platform_config.py`

### Example 1: Optional GPIO/TensorRT features
```python
from platform_config import IS_JETSON, GPIO, HAS_TENSORRT, DEVICE

if IS_JETSON:
    # Only runs on Jetson
    GPIO.setmode(GPIO.BOARD)
    # Your Jetson GPIO code here
    
if HAS_TENSORRT:
    # Use optimized TensorRT models
    pass
else:
    # Use PyTorch/ONNX models
    pass
```

### Example 2: Device-agnostic model loading
```python
from platform_config import DEVICE
import torch

# Model automatically uses GPU if available, CPU otherwise
model = load_model()
model = model.to(DEVICE)

# Works on Windows (CPU) AND Jetson (GPU)
output = model(input_tensor)
```

### Example 3: Debug mode flag
```python
# Add this to your main.py or config
import platform_config

if not platform_config.IS_JETSON:
    print("Running in development mode on Windows")
    # Enable extra debugging, logging, visualization
    DEBUG = True
else:
    print("Running on Jetson production mode")
    DEBUG = False
```

## Workflow Recommendations

### Option 1: Develop on Windows → Push to Jetson (Recommended)
1. Write/test core logic on Windows using `requirements_windows.txt`
2. Wrap Jetson-specific code with `if IS_JETSON:` checks
3. Once stable, push to Jetson and install `requirements2.txt`
4. Test on real hardware

### Option 2: Parallel Development
1. Work on Windows for model/algorithm development
2. Use SSH to connect to Jetson and test in parallel
3. Keep files synced via git

## Important Notes

- **Torch Version**: `1.10.0` is compatible with Python 3.6+
- **Python 3.6 is old**: If possible, upgrade to Python 3.8+ for better library support
- **TensorRT**: Only available on Jetson; use PyTorch models on Windows
- **GPIO**: Only usable on Jetson; add checks before importing

## If Dependencies Break

```bash
# On Windows: See what conflicts
pip install -r requirements_windows.txt --dry-run

# On Jetson: See what conflicts  
pip install -r requirements2.txt --dry-run

# Use specific versions if needed
pip install numpy==1.19.4 torch==1.10.0 --upgrade
```

## Git Strategy

```bash
# Keep both requirements files in version control
git add requirements_windows.txt requirements2.txt platform_config.py
git commit -m "Add cross-platform support"

# When pushing to Jetson
git pull origin main
pip install -r requirements2.txt
python run_pipeline.py
```
