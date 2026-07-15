# Testing Guide

Guide for testing the Jetson Nano pipeline on Windows before deployment.

---

## Environment Setup

### Windows (Recommended for development)

```bash
# Navigate to traffic directory (parent of JETSON)
cd c:\Users\sschw\schwynn\Work\Teacher-Supervised\traffic

# Activate Python environment
python -m venv test-env
test-env\Scripts\activate

# Install Windows dependencies
cd JETSON
pip install -r config/requirements_windows.txt
cd ..
```

### Using Conda (if available)

```bash
conda create -n traffic-test python=3.8
conda activate traffic-test
cd JETSON
pip install -r config/requirements_windows.txt
cd ..
```

---

## Testing the Pipeline

### 1. Verify Imports
```bash
# From traffic/ (parent directory)
python -c "
import sys
sys.path.insert(0, 'JETSON/src')
from platform_config import IS_JETSON, DEVICE
print(f'Running on Windows: {not IS_JETSON}')
print(f'Using device: {DEVICE}')
from sort import Sort
print('✓ All imports OK')
"
```

### 2. Run With Test Config
```bash
cd JETSON
python src/run_pipeline.py --config ../config/config_test.json
```

Monitor:
- Console output for errors
- FPS in terminal
- GPU/CPU usage (if available)

### 3. Check Outputs
Results saved in output directory specified in config:
- `result.txt` - FPS and timing metrics
- `tracking.json` - Tracking data
- `density.csv` - Detection density
- `heatmap/` - Visualization files

---

## Testing on Jetson Nano

### 1. Sync Code
```bash
# On Windows development machine
git add -A
git commit -m "Testing changes"
git push origin main

# On Jetson Nano
git pull origin main
cd JETSON
```

### 2. Install Jetson Dependencies
```bash
pip install -r config/requirements2.txt
```

### 3. Run Pipeline
```bash
python src/run_pipeline.py --config ../config/config_test.json

# Monitor GPU
nvidia-smi  # In another terminal
```

---

## Comparison: Windows vs Jetson

| Feature | Windows | Jetson |
|---------|---------|--------|
| Requirements file | `config/requirements_windows.txt` | `config/requirements2.txt` |
| CUDA available | Optional (CPU fallback) | Yes (required) |
| TensorRT | No | Yes |
| SORT tracker | ✅ Works | ✅ Works |
| Other trackers | ✅ Works | ❌ Not available |
| Platform detection | `IS_JETSON = False` | `IS_JETSON = True` |

---

## Troubleshooting

### Import Errors
Check platform detection works:
```bash
python -c "
import sys
sys.path.insert(0, 'JETSON/src')
from platform_config import IS_JETSON, DEVICE
print(f'Platform: {\"JETSON\" if IS_JETSON else \"WINDOWS\"}')
print(f'Device: {DEVICE}')
"
```

### PyTorch Version Issues
On Windows, use:
```bash
pip install torch==1.10.0 torchvision==0.11.1
```

On Jetson, verify:
```bash
python -c "import torch; print(f'torch: {torch.__version__}')"
```

### Config File Not Found
Ensure config path is relative to current directory:
```bash
# If running from JETSON/src
python run_pipeline.py --config ../../config/config_test.json

# If running from JETSON
python src/run_pipeline.py --config ../config/config_test.json
```

---

## Cleanup

Remove test environment:
```bash
# Deactivate and remove venv
test-env\Scripts\deactivate
Remove-Item -Recurse test-env
```

---

See also:
- [CROSS_PLATFORM_GUIDE.md](CROSS_PLATFORM_GUIDE.md) - Development workflow
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Jetson setup
