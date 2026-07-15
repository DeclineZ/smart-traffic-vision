# JETSON Deployment

Directory structure for Jetson Nano deployment.

## 📁 Structure

```
JETSON/
├── docs/                          # Documentation
│   ├── README.md                  # Original README
│   ├── DEPLOYMENT_GUIDE.md        # Deployment instructions
│   ├── TESTING_GUIDE.md           # Testing procedures
│   ├── CODE_CHANGES.md            # Change log
│   └── CROSS_PLATFORM_GUIDE.md    # Windows ↔ Jetson development
│
├── config/                        # Configuration & dependencies
│   ├── requirements.txt           # Shared dependencies
│   ├── requirements2.txt          # Jetson-specific dependencies
│   └── requirements_windows.txt   # Windows development dependencies
│
├── src/                           # Source code
│   ├── pipeline.py                # Main pipeline
│   ├── run_pipeline.py            # Pipeline entry point
│   ├── sort.py                    # Tracking/sorting logic
│   └── platform_config.py         # Cross-platform configuration
│
├── venv/                          # Virtual environment (ignore in git)
└── README.md                      # This file
```

## 🚀 Quick Start

### On Windows (Development)
```bash
pip install -r config/requirements_windows.txt
cd src
python run_pipeline.py
```

### On Jetson (Deployment)
```bash
pip install -r config/requirements2.txt
cd src
python run_pipeline.py
```

## 📖 Documentation

- 👉 **Start here**: [docs/README.md](docs/README.md)
- Deploy: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- Test: [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)
- Dev: [docs/CROSS_PLATFORM_GUIDE.md](docs/CROSS_PLATFORM_GUIDE.md)
- Changes: [docs/CODE_CHANGES.md](docs/CODE_CHANGES.md)

## 💡 Key Files

- **src/platform_config.py** - Use this to write cross-platform code
- **config/requirements_windows.txt** - Windows development setup
- **config/requirements2.txt** - Jetson production setup

## 🔗 Platform Detection

In any Python file, use `platform_config`:

```python
from src.platform_config import IS_JETSON, DEVICE

if IS_JETSON:
    # Jetson-specific code (GPIO, TensorRT)
    pass
else:
    # Windows development code
    pass

# Device-agnostic
model = model.to(DEVICE)  # Works on both platforms
```

## ✅ Workflow

1. **Develop on Windows** → Use `requirements_windows.txt`
2. **Test locally** → All core logic
3. **Push to Jetson** → Git pull + `pip install -r config/requirements2.txt`
4. **Deploy** → `python src/run_pipeline.py`
