# Smart Traffic Vision

Multi-camera vehicle detection and tracking for adaptive traffic control. The runner measures current lane occupancy, queued/moving/unknown vehicles, and directed gate crossings. Camera health and observation times travel with the measurements.

The primary model is `models/yolo26s_thai_traffic.pt`. ByteTrack is the default tracker. TensorRT is optional and must be built and verified on the deployment hardware.

## Guides

| Guide | Use |
|---|---|
| [Existing controller integration](docs/CONTROLLER_MAIN_COMPATIBILITY.md) | Wire format, producer cutover, freshness and fallback limitations |
| [Measurement contract](docs/CONTROLLER_CONTRACT.md) | Rich shadow/local payload fields and measurement meanings |
| [Calibration](docs/CALIBRATION_GUIDE.md) | Lane polygons, gates, references and rollback |
| [Validation and hardware](docs/HARDWARE_BENCHMARK_GUIDE.md) | Model parity, count accuracy, transport and soak checks |
| [Deployment](deploy/README.md) | Services, credentials, monitoring and maintenance |
| [Future system-owner changes](docs/SYSTEM_OWNER_CHANGE_PLAN.md) | Proposed controller/backend work requiring the owners' approval |

## Setup

Use a Python version supported by your CUDA/PyTorch build and a virtual environment. The current development lock was verified with Python 3.13.5:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux, activate with `source .venv/bin/activate`. Install CUDA-capable PyTorch and torchvision appropriate for the target GPU first, then install the application dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements-lock.txt` records the verified development environment, including optional tools. Validate a fresh target installation before deployment; it is not a portable guarantee for every GPU/runtime.

Obtain the approved weights separately; models and recordings are excluded from Git. Put recordings under `videos/` using the paths in the camera configs. Store live URLs and passwords outside the repository; see [the sources example](deploy/sources.example.json). Node.js is needed only for controller compatibility tests. FFmpeg, MediaMTX and `amqtt` are optional dependencies for the local transport smoke tool.

## Run

Replay all five recordings in shadow mode:

```bash
python main.py run --cameras all --display --replay-fps 25
```

Set replay FPS to the recording's actual capture rate. Some supplied files report suspicious 150 FPS metadata; replay speed affects observation age and motion classification.

Evaluate live cameras without updating the controller:

```bash
python main.py run --cameras all --sources-file /path/to/sources.json \
  --output-mode shadow --mqtt-broker mqtt://controller-host:1883 --display
```

For the existing controller main, use the complete calibrated stop-line profile:

```bash
python main.py run --cameras all --sources-file /path/to/sources.json \
  --output-mode controller --controller-config config/controller_main.json \
  --mqtt-broker mqtt://controller-host:1883 \
  --health-file /path/to/health.json --record-payloads /path/to/payloads.jsonl \
  --log-file /path/to/vision.log --display
```

Stop the intersection's mock generator and other count producers before cutover. Main's default Compose setup includes a mock producer. Recorded footage is rejected in controller mode unless the explicit offline-test override is used. Use shadow mode when evaluating alongside the current producer.

Choose another approved backend with `--model`; for example, `--model models/yolo26s_thai_traffic.engine`. Use `python main.py run --help` for thresholds, TLS, source timeouts and output options.

## Configuration and measurement rules

Camera configs hold source paths, physical camera identity, lane/gate geometry, queue settings, calibration and camera-health thresholds. Detector, tracker, MQTT and publication settings come from CLI/environment options; old ignored config sections have been removed from the bundled files.

- A vehicle belongs to the first polygon covering its road-contact point, normally the bottom-centre of its box. Overlapping lanes therefore need calibration review.
- Occupancy is queued + moving + unknown-state vehicles. Motion state uses elapsed time and perspective-normalised displacement; it is not a frame-count threshold.
- An unavailable camera produces unknown readings in the rich payload. Zero means an observed empty region.
- Main receives occupancy for all nine configured stop-line regions under its registered source `CAM-01`. Northeast upstream regions are excluded. Any invalid or stale required reading withholds the entire main snapshot.
- Live timestamps describe local frame acquisition. They cannot prove how long a camera/NVR buffered the image before delivery; measure that delay on site.
- Calibration is loaded at startup. Restart after saving or restoring it.

The existing controller dashboard retains the last counts during silence. Monitor vision health and delivery state as well as that dashboard; see [controller limitations](docs/CONTROLLER_MAIN_COMPATIBILITY.md).

## Developer checks

```bash
python -m unittest discover -s tests -v
python -m tools.validate_calibration
python main.py run --cameras all --output-mode controller --allow-replay-controller --check-config
```

The last command validates bundled recordings without starting cameras or MQTT; the replay override is only for this isolated check. The compatibility fixtures execute pinned controller main source with external I/O replaced. Keep these fixtures and regression tests when changing publication behavior.

Bundled geometry passes structural validation but has six overlap warnings. This is not physical calibration approval. E1/NE1 geometry, upstream ownership, labelled count accuracy and a 24–72 hour field soak still require commissioning.

## Operator tools

| Command | Purpose |
|---|---|
| `python main.py calibrate --video <source> --sec 0 --n 1` | Draw road polygons and directed gates |
| `python -m tools.calibration_history list <config>` | Inspect saved calibration revisions |
| `python -m tools.calibration_history restore <config> <revision>` | Restore a validated revision, then restart |
| `python -m tools.healthcheck --file <health.json>` | Check runner/camera health |
| `python -m tools.eval_counts --help` | Compare recorded measurements with labelled truth |
| `python -m tools.diagnostic_export --help` | Bundle measurements/configuration for an incident |
| `python main.py benchmark --help` | Profile hardware with the separate synthetic benchmark |

Save generated reports in ignored `output/` or an external evidence bundle. The repository keeps the instructions, tests and active calibration rather than an accumulating audit history.
