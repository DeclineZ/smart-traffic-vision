# Validation and hardware checks

Use the production runner to establish count validity and freshness. The separate hardware benchmark helps diagnose scaling with the production model, tracker and decoder, but its synthetic stream loop does not establish production acceptance.

Save generated reports in ignored `output/` or an external evidence bundle.

## Production soak

A short replay check exercises the real model, per-camera trackers, analytics, health and main projection with two camera outages:

```bash
python -m tools.soak --minutes 5 --replay-fps 25 \
  --output-mode controller --allow-replay-controller \
  --outage east:60:15 --outage northeast:150:15 --out output/soak.json
```

The runner defaults to `models/yolo26s_thai_traffic_manual_v1_640.engine` (see TensorRT export below); pass `--model` to test another verified engine. Publishing stays in memory unless `--mqtt-broker` is explicitly supplied. Never route replay counts to a live signal controller.

Require zero processing errors, no unexplained required-camera suppression, bounded memory/history, and recovery after each injected outage. The dead camera must become unknown, never an invented zero; healthy cameras must stay valid. Northeast's upstream outage must not block the main projection.

Check p95/p99 observation age and suppression reasons alongside throughput. A fast average can hide a stall. Use the actual capture rate with `--replay-fps`; metadata can be wrong. Do not relax freshness limits to obtain a pass.

On field hardware, run for 24–72 hours with live sources:

```bash
python -m tools.soak --minutes 1440 --sources-file /path/to/sources.json \
  --output-mode controller --outage east:3600:60 --out output/field-soak.json
```

This still uses an in-memory publisher by default. Injected outages withhold decoded frames; separately disconnect cameras/network links to test transport recovery. Measure camera/NVR buffering, check host clock synchronisation, and include rush hour, rain and day/night transitions. Do not share the inference GPU with other demanding workloads.

## TensorRT export and parity

Build on the target GPU/runtime. Dynamic batching is the default and is needed as the number of available cameras changes:

```bash
python export_trt.py --model models/yolo26s_thai_traffic.pt --batch 8 --no-half
python -m tools.trt_parity --model models/yolo26s_thai_traffic.pt \
  --engine models/yolo26s_thai_traffic.engine --frames 40 --out output/trt-parity.json
```

For the 960 px model add `--imgsz 960 --fixed-size` (and `--workspace 4`): Ultralytics' dynamic export lets image height/width grow to at least twice `imgsz`, which at 960 px with 8 cameras needs more scratch memory than an 8 GB GPU has. `--fixed-size` keeps the batch dynamic (1-8) and caps height/width at `imgsz`; check it with `tools.trt_parity --imgsz 960`.

The first command builds FP32. FP16 is the export default when `--no-half` is omitted; some TensorRT/Ultralytics combinations need NVIDIA's model-optimisation package. Follow the export diagnostic and validate that environment before enabling it.

Keep the export metadata beside the engine and verify the source-model and engine hashes. Rebuild and recheck after changing the GPU/runtime, source weights, precision or input size.

Parity checks detection matching, class agreement, lane occupancy and warmed latency for batches of 1–5 cameras. Passing parity establishes agreement with PyTorch, not ground-truth count accuracy.

## GPU video decoding

`--decoder auto` (the default for `main.py run` and `main.py benchmark`) decodes each camera with NVIDIA NVDEC through the ffmpeg bundled in `imageio-ffmpeg` and resizes on the GPU to the detector's input width, so Python receives 640x360 frames instead of 1080p. Software decoding of eight 1080p H.264 streams costs ~11 ms of CPU per frame; this path ~1 ms. On the RTX 5060 laptop it is what lets the runner analyse 8 cameras at the full rate. `auto` falls back to OpenCV when no NVIDIA decoder is available; `--decoder nvdec` refuses to fall back; `--decoder opencv` forces the CPU path. Detections from downscaled frames are mapped back to camera pixels before lanes and gates.

The resize reproduces what Ultralytics does to a full frame (cv2 `INTER_LINEAR`), because that is what accuracy was validated on. Re-check after changing decoder, ffmpeg build, input size or camera codec:

```bash
python -m tools.decode_parity --model models/yolo26s_thai_traffic_manual_v1_640.engine \
  --frames 30 --val-dir data/manual_v1_dataset --out output/decode-parity.json
```

The acceptance test decodes each hand-labelled frame from its recording through the GPU path and scores it against the labels next to the original image (recall may not drop by more than 1 point; measured 2026-10-15: 81.1% vs 81.7%). Box-level agreement with OpenCV decoding is reported for information (~95-97% of boxes): hardware and software decoders differ by about one grey level, which flips some low-confidence boxes either way.

Each camera runs one ffmpeg process with its own CUDA context (~0.2 GB of GPU memory per camera).

## Labelled count accuracy

Use clips held out from training, with labels at agreed observation times:

```bash
python -m tools.eval_counts --payloads /path/to/payloads.jsonl \
  --truth /path/to/truth.json --out output/count-accuracy.json
```

See the tool's module documentation for label fields and alignment. Evaluate occupancy, queued count and gate flow separately, including empty lanes, motorcycles, occlusion, parked vehicles, shadows, rain and night footage. Review false zeros, false counts in empty lanes, bias, absolute error and unknown coverage. Agree acceptable error with the controller owners before enabling adaptive decisions.

## Transport and integration

[Deployment instructions](../deploy/README.md) describe the loopback RTSP/MQTT smoke tool. It verifies real OpenCV acquisition, camera reconnect and MQTT reconnect without using field infrastructure.

The full test suite checks the pinned main contract and stale fallback using actual controller source with I/O replaced. A real deployed backend/database/dashboard check remains separate. Follow the [system-owner plan](SYSTEM_OWNER_CHANGE_PLAN.md) for that integration and physical signal acceptance.

## Hardware profiler

Use the approved model explicitly:

```bash
python main.py benchmark --streams 1 2 4 5 8 --frame-skips 1 2 --duration 20 \
  --out-dir output/hardware --save-plots
```

Defaults match `main.py run`: the 640 TensorRT engine, batched mode, ByteTrack, conf 0.10, GPU decoding. Cameras are opened before timing starts. Use `--decoder opencv` to compare the CPU decoding path, `--display` to measure rendering overhead and `--unpaced` to explore maximum capacity. Close games and other GPU-heavy applications first: a running game roughly halved the 8-camera result on the development laptop. Reports show throughput, frame drops, stage timing, GPU memory/utilisation and CPU load. These are synthetic sizing measurements; final acceptance comes from the production soak, count accuracy and live transport checks above.
