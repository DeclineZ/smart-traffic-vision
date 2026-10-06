# Deploying the vision runner

This directory holds service definitions for a field PC. None of it has been run on the field hardware yet. Commission it with the checklist at the end.

The service example selects **controller** output for the existing main adapter.
It needs live sources and sends complete fresh stop-line occupancy using `CAM-01`;
no controller migration is needed. Set `VISION_OUTPUT_MODE=shadow` while evaluating
alongside an existing producer. At cutover stop mock/other counts for `INT-001`.
See [main compatibility and pilot limits](../docs/CONTROLLER_MAIN_COMPATIBILITY.md).

## Files

| File | Purpose |
|---|---|
| `traffic-vision.service` | systemd unit: restarts automatically, includes hardening, and writes rotating logs, a payload record and a health file |
| `traffic-vision-health.service` / `.timer` | Every 30 s, restarts the runner if its health file stops updating (hung process). Camera outages are reported, never "fixed" by restarts |
| `vision.env.example` | Model, identity and broker settings (`/etc/traffic-vision/vision.env`, mode 600) |
| `sources.example.json` | Camera stream URLs with credentials (`/etc/traffic-vision/sources.json`, mode 600). Credentials stay out of the process list and are redacted in logs and payloads |

## Linux install

```bash
sudo useradd --system --home /opt/smart-traffic-vision traffic
sudo mkdir -p /etc/traffic-vision && sudo chmod 750 /etc/traffic-vision
sudo cp deploy/vision.env.example /etc/traffic-vision/vision.env
sudo cp deploy/sources.example.json /etc/traffic-vision/sources.json
sudo chown -R traffic:traffic /etc/traffic-vision
sudo chmod 600 /etc/traffic-vision/*
sudo cp deploy/traffic-vision*.service deploy/traffic-vision-health.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Set the real RTSP URLs, broker credentials, model path and nonempty model hash
before enabling the service. The `traffic` user needs read/execute access to
`/opt/smart-traffic-vision`, its Python environment, models and calibration files.
Set the broker address/listener and credentials to match the existing deployment.
For an available TLS listener, copy its approved CA into `/etc/traffic-vision`,
give it the same ownership/permissions and set `MQTT_TLS_CA` in the environment.
The service no longer forces a nonexistent TLS listener or certificate path.
Broker access must permit the selected counts and health topics.

Install dependencies in a fresh environment as described in the [README](../README.md). The lock file records the verified development installation; validate the target GPU/runtime. The default deployment model is the approved `.pt` checkpoint. To use TensorRT, build and check the engine **on the field PC** using the [export and parity instructions](../docs/HARDWARE_BENCHMARK_GUIDE.md), then change `VISION_MODEL` and its hash together.

Before enabling the service, verify the configuration without starting anything:

```bash
sudo -u traffic .venv/bin/python run_multi_camera.py --check-config --output-mode controller --controller-config config/controller_main.json --sources-file /etc/traffic-vision/sources.json --model models/yolo26s_thai_traffic.pt
```

After this succeeds and the environment file is complete:

```bash
sudo systemctl enable --now traffic-vision.service traffic-vision-health.timer
```

## Windows install

Use a service wrapper such as NSSM (or a Task Scheduler task "at startup, restart on failure"). Run `.venv\Scripts\python.exe run_multi_camera.py` with the same arguments as `ExecStart`, using Windows paths. Set the service to restart on exit. Schedule `python -m tools.healthcheck --file <health.json> --max-age 15 --stale-only --startup-grace 60` every 30 s and restart when it returns 2. Use a separate check without `--stale-only` for camera/delivery alerts; those faults should not restart a healthy process.

## Exit codes and health

The runner warms inference before opening cameras or starting MQTT. Its health
file reports `starting/model_warmup` during that work. The watchdog's
`--startup-grace 60` permits that state for at most 60 seconds, then applies the
normal stale-file limit. Ordinary health checks return degraded during warmup.
This allowance does not extend camera or controller observation freshness.

- `run_multi_camera.py`: 0 = stopped normally, 1 = gave up after repeated processing errors (the supervisor restarts it), 2 = configuration/startup error (fix the configuration; restarting will not help).
- `tools/healthcheck.py`: 0 = all cameras usable and controller output publishing (if enabled), 1 = some camera unusable or controller output suppressed/disconnected, 2 = no usable camera, or the health file is stale. The restart timer uses `--stale-only`, so camera/delivery faults do not restart a healthy process.
- Shadow counts go to `traffic/counts/shadow` and retained health to `traffic/health/shadow/<source-id>`. The broker publishes `{"status":"offline"}` on that health topic if the runner dies. These topics do not feed the existing controller/dashboard.
- Controller counts go to `traffic/counts` as complete legacy snapshots. A required-camera fault withholds the whole snapshot; health/records continue. Inspect `controllerDelivery.state` and `blockers` in the health file. `accepted` counts local publisher acceptance, not confirmed database writes. Main ignores the diagnostic health topic and keeps displaying old counts during silence.

## Maintenance

Pass `--maintenance-file /path/to/maintenance.json`, or set
`VISION_MAINTENANCE_FILE` in the service environment. The file maps camera names or physical
IDs to reasons, for example `{"east": "Lens cleaning"}`. Updates apply while the
runner is running. Listed cameras become unknown; a required stop-line camera
withholds the entire main count snapshot. Remove its entry to resume after fresh
observations and tracker confirmation. A malformed file disables measurements
until corrected or removed. A missing file means no maintenance.

## Commissioning checklist

Use the [validation guide](../docs/HARDWARE_BENCHMARK_GUIDE.md) to record evidence
for the actual deployment hardware and camera views.

1. Clocks: NTP on vision and main's host. Verify main's configured freshness is 2,000 ms (or adjust the vision profile), observation age/delivery fit that budget, and `CAM-01` is registered for `INT-001`.
2. `--check-config` passes. The model hash matches the approved checkpoint or verified engine.
3. Re-take calibration reference frames from the live cameras (segmentor save) and confirm lane coverage on site.
4. Verify held-out labelled count accuracy. If using TensorRT, run `python -m tools.trt_parity` on the field GPU. Run a 24–72 hour production soak with live sources and the chosen output mode/model; see the validation guide. The soak uses an in-memory sink unless a broker is supplied. Injected outages withhold frames and do not test network reconnect.
5. Separately pull each camera's cable and power-cycle the broker while recording shadow payloads. Every affected lane and gate must become unknown. Verify camera recovery and broker reconnection. Check source/NVR latency: timestamps describe local decoding, and cannot reveal old frames queued upstream.
6. Coordinate producer cutover: stop the mock generator and any other `INT-001` count source, and clear any retained historical counts. Exercise real persistence and outage/fallback/recovery against main. Fallback starts at the next scheduled decision and the existing dashboard does not dim stale counts; operators need vision's health/preview too. Confirm baseline min-green/early-skip behavior and physical signal safety before letting a pilot change lights.

## Isolated transport check

`tools.local_transport_smoke` needs FFmpeg, the optional `amqtt` test package and
an explicit MediaMTX executable. It has been checked with the
[official MediaMTX 1.21.1 release](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.1).
Keep helper binaries outside version control. The tool binds RTSP and MQTT to
random loopback ports and disables MediaMTX's other listeners. It restreams one
recording at 640×360, exercises the real camera reader and publisher, and feeds
the received wire fixture to the pinned-main harness. It does no YOLO inference
and starts no system backend/database. It is not a field acceptance test.

```powershell
.venv\Scripts\python.exe -m tools.local_transport_smoke --mediamtx .venv/network-smoke/mediamtx.exe --ffmpeg ffmpeg --out output/local-transport.json
```
