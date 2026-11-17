"""
watch_counts.py - Real-time terminal viewer for controller payloads.
Shows the exact clean format:
{
  "intersectionId": "INT-001",
  "cameraId": "CAM-01",
  "timestamp": "2026-11-03T...",
  "lanes": [
    { "laneId": "N1", "direction": "N", "count": ... },
    ...
  ],
  "outgoingfull": { "N": false, "E": false, "S": false, "W": false }
}
"""

import json
import os
import sys
import time

LOG_FILE = "live_counts.jsonl"

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Waiting for {LOG_FILE} to be created by run_multi_camera.py...")
        while not os.path.exists(LOG_FILE):
            time.sleep(0.5)

    print(f"[*] Streaming controller wire payloads from {LOG_FILE}:\n")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        # Seek to end
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                data = json.loads(line)
                wire = data.get("controllerPayload")
                if wire:
                    # Clean display format (filtering to top-level keys requested)
                    clean_out = {
                        "intersectionId": wire.get("intersectionId"),
                        "cameraId": wire.get("cameraId"),
                        "timestamp": wire.get("timestamp"),
                        "lanes": wire.get("lanes"),
                        "outgoingfull": wire.get("outgoingfull"),
                    }
                    print(json.dumps(clean_out, indent=2))
                    print("-" * 50)
            except Exception:
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting viewer.")
