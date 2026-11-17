"""
stream_cameras.py - Interactive Network Camera Streamer for Traffic Simulation.

Features:
- Live playback control from keyboard:
    [SPACE]   : Pause / Resume playback
    [RIGHT]   : Forward +5 seconds
    [LEFT]    : Rewind -5 seconds
    [UP]      : Fast forward +15 seconds
    [DOWN]    : Fast rewind -15 seconds
    [R]       : Reset to beginning (0s)
    [Q] / ESC : Quit
- Continuous streaming over UDP (MPEG-TS):
    North:          udp://127.0.0.1:8554
    North Upstream: udp://127.0.0.1:8555
- Keeps network stream alive even when paused (repeats last frame so downstream detector stays connected).
- Automatically writes active endpoints to sources.json.
"""

import argparse
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

# Camera configuration (All 4 directions prepared; North active by default)
DEFAULT_CAMERA_STREAMS = {
    # --- 1. North Approach (Dual Camera - Active by default) ---
    "north": {
        "video": "videos/cam44_north.avi",
        "port": 8554,
        "description": "North Stop-line Main (Cam 1: 0-55m)",
    },
    "north_upstream": {
        "video": "videos/cam43_south.avi",   # Using available cam43 video as upstream feed
        "port": 8555,
        "description": "North Upstream (Cam 2: 100m+)",
    },

    # --- 2. South Approach (Available to scale) ---
    "south": {
        "video": "videos/cam43_south.avi",
        "port": 8556,
        "description": "South Stop-line Main",
        "disabled": True,
    },
    "south_upstream": {
        "video": "videos/cam44_north.avi",
        "port": 8557,
        "description": "South Upstream",
        "disabled": True,
    },

    # --- 3. East Approach (Available to scale) ---
    "east": {
        "video": "videos/cam03_east.avi",
        "port": 8558,
        "description": "East Stop-line Main",
        "disabled": True,
    },
    "east_upstream": {
        "video": "videos/cam03_east.avi",
        "port": 8559,
        "description": "East Upstream",
        "disabled": True,
    },

    # --- 4. West Approach (Available to scale) ---
    "west": {
        "video": "videos/cam46_west.avi",
        "port": 8560,
        "description": "West Stop-line Main",
        "disabled": True,
    },
    "west_upstream": {
        "video": "videos/cam46_west.avi",
        "port": 8561,
        "description": "West Upstream",
        "disabled": True,
    },
}


class CameraStreamWorker:
    def __init__(self, name: str, video_path: str, port: int, host: str = "127.0.0.1"):
        self.name = name
        self.video_path = video_path
        self.port = port
        self.host = host
        self.url = f"udp://{host}:{port}?pkt_size=1316"

        self.cap = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.video_path}")

        self.fps = self.cap.get(cv.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT) or 1)
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 720)
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0

        # Output resolution scaled for efficient low-latency UDP transmission
        self.out_w = self.width
        self.out_h = self.height
        if self.out_h % 2 != 0:
            self.out_h += 1

        self.ffmpeg_proc: Optional[subprocess.Popen] = None
        self.last_frame: Optional[np.ndarray] = None
        self.current_frame_idx = 0
        self._init_ffmpeg()

    def _init_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.out_w}x{self.out_h}",
            "-r", f"{self.fps:.2f}",
            "-i", "-",
            "-c:v", "mpeg4",
            "-q:v", "3",
            "-g", "15",
            "-an",
            "-f", "mpegts",
            self.url,
        ]
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def read_next_frame(self, is_paused: bool) -> bool:
        if is_paused:
            # When paused, repeat last frame to keep stream alive
            if self.last_frame is not None and self.ffmpeg_proc and self.ffmpeg_proc.stdin:
                try:
                    self.ffmpeg_proc.stdin.write(self.last_frame.tobytes())
                except (BrokenPipeError, OSError):
                    pass
            return True

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Loop video to beginning
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False

        self.current_frame_idx = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))

        if (frame.shape[1], frame.shape[0]) != (self.out_w, self.out_h):
            frame = cv.resize(frame, (self.out_w, self.out_h))

        self.last_frame = frame
        if self.ffmpeg_proc and self.ffmpeg_proc.stdin:
            try:
                self.ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
        return True

    def seek_seconds(self, delta_sec: float):
        current_sec = self.current_frame_idx / self.fps if self.fps > 0 else 0.0
        target_sec = max(0.0, min(self.duration_sec, current_sec + delta_sec))
        target_frame = int(target_sec * self.fps)
        self.cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame_idx = target_frame
        # Grab frame immediately to update paused buffer
        ret, frame = self.cap.read()
        if ret and frame is not None:
            if (frame.shape[1], frame.shape[0]) != (self.out_w, self.out_h):
                frame = cv.resize(frame, (self.out_w, self.out_h))
            self.last_frame = frame

    def reset(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        ret, frame = self.cap.read()
        if ret and frame is not None:
            if (frame.shape[1], frame.shape[0]) != (self.out_w, self.out_h):
                frame = cv.resize(frame, (self.out_w, self.out_h))
            self.last_frame = frame

    def close(self):
        if self.cap:
            self.cap.release()
        if self.ffmpeg_proc:
            try:
                if self.ffmpeg_proc.stdin:
                    self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.terminate()
            except Exception:
                pass


def generate_sources_json(active_streams: Dict[str, Dict], filename: str = "sources.json") -> None:
    sources = {name: f"udp://127.0.0.1:{cfg['port']}" for name, cfg in active_streams.items()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Traffic Video Streamer with Pause/Play and Fast-Forward/Rewind."
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["north", "north_upstream"],
        help="Cameras to stream (default: 'north' 'north_upstream')",
    )
    parser.add_argument(
        "--all-8",
        action="store_true",
        help="Stream all 8 cameras for full 4-way intersection",
    )
    parser.add_argument(
        "--sources-file",
        default="sources.json",
        help="Sources file to auto-populate (default: sources.json)",
    )
    args = parser.parse_args()

    active_streams: Dict[str, Dict] = {}
    if args.all_8:
        active_streams = DEFAULT_CAMERA_STREAMS
    else:
        for c in args.cameras:
            if c in DEFAULT_CAMERA_STREAMS:
                active_streams[c] = DEFAULT_CAMERA_STREAMS[c]

    generate_sources_json(active_streams, args.sources_file)

    workers: List[CameraStreamWorker] = []
    for name, cfg in active_streams.items():
        vpath = cfg["video"]
        if not os.path.exists(vpath) and name == "north_upstream":
            vpath = "videos/cam43_south.avi"
        w = CameraStreamWorker(name, vpath, cfg["port"])
        workers.append(w)
        print(f"[*] Started stream [{name:<14}] -> {w.url} ({w.duration_sec:.1f}s)")

    is_paused = False
    print("\n" + "=" * 65)
    print("  INTERACTIVE STREAMING CONTROLS:")
    print("  [SPACE]       : Pause / Resume")
    print("  [RIGHT ARROW] : Forward +5 seconds")
    print("  [LEFT ARROW]  : Rewind  -5 seconds")
    print("  [UP ARROW]    : Fast Forward +15 seconds")
    print("  [DOWN ARROW]  : Fast Rewind  -15 seconds")
    print("  [R]           : Reset video to 0:00")
    print("  [Q] or ESC    : Quit streamer")
    print("=" * 65 + "\n")

    base_fps = workers[0].fps if workers else 25.0
    frame_interval = 1.0 / base_fps

    try:
        last_status_print = 0.0
        while True:
            start_t = time.perf_counter()

            # Handle non-blocking keyboard input
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'q', b'Q', b'\x1b'):  # 'q' or ESC
                    print("\n[!] User stopped streaming.")
                    break
                elif ch == b' ':  # Space (Pause / Resume)
                    is_paused = not is_paused
                    state_str = "PAUSED [||]" if is_paused else "PLAYING [> ]"
                    print(f"\r--> Stream {state_str}                                ", end="", flush=True)
                elif ch in (b'r', b'R'):  # Reset
                    for w in workers:
                        w.reset()
                    print("\r--> Video Reset to [0:00]                                ", end="", flush=True)
                elif ch in (b'\x00', b'\xe0'):  # Extended keys (arrows)
                    ext = msvcrt.getch()
                    if ext == b'M':  # Right Arrow (+5s)
                        for w in workers:
                            w.seek_seconds(5.0)
                        print("\r--> Seek: [+5s Forward]                                  ", end="", flush=True)
                    elif ext == b'K':  # Left Arrow (-5s)
                        for w in workers:
                            w.seek_seconds(-5.0)
                        print("\r--> Seek: [-5s Rewind]                                   ", end="", flush=True)
                    elif ext == b'H':  # Up Arrow (+15s)
                        for w in workers:
                            w.seek_seconds(15.0)
                        print("\r--> Fast Forward: [+15s]                                ", end="", flush=True)
                    elif ext == b'P':  # Down Arrow (-15s)
                        for w in workers:
                            w.seek_seconds(-15.0)
                        print("\r--> Fast Rewind:  [-15s]                                ", end="", flush=True)

            # Process frame for each stream
            for w in workers:
                w.read_next_frame(is_paused)

            # Display running progress periodically
            now = time.time()
            if now - last_status_print > 0.5:
                last_status_print = now
                if workers:
                    w0 = workers[0]
                    curr_s = w0.current_frame_idx / w0.fps if w0.fps > 0 else 0.0
                    status_lbl = "PAUSED" if is_paused else "STREAMING"
                    print(f"\r[{status_lbl}] Pos: {curr_s:05.1f}s / {w0.duration_sec:.1f}s | {len(workers)} streams active", end="", flush=True)

            # Pace to real-time FPS
            elapsed = time.perf_counter() - start_t
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[*] Exiting streaming loop...")
    finally:
        for w in workers:
            w.close()
        print("[✓] All streams terminated cleanly.")


if __name__ == "__main__":
    main()
