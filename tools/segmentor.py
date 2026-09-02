"""
Interactive Lane Polygon Segmentor & Calibration Tool.
Allows calibrating perspective lane polygon coordinates by clicking vertices on video frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2 as cv
import numpy as np


class RoadSegmenter:
    """Interactive tool to mark polygonal lane and road boundaries on camera frames."""

    def __init__(self, video_path: str, time_in_seconds: float = 0.0):
        self.video_path = video_path
        self.time_in_seconds = time_in_seconds
        self.coordinates: List[Tuple[int, int]] = []
        self.frame = self._extract_frame()

        if self.frame is None:
            raise ValueError(f"Unable to read video frame from source: {video_path}")

    def _extract_frame(self) -> np.ndarray | None:
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        # Calculate target frame index from timestamp
        fps = cap.get(cv.CAP_PROP_FPS) or 25.0
        target_frame = int(self.time_in_seconds * fps)
        cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))
            print(f"  Added point #{len(self.coordinates)}: [{x}, {y}]")

    def segment(self) -> List[List[int]]:
        """
        Opens interactive calibration window.
        Returns list of [x, y] coordinates in sequential polygon order.
        """
        window_name = "Lane Polygon Calibration Tool"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "=" * 60)
        print(" Lane Polygon Calibration Active")
        print(" Left-Click: Add polygon vertex")
        print(" 'r' key:    Reset / clear current vertices")
        print(" 's' key:    Print JSON polygon array")
        print(" 'q' / ESC:  Save and finish")
        print("=" * 60 + "\n")

        while True:
            display_frame = self.frame.copy()

            # Draw polygon perimeter lines and vertices
            if len(self.coordinates) > 1:
                pts = np.array(self.coordinates, dtype=np.int32)
                cv.polylines(display_frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            for idx, (x, y) in enumerate(self.coordinates):
                cv.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv.putText(
                    display_frame,
                    f"#{idx + 1}",
                    (x + 8, y - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            # HUD overlay
            overlay_text = f"Points: {len(self.coordinates)} | 'r' to reset | 'q' to save & exit"
            cv.putText(display_frame, overlay_text, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.imshow(window_name, display_frame)
            key = cv.waitKey(20) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("r"):
                self.coordinates.clear()
                print("  Cleared all polygon points.")
            elif key == ord("s"):
                formatted = [[int(x), int(y)] for x, y in self.coordinates]
                print(f"\nCurrent Polygon Coordinates ({len(formatted)} points):")
                print(f'"polygon": {formatted}\n')

        cv.destroyAllWindows()
        result = [[int(x), int(y)] for x, y in self.coordinates]
        print(f"\nFinal Calibrated Coordinates ({len(result)} vertices):")
        print(f'"polygon": {result}\n')
        return result


def main():
    parser = argparse.ArgumentParser(description="Interactive Lane Polygon Segmentor & Calibration Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or RTSP stream URL")
    parser.add_argument("--sec", type=float, default=0.0, help="Timestamp in seconds to capture preview frame")
    parser.add_argument("--n", type=int, default=1, help="Number of sequential lanes to calibrate")
    args = parser.parse_args()

    for lane_idx in range(args.n):
        print(f"\n--- Calibrating Lane {lane_idx + 1} of {args.n} ---")
        segmentor = RoadSegmenter(video_path=args.video, time_in_seconds=args.sec)
        segmentor.segment()


if __name__ == "__main__":
    main()
