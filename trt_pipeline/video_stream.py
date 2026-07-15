import cv2 as cv
import threading
import queue
import numpy as np
from datetime import datetime
import os


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def prepare_video_source(video_path):
    """
    Prepare video source - handles file paths, network URLs, and camera indices.
    
    Args:
        video_path: Path to video file, RTSP/RTMP/HTTP URL, or camera index
        
    Returns:
        Processed video source path (str or int)
    """
    # Network stream - return as-is
    if isinstance(video_path, str) and video_path.startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    ):
        return video_path
    
    # Camera index
    if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
        return int(video_path)
    
    # File path
    if isinstance(video_path, str):
        # Already absolute
        if os.path.isabs(video_path):
            return video_path
        
        # Convert relative to absolute
        ROOT_DIR = os.path.abspath(os.curdir)
        return os.path.join(ROOT_DIR, video_path)
    
    return video_path


def is_network_stream(video_source):
    """Check if video source is a network stream"""
    return isinstance(video_source, str) and video_source.startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )


def capture_one_frame(video_source):
    """
    Capture a single frame from a video source without starting a stream.
    Useful for testing connections or getting a snapshot.
    
    Args:
        video_source: Path to video file, RTSP URL, or camera index
    
    Returns:
        np.ndarray: The captured frame, or None if capture failed
    """
    print(f'{datetime.now()} Capturing one frame from: {video_source}')
    
    # Prepare the video source
    video_source = prepare_video_source(video_source)

    cap = cv.VideoCapture(video_source)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
    
    # Set timeout for network streams (5 seconds)
    if is_network_stream(video_source):
        cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    
    if not cap.isOpened():
        print(f"{datetime.now()} Could not open video source: {video_source}")
        return None
    
    try:
        ret, frame = cap.read()
        if not ret:
            print(f"{datetime.now()} Failed to read frame from video source")
            return None
        
        h, w = frame.shape[:2]
        print(f"{datetime.now()} Successfully captured frame: {w}x{h}")
        return frame
        
    except Exception as e:
        print(f"{datetime.now()} Error capturing frame: {e}")
        return None
    finally:
        cap.release()


class AsyncImageSaver:
    """Thread-safe async image saver for non-blocking disk writes"""
    
    def __init__(self, queue_size=256):
        self.q = queue.Queue(maxsize=queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background worker that saves images from queue"""
        while self.running or not self.q.empty():
            try:
                path, image = self.q.get(timeout=0.1)
                cv.imwrite(path, image, [cv.IMWRITE_JPEG_QUALITY, 30])
            except queue.Empty:
                continue

    def save(self, path, image):
        """Queue an image for async saving"""
        if self.running:
            self.q.put((path, image))

    def stop(self):
        """Stop the saver and wait for queue to finish"""
        self.running = False
        self.thread.join()


class VideoStream:
    """
    Threaded video stream reader with support for:
    - Video files (.avi, .mp4, etc.)
    - RTSP/RTMP/HTTP network streams
    - USB/IP camera devices
    """
    
    def __init__(self, video_path, skip=1, queue_size=2):
        """
        Initialize video stream reader.
        
        Args:
            video_path: Path to video file, RTSP URL, or camera index
            skip: Process every Nth frame (1 = all frames, 2 = every other frame)
            queue_size: Size of frame buffer queue
        """
        self.skip = max(1, skip)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        
        # Prepare and open video source
        self.video_source = prepare_video_source(video_path)
        self.cap = self._open_video_capture()
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_source}")
        
        # Log video info
        self._log_video_info()
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open_video_capture(self):
        """Open video capture with appropriate settings"""
        cap = cv.VideoCapture(self.video_source)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
        
        # Set timeouts for network streams
        if is_network_stream(self.video_source):
            cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            print(f"{datetime.now()} Network stream timeouts set to 5 seconds")
        
        return cap

    def _log_video_info(self):
        """Log video stream information"""
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv.CAP_PROP_FPS)
        
        source_type = "Network Stream" if is_network_stream(self.video_source) else "Video File/Camera"
        
        print(f"{datetime.now()} Video Source: {source_type}")
        print(f"{datetime.now()} Resolution: {width}x{height}")
        print(f"{datetime.now()} FPS: {fps:.2f}")
        print(f"{datetime.now()} Skip: {self.skip} (processing every {self.skip} frame(s))")

    def _reader(self):
        """Background thread that reads frames from video source"""
        idx = 0
        consecutive_failures = 0
        max_failures = 10
        
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"{datetime.now()} Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print(f"{datetime.now()} Max consecutive failures reached. Stopping stream.")
                    self.stopped = True
                    break
                
                # Try to reconnect network streams
                if is_network_stream(self.video_source):
                    print(f"{datetime.now()} Attempting to reconnect to network stream...")
                    self.cap.release()
                    self.cap = self._open_video_capture()
                    if not self.cap.isOpened():
                        print(f"{datetime.now()} Reconnection failed")
                        self.stopped = True
                        break
                    print(f"{datetime.now()} Reconnected successfully")
                    consecutive_failures = 0
                    continue
                else:
                    # For video files, end of video
                    self.stopped = True
                    break
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            # Apply frame skipping
            if idx % self.skip == 0:
                try:
                    if is_network_stream(self.video_source):
                        # For network streams, drop old frames if queue is full
                        if self.queue.full():
                            try:
                                self.queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.queue.put((idx, frame), timeout=0.1)
                    else:
                        # For files, blocking put
                        self.queue.put((idx, frame))
                except queue.Full:
                    pass
            
            idx += 1

        self.cap.release()
        print(f"{datetime.now()} Video stream reader stopped")

    def read(self):
        """
        Read next frame from queue.
        
        Returns:
            Tuple of (frame_index, frame) or None if stream ended
        """
        if self.stopped and self.queue.empty():
            return None
        
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            if self.stopped:
                return None
            return self.read()

    def stop(self):
        """Stop the video stream reader"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def is_opened(self):
        """Check if video stream is open"""
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        """Get video FPS"""
        return self.cap.get(cv.CAP_PROP_FPS) if self.cap else 0

    def get_frame_count(self):
        """Get total frame count (only for video files, -1 for streams)"""
        if is_network_stream(self.video_source):
            return -1
        return int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) if self.cap else 0

    def get_resolution(self):
        """Get video resolution as (width, height)"""
        if self.cap:
            width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)