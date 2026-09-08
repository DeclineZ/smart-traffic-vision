"""Bounded, ordered decoding with optional overlap; no frame dropping."""
import hashlib
import queue
import threading
import time


class FrameReader:
    def __init__(self, cap, count, first, first_ms, depth=0):
        self.cap, self.count = cap, count
        self.first, self.first_ms = first, first_ms
        self.digest = hashlib.sha256()
        self.stop = threading.Event()
        self.queue = queue.Queue(maxsize=max(1, depth))
        self.thread = None
        if depth:
            self.thread = threading.Thread(target=self._produce, daemon=True)
            self.thread.start()

    def _frames(self):
        for i in range(self.count):
            if self.stop.is_set():
                return
            start = time.perf_counter()
            ok, frame = (True, self.first) if i == 0 else self.cap.read()
            decode = self.first_ms if i == 0 else (time.perf_counter() - start) * 1000
            if not ok:
                raise ValueError(f'Early decode failure at segment frame {i}')
            start = time.perf_counter()
            self.digest.update(frame.data)
            yield frame, decode, (time.perf_counter() - start) * 1000

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            for item in self._frames():
                self._put(item)
        except Exception as error:
            self._put(error)
        finally:
            self._put(None)

    def __iter__(self):
        if self.thread is None:
            yield from self._frames()
            return
        while True:
            item = self.queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

