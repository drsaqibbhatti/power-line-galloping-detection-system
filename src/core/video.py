from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
import cv2

@dataclass
class FramePacket:
    frame_bgr: any
    t: float
    idx: int

class VideoStream:
    """
    Supports webcam index, file path, or RTSP/HTTP.
    Optional FPS downsampling and resizing.
    Maintains a pre-event buffer.
    """
    def __init__(self, source: str, fps_target: int = 0, resize_width: int = 0, buffer_seconds: float = 10.0):
        self.source = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps_target = fps_target
        self.resize_width = resize_width

        # buffer length in frames (approx)
        fps_for_buffer = fps_target if fps_target and fps_target > 0 else (self.native_fps if self.native_fps > 0 else 25.0)
        self.buffer = deque(maxlen=int(buffer_seconds * fps_for_buffer))
        self.frame_idx = 0
        self._last_emit = 0.0

    def _resize(self, frame):
        if self.resize_width and self.resize_width > 0:
            h, w = frame.shape[:2]
            if w != self.resize_width:
                scale = self.resize_width / float(w)
                nh = int(round(h * scale))
                frame = cv2.resize(frame, (self.resize_width, nh), interpolation=cv2.INTER_LINEAR)
        return frame

    def read(self) -> FramePacket | None:
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame = self._resize(frame)
        t = time.time()

        # fps downsample
        if self.fps_target and self.fps_target > 0:
            if self._last_emit > 0:
                dt = t - self._last_emit
                if dt < (1.0 / self.fps_target):
                    # skip frame
                    self.frame_idx += 1
                    return self.read()
            self._last_emit = t

        pkt = FramePacket(frame, t, self.frame_idx)
        self.buffer.append(pkt)
        self.frame_idx += 1
        return pkt

    def get_prebuffer(self):
        return list(self.buffer)

    def release(self):
        self.cap.release()
