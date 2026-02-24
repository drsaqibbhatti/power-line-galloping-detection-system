from __future__ import annotations
import cv2
import numpy as np

def mask_from_polygon(shape_hw: tuple[int, int], polygon_xy: list[list[int]]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def sample_points_in_mask(mask: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 1, 2), dtype=np.float32)
    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts.reshape(-1, 1, 2)

class KLTTracker:
    """
    Tracks points in ROI using Lucas-Kanade optical flow.
    Produces per-frame displacement statistics (median dy).
    """
    def __init__(self, klt_win=21, klt_max_level=3, max_points=120, refresh_every_frames=60, min_tracked_points=25):
        self.lk_params = dict(
            winSize=(int(klt_win), int(klt_win)),
            maxLevel=int(klt_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.max_points = int(max_points)
        self.refresh_every = int(refresh_every_frames)
        self.min_tracked = int(min_tracked_points)

        self.prev_gray = None
        self.prev_pts = None
        self.frame_since_refresh = 0
        self.roi_mask = None

    def init(self, frame_bgr: np.ndarray, roi_polygon_xy: list[list[int]], min_points: int = 80):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        self.roi_mask = mask_from_polygon((h, w), roi_polygon_xy)

        pts = sample_points_in_mask(self.roi_mask, max(min_points, self.max_points))
        if pts.shape[0] > self.max_points:
            pts = pts[: self.max_points]

        self.prev_gray = gray
        self.prev_pts = pts
        self.frame_since_refresh = 0

    def _refresh_points(self, gray: np.ndarray):
        pts = sample_points_in_mask(self.roi_mask, self.max_points)
        self.prev_pts = pts
        self.frame_since_refresh = 0

    def step(self, frame_bgr: np.ndarray) -> dict:
        if self.prev_gray is None or self.prev_pts is None or self.prev_pts.shape[0] == 0:
            return {"ok": False, "dy_med": 0.0, "dx_med": 0.0, "num_pts": 0, "pts": np.zeros((0, 1, 2), np.float32)}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        next_pts, st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        if next_pts is None or st is None:
            self.prev_gray = gray
            return {"ok": False, "dy_med": 0.0, "dx_med": 0.0, "num_pts": 0, "pts": np.zeros((0, 1, 2), np.float32)}

        good = st.squeeze(1).astype(bool)
        prev_good = self.prev_pts[good]
        next_good = next_pts[good]

        num = int(next_good.shape[0])
        if num < self.min_tracked:
            self._refresh_points(gray)
            self.prev_gray = gray
            return {"ok": False, "dy_med": 0.0, "dx_med": 0.0, "num_pts": num, "pts": next_good}

        d = (next_good - prev_good).reshape(-1, 2)
        dx = float(np.median(d[:, 0]))
        dy = float(np.median(d[:, 1]))

        self.prev_gray = gray
        self.prev_pts = next_good.reshape(-1, 1, 2)

        self.frame_since_refresh += 1
        if self.refresh_every > 0 and self.frame_since_refresh >= self.refresh_every:
            self._refresh_points(gray)

        return {"ok": True, "dy_med": dy, "dx_med": dx, "num_pts": num, "pts": self.prev_pts}
