from __future__ import annotations
import cv2
import numpy as np

class Stabilizer:
    """
    Global motion compensation using ORB + affine transform.
    Helps reduce false galloping from camera shake.
    """
    def __init__(self, max_features: int = 400, keep_ratio: float = 0.7):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.keep_ratio = float(keep_ratio)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def reset(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def stabilize(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or des is None or self.prev_des is None or len(kp) < 10 or len(self.prev_kp) < 10:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return frame_bgr, np.eye(2, 3, dtype=np.float32)

        matches = self.bf.match(self.prev_des, des)
        if not matches:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return frame_bgr, np.eye(2, 3, dtype=np.float32)

        matches = sorted(matches, key=lambda m: m.distance)
        keep = int(len(matches) * self.keep_ratio)
        matches = matches[:max(keep, 10)]

        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        stabilized = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return stabilized, M
