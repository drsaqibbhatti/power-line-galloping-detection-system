from __future__ import annotations
import cv2
import numpy as np

def draw_overlay(frame, roi_polygon_xy, pts, status_text: str):
    out = frame.copy()

    if roi_polygon_xy:
        pts_poly = np.array(roi_polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts_poly], isClosed=True, color=(255, 200, 0), thickness=2)

    if pts is not None and len(pts) > 0:
        for p in pts.reshape(-1, 2):
            cv2.circle(out, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

    cv2.putText(out, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out
