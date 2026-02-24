from __future__ import annotations
import argparse
import cv2
from core.io import write_json

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="video file or camera index like 0")
    p.add_argument("--out", default="roi.json")
    return p.parse_args()

def main():
    args = parse_args()
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read frame")

    cv2.namedWindow("ROI Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Tool", on_mouse)

    while True:
        vis = frame.copy()
        for p in points:
            cv2.circle(vis, tuple(p), 4, (0, 255, 0), -1)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(vis, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)

        cv2.putText(
            vis,
            "LClick:add  RClick:undo  Enter:save  C:clear  Esc:quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("ROI Tool", vis)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k in (10, 13):
            if len(points) >= 3:
                write_json(args.out, {"polygon_xy": points})
                print(f"Saved ROI -> {args.out}")
                break
        if k in (ord("c"), ord("C")):
            points.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
