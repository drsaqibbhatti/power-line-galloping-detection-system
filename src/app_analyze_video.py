from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from core.io import read_json
from core.video import VideoStream
from core.stabilize import Stabilizer
from core.tracker import KLTTracker
from core.signal import SignalState, ema_update, detrend, window_metrics
from core.detector import DetectorConfig, GallopingDetector

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default="", help="override video.source")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.source:
        cfg["video"]["source"] = args.source

    roi_poly = read_json(cfg["roi"]["roi_path"]).get("polygon_xy", [])
    if not roi_poly:
        raise RuntimeError("ROI not found. Run roi_tool.py first.")

    vs = VideoStream(
        source=str(cfg["video"]["source"]),
        fps_target=int(cfg["video"]["fps_target"]),
        resize_width=int(cfg["video"]["resize_width"]),
        buffer_seconds=float(cfg["video"]["buffer_seconds"]),
    )
    stabilizer = Stabilizer(int(cfg["stabilization"]["max_features"]), float(cfg["stabilization"]["keep_ratio"]))
    stab_enabled = bool(cfg["stabilization"]["enabled"])

    tracker = KLTTracker(
        int(cfg["tracking"]["klt_win"]),
        int(cfg["tracking"]["klt_max_level"]),
        int(cfg["tracking"]["klt_max_points"]),
        int(cfg["tracking"]["refresh_every_frames"]),
        int(cfg["tracking"]["min_tracked_points"]),
    )

    det_cfg = DetectorConfig(
        float(cfg["detector"]["amp_px_threshold"]),
        float(cfg["detector"]["freq_hz_min"]),
        float(cfg["detector"]["freq_hz_max"]),
        float(cfg["detector"]["persist_seconds"]),
        float(cfg["detector"]["cooldown_seconds"]),
    )
    detector = GallopingDetector(det_cfg)

    sig = SignalState(t=[], y=[], y_smooth=0.0, init=False)

    first = vs.read()
    if first is None:
        raise RuntimeError("No frames")
    fr = first.frame_bgr
    if stab_enabled:
        fr, _ = stabilizer.stabilize(fr)
    tracker.init(fr, roi_poly, min_points=int(cfg["roi"]["min_points"]))

    import numpy as np
    events = []
    while True:
        pkt = vs.read()
        if pkt is None:
            break
        fr = pkt.frame_bgr
        if stab_enabled:
            fr, _ = stabilizer.stabilize(fr)
        tr = tracker.step(fr)
        dy = tr["dy_med"]

        if not sig.init:
            sig.y_smooth = dy
            sig.init = True
        sig.y_smooth = ema_update(sig.y_smooth, dy, float(cfg["signal"]["smooth_alpha"]))
        sig.t.append(pkt.t)
        sig.y.append(sig.y_smooth)

        now = pkt.t
        win_s = float(cfg["signal"]["window_seconds"])
        while sig.t and (now - sig.t[0]) > (win_s * 1.2):
            sig.t.pop(0)
            sig.y.pop(0)

        t_arr = np.array(sig.t, dtype=float)
        y_arr = np.array(sig.y, dtype=float)
        y_use = detrend(y_arr.copy()) if bool(cfg["signal"]["detrend"]) else y_arr
        met = window_metrics(t_arr, y_use, det_cfg.freq_hz_min, det_cfg.freq_hz_max)

        res = detector.update(now, met["amp_pp"], met["f_dom"])
        if res["event"]:
            events.append({"t": now, "severity": res["severity"], "amp_pp": met["amp_pp"], "f_dom": met["f_dom"]})

    vs.release()
    print("Events:")
    for e in events:
        print(e)

if __name__ == "__main__":
    main()
