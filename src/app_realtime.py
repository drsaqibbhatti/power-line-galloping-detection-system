from __future__ import annotations
import argparse
from pathlib import Path
import csv
import yaml
import cv2

from core.io import read_json
from core.video import VideoStream
from core.stabilize import Stabilizer
from core.tracker import KLTTracker
from core.signal import SignalState, ema_update, detrend, window_metrics
from core.detector import DetectorConfig, GallopingDetector
from core.alerts import send_webhook, send_telegram
from core.viz import draw_overlay

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    roi_data = read_json(cfg["roi"]["roi_path"])
    roi_poly = roi_data.get("polygon_xy", [])
    if not roi_poly:
        raise RuntimeError(
            f"ROI not found. Run: python -m src.roi_tool --video <source> --out {cfg['roi']['roi_path']}"
        )

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    vs = VideoStream(
        source=str(cfg["video"]["source"]),
        fps_target=int(cfg["video"]["fps_target"]),
        resize_width=int(cfg["video"]["resize_width"]),
        buffer_seconds=float(cfg["video"]["buffer_seconds"]),
    )

    stabilizer = Stabilizer(
        max_features=int(cfg["stabilization"]["max_features"]),
        keep_ratio=float(cfg["stabilization"]["keep_ratio"]),
    )
    stab_enabled = bool(cfg["stabilization"]["enabled"])

    tracker = KLTTracker(
        klt_win=int(cfg["tracking"]["klt_win"]),
        klt_max_level=int(cfg["tracking"]["klt_max_level"]),
        max_points=int(cfg["tracking"]["klt_max_points"]),
        refresh_every_frames=int(cfg["tracking"]["refresh_every_frames"]),
        min_tracked_points=int(cfg["tracking"]["min_tracked_points"]),
    )

    sig = SignalState(t=[], y=[], y_smooth=0.0, init=False)

    det_cfg = DetectorConfig(
        amp_px_threshold=float(cfg["detector"]["amp_px_threshold"]),
        freq_hz_min=float(cfg["detector"]["freq_hz_min"]),
        freq_hz_max=float(cfg["detector"]["freq_hz_max"]),
        persist_seconds=float(cfg["detector"]["persist_seconds"]),
        cooldown_seconds=float(cfg["detector"]["cooldown_seconds"]),
    )
    detector = GallopingDetector(det_cfg)

    pkt = vs.read()
    if pkt is None:
        raise RuntimeError("No frames available")
    frame = pkt.frame_bgr
    if stab_enabled:
        frame, _ = stabilizer.stabilize(frame)
    tracker.init(frame, roi_poly, min_points=int(cfg["roi"]["min_points"]))

    writer = None
    if cfg["outputs"]["save_annotated_video"]:
        h, w = frame.shape[:2]
        fps = cfg["video"]["fps_target"] if cfg["video"]["fps_target"] else (vs.native_fps if vs.native_fps else 25.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, float(fps), (w, h))

    csv_path = out_dir / "timeseries.csv"
    csv_f = None
    csv_w = None
    if cfg["outputs"]["save_csv"]:
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t", "dy_med", "dy_smooth", "amp_pp", "rms", "f_dom", "num_pts", "event", "severity"])

    cv2.namedWindow("Galloping Monitor", cv2.WINDOW_NORMAL)
    import numpy as np

    while True:
        pkt = vs.read()
        if pkt is None:
            break

        frame = pkt.frame_bgr
        if stab_enabled:
            frame, _ = stabilizer.stabilize(frame)

        tr = tracker.step(frame)
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

        met = window_metrics(
            t=t_arr,
            x=y_use,
            freq_min=float(cfg["detector"]["freq_hz_min"]),
            freq_max=float(cfg["detector"]["freq_hz_max"]),
        )

        res = detector.update(now=now, amp_pp=met["amp_pp"], f_dom=met["f_dom"])

        status = f"AmpPP:{met['amp_pp']:.1f}px  F:{met['f_dom']:.2f}Hz  Pts:{tr['num_pts']}  "
        status += f"EVENT {res['severity']}" if res["event"] else "OK"

        vis = draw_overlay(frame, roi_poly, tr["pts"], status)

        if writer is not None:
            writer.write(vis)

        if csv_w is not None:
            csv_w.writerow([now, dy, sig.y_smooth, met["amp_pp"], met["rms"], met["f_dom"], tr["num_pts"], int(res["event"]), res["severity"] or ""])

        if res["event"]:
            payload = {
                "type": "galloping_event",
                "severity": res["severity"],
                "amp_pp_px": met["amp_pp"],
                "f_dom_hz": met["f_dom"],
                "timestamp": now,
            }

            if cfg["alerts"]["webhook"]["enabled"] and cfg["alerts"]["webhook"]["url"]:
                ok, msg = send_webhook(cfg["alerts"]["webhook"]["url"], payload)
                print(f"[WEBHOOK] {ok} {msg}")

            if cfg["alerts"]["telegram"]["enabled"]:
                ok, msg = send_telegram(
                    cfg["alerts"]["telegram"]["bot_token"],
                    cfg["alerts"]["telegram"]["chat_id"],
                    f"⚠️ Galloping detected: {res['severity']} | AmpPP={met['amp_pp']:.1f}px | F={met['f_dom']:.2f}Hz",
                )
                print(f"[TELEGRAM] {ok} {msg}")

            if cfg["outputs"]["save_event_clips"]:
                clip_path = out_dir / f"event_{int(now)}.mp4"
                pre = vs.get_prebuffer()
                if pre:
                    h, w = pre[0].frame_bgr.shape[:2]
                    fps = cfg["video"]["fps_target"] if cfg["video"]["fps_target"] else (vs.native_fps if vs.native_fps else 25.0)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    wri = cv2.VideoWriter(str(clip_path), fourcc, float(fps), (w, h))
                    for p in pre:
                        wri.write(p.frame_bgr)
                    wri.release()
                    print(f"[EVENT CLIP] Saved {clip_path}")

        cv2.imshow("Galloping Monitor", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    vs.release()
    if writer is not None:
        writer.release()
    if csv_f is not None:
        csv_f.close()
    cv2.destroyAllWindows()
    print(f"Done. Outputs at: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
