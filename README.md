# Power Line Galloping Detection System

Video-based monitoring system for detecting abnormal oscillations (**galloping**) in power transmission lines.  
Designed for **real-time detection + alerting** to help prevent outages, reduce equipment damage, and improve grid stability.

**Project page:** https://drsaqibbhatti.com/projects/power-line-galloping.html  

---

## Overview

Power line galloping can cause outages and equipment damage, and manual monitoring is impractical at scale.  
This project provides a **video-based detection pipeline** using **motion/time-series analysis**, built to be robust to outdoor conditions (wind/weather/distance) and camera positioning, with integrated alerts for real-time monitoring.

**Tech stack:** Python, OpenCV, PyTorch, NumPy

---

## Key Features

- **Real-time video monitoring** (webcam / video file / RTSP)
- **Motion tracking** inside a user-defined ROI (region of interest)
- **Time-series analysis** of displacement to detect oscillation events
- **Event triggering with persistence** (reduces false positives)
- **Output artifacts**
  - Annotated video
  - Time-series CSV logging
  - Event clips (pre-event buffer)

