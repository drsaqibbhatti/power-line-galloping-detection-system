from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    amp_px_threshold: float
    freq_hz_min: float
    freq_hz_max: float
    persist_seconds: float
    cooldown_seconds: float

class GallopingDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._above_since: float | None = None
        self._last_trigger: float = 0.0

    def update(self, now: float, amp_pp: float, f_dom: float) -> dict:
        in_band = (self.cfg.freq_hz_min <= f_dom <= self.cfg.freq_hz_max) if f_dom > 0 else False
        above = (amp_pp >= self.cfg.amp_px_threshold) and in_band

        if (now - self._last_trigger) < self.cfg.cooldown_seconds:
            above = False
            self._above_since = None

        if above:
            if self._above_since is None:
                self._above_since = now
            dur = now - self._above_since
            if dur >= self.cfg.persist_seconds:
                self._last_trigger = now
                self._above_since = None
                return {"event": True, "severity": self._severity(amp_pp), "duration": dur}
            return {"event": False, "severity": None, "duration": dur}
        else:
            self._above_since = None
            return {"event": False, "severity": None, "duration": 0.0}

    def _severity(self, amp_pp: float) -> str:
        if amp_pp >= self.cfg.amp_px_threshold * 2.0:
            return "CRITICAL"
        if amp_pp >= self.cfg.amp_px_threshold * 1.3:
            return "HIGH"
        return "MEDIUM"
