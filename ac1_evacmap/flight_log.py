"""Журнал обльоту: кадри камери, покази висотоміра, кути рискання та
положення точок зависання. Один і той самий формат використовують модуль
керування дроном (запис), модуль синтетичного обльоту (генерація) та
модуль побудови карти (відтворення).

Структура теки журналу:
    meta.json              -- камера, розмір приміщення, параметри обльоту
    telemetry.csv          -- index, x, y, z_tof, yaw_deg, waypoint
    frames/frame_0000.png  -- кадри RGB (за наявності)
    pred/pred_0000.npz     -- обернена відносна глибина та маска підлоги
                              (записує модуль сприйняття або симулятор)
"""
import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameRecord:
    index: int
    x: float
    y: float
    z_tof: float
    yaw_deg: float
    waypoint: int


@dataclass
class FlightLog:
    root: str
    meta: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    # ---------- запис ----------
    def prepare(self):
        os.makedirs(os.path.join(self.root, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "pred"), exist_ok=True)

    def add_frame(self, rec, image=None, rel_inv_depth=None, floor_mask=None):
        self.records.append(rec)
        if image is not None:
            from PIL import Image
            Image.fromarray(image).save(self.frame_path(rec.index))
        if rel_inv_depth is not None:
            self.save_pred(rec.index, rel_inv_depth, floor_mask)

    def save_pred(self, index, rel_inv_depth, floor_mask):
        np.savez_compressed(self.pred_path(index),
                            rel_inv_depth=rel_inv_depth.astype(np.float32),
                            floor_mask=floor_mask.astype(np.bool_))

    def save(self):
        self.prepare()
        with open(os.path.join(self.root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.root, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
            for r in self.records:
                w.writerow([r.index, f"{r.x:.3f}", f"{r.y:.3f}", f"{r.z_tof:.3f}",
                            f"{r.yaw_deg:.1f}", r.waypoint])

    # ---------- читання ----------
    @classmethod
    def load(cls, root):
        log = cls(root=root)
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            log.meta = json.load(f)
        with open(os.path.join(root, "telemetry.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                log.records.append(FrameRecord(int(row["index"]), float(row["x"]), float(row["y"]),
                                               float(row["z_tof"]), float(row["yaw_deg"]),
                                               int(row["waypoint"])))
        return log

    def frame_path(self, index):
        return os.path.join(self.root, "frames", f"frame_{index:04d}.png")

    def pred_path(self, index):
        return os.path.join(self.root, "pred", f"pred_{index:04d}.npz")

    def has_frame(self, index):
        return os.path.exists(self.frame_path(index))

    def has_pred(self, index):
        return os.path.exists(self.pred_path(index))

    def load_frame(self, index):
        from PIL import Image
        return np.asarray(Image.open(self.frame_path(index)).convert("RGB"))

    def load_pred(self, index):
        z = np.load(self.pred_path(index))
        return z["rel_inv_depth"], z["floor_mask"]

    def __len__(self):
        return len(self.records)
