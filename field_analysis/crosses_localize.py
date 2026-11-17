"""Контрольний експеримент: уточнення положення точок облету за жовтими хрестами на підлозі
(їхні координати відомі з місії: точки облету і зліт). У рятувальних умовах хрестів немає, тож це
лише оцінка того, скільки дає точне положення.

  python crosses_localize.py <лог АС Р1> [--out <тека>] [--debug <тека png>]

Жовта пляма (HSV) нижче горизонту, оточена підлогою (маска АС Р1), плоска (проєкція рамки на
підлогу не довша за 0,8 м) -> центр -> перетин променя з підлогою за висотою дрона -> оцінка
координат хреста -> найближчий відомий хрест у межах 0,9 м -> поправка положення точки
(медіана за всіма кадрами точки). Пише <лог>_xy з уточненими x, y (кадри й pred копіюються).
"""
import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ac1_evacmap"))
from flight_log import FlightLog                 # noqa: E402
from mono_depth import CameraModel, floor_depth  # noqa: E402

HSV_LO, HSV_HI = (16, 110, 120), (40, 255, 255)      # бежева плитка має S < 100
CROSS_AREA = 29300.0      # ~площа хреста у пікселях на 1 м: 2 смуги 0,35 x 0,05 м при f ~915 пікс.
MATCH_M = 0.9


def ground_point(cam, R, z_floor, d_c, u, v, pos):
    zf = z_floor[v, u]
    if not np.isfinite(zf):
        return None
    return pos + R @ (d_c[v, u] * zf)


def detect(frame_bgr, floor_mask, cam, rec, debug_path=None):
    """Список оцінок координат хрестів у світі для одного кадру."""
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, HSV_LO, HSV_HI)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(yellow, 8)
    v_h = int(H / 2 - cam.fx * np.tan(np.radians(cam.tilt_deg)))        # рядок горизонту
    R = cam.rotation_world(rec.yaw_deg)
    z_floor = floor_depth(cam, rec.yaw_deg, rec.z_tof)
    d_c = cam.pixel_dirs()
    pos = np.array([rec.x, rec.y, rec.z_tof])
    found = []
    dbg = frame_bgr.copy() if debug_path else None
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 250 or area > 40000 or y + h <= v_h + 15:
            continue
        if y + h >= H - 3 or x <= 2 or x + w >= W - 3:          # обрізаний краєм кадру: центр зміщений
            continue
        comp = lab == k
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~comp
        floor_frac = float(floor_mask[ring].mean()) if ring.any() else 0.0
        if floor_frac < 0.6:
            continue
        cu, cv_ = int(cents[k][0]), int(cents[k][1])
        p_c = ground_point(cam, R, z_floor, d_c, cu, cv_, pos)
        p_b = ground_point(cam, R, z_floor, d_c, cu, min(H - 1, y + h - 1), pos)
        p_t = ground_point(cam, R, z_floor, d_c, cu, max(0, y), pos)
        if p_c is None or p_b is None or p_t is None:
            continue
        length = float(np.hypot(*(p_t[:2] - p_b[:2])))
        dist = float(np.hypot(*(p_c[:2] - pos[:2])))
        if length > 0.8 or dist > 4.5 or area < max(400.0, 0.3 * CROSS_AREA / dist ** 2):
            continue
        found.append({"uv": (cu, cv_), "xy": (float(p_c[0]), float(p_c[1])), "len": length, "dist": dist,
                      "area": int(area)})
        if dbg is not None:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(dbg, f"{p_c[0]:.2f},{p_c[1]:.2f} L{length:.2f}", (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if dbg is not None:
        cv2.line(dbg, (0, v_h), (W, v_h), (255, 0, 0), 1)
        cv2.imwrite(debug_path, dbg)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    log = FlightLog.load(a.log)
    cam = CameraModel(*[int(v) for v in log.meta["frame_size"]], float(log.meta["hfov_deg"]), float(log.meta["tilt_deg"]))
    crosses = [tuple(p) for p in log.meta["waypoints"]] + ([tuple(log.meta["origin"])] if log.meta.get("origin") else [])
    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
    per_wp = {}
    details = []
    for rec in log.records:
        if not log.has_frame(rec.index) or not log.has_pred(rec.index):
            continue
        frame = cv2.imread(log.frame_path(rec.index))
        floor = np.load(log.pred_path(rec.index))["floor_mask"]
        dets = detect(frame, floor, cam, rec, os.path.join(a.debug, f"f{rec.index:04d}.jpg") if a.debug else None)
        best = {}
        for d in dets:
            est = np.array(d["xy"])
            dists = [np.hypot(*(np.array(c) - est)) for c in crosses]
            j = int(np.argmin(dists))
            if dists[j] <= MATCH_M and (j not in best or d["area"] > best[j]["area"]):
                best[j] = {**d, "j": j}
        for d in best.values():
            est, j = np.array(d["xy"]), d["j"]
            if True:
                corr = np.array(crosses[j]) - est
                per_wp.setdefault(rec.waypoint, []).append(corr)
                details.append({"frame": rec.index, "waypoint": rec.waypoint, "cross": crosses[j],
                                "est": d["xy"], "corr": [round(float(v), 3) for v in corr], "dist_m": round(d["dist"], 2),
                                "r": [float(est[0] - rec.x), float(est[1] - rec.y)]})
    out = a.out or a.log.rstrip("/\\") + "_xy"
    os.makedirs(out, exist_ok=True)
    for sub in ("frames", "pred"):
        if os.path.isdir(os.path.join(a.log, sub)):
            shutil.copytree(os.path.join(a.log, sub), os.path.join(out, sub), dirs_exist_ok=True)
    # розв'язок на точку: положення p і масштаб k відстаней по підлозі (похибка висоти за ToF):
    # cross_i = p + k * r_i, де r_i -- зсув хреста від номінального положення точки
    corrections, scales = {}, {}
    for wp, cs in per_wp.items():
        cs = np.array(cs)
        if len(cs) >= 3 and len({tuple(d["cross"]) for d in details if d["waypoint"] == wp}) >= 2:
            r = np.array([d["r"] for d in details if d["waypoint"] == wp])
            c = np.array([d["cross"] for d in details if d["waypoint"] == wp], float)
            rc, cc = r - r.mean(0), c - c.mean(0)
            k = float(np.clip((rc * cc).sum() / max((rc * rc).sum(), 1e-9), 0.75, 1.3))
            p = c.mean(0) - k * r.mean(0)
            nominal = next(np.array([rec.x, rec.y]) for rec in log.records if rec.waypoint == wp)
            corrections[wp], scales[wp] = p - nominal, k
        else:
            corrections[wp], scales[wp] = np.median(cs, axis=0), 1.0
    with open(os.path.join(out, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
        for rec in log.records:
            c = corrections.get(rec.waypoint, np.zeros(2))
            w.writerow([rec.index, f"{rec.x + c[0]:.3f}", f"{rec.y + c[1]:.3f}", f"{rec.z_tof:.3f}", f"{rec.yaw_deg:.1f}", rec.waypoint])
    meta = dict(log.meta)
    meta["cross_localize"] = {str(k): {"n": len(per_wp[k]), "corr": [round(float(v), 3) for v in c],
                                       "height_scale": round(scales[k], 3)} for k, c in corrections.items()}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(details, open(os.path.join(out, "crosses.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wps = sorted({r.waypoint for r in log.records})
    print(f"{os.path.basename(a.log)}: " + ", ".join(
        f"т{k + 1}: {len(per_wp.get(k, []))} хрест., поправка ({corrections[k][0]:+.2f}, {corrections[k][1]:+.2f}), k {scales[k]:.2f}"
        if k in corrections else f"т{k + 1}: хрестів немає" for k in wps))


if __name__ == "__main__":
    main()
