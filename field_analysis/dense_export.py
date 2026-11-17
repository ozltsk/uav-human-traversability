"""Щільний журнал АС Р1: до 4 знімків у точці додаються кадри з поворотів (через 22,5°).

  python dense_export.py <тека запису> <журнал АС Р1 (16 кадрів)> [--step-deg 22.5] [--out <тека>]

Кожен поворот на 90° у точці триває ~3,6 с і дає 82-94 кадри. Для кожного проміжного курсу
береться найрізкіший кадр у вікні ±4°. Рискання кадру -- з телеметрії з поправкою на запізнення
відео (взаємна кореляція руху картинки й руху дрона, video_lag_xcorr). Положення -- як у точці,
висота -- медіана нижнього ToF у вікні ±0,2 с. Наявні 16 кадрів і їхні передбачення копіюються,
нейромережі потрібно запустити лише для нових кадрів: `ac1_evacmap/app.py predict --log <out>`.
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rmtt_field"))
sys.path.insert(0, HERE)
import postprocess as pp                      # noqa: E402
from video_lag_xcorr import motion_signals    # noqa: E402


def video_lag(run):
    grid, a, b = motion_signals(run)
    lags = np.arange(-30, 31)
    corr = [np.corrcoef(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)])[0, 1] for k in lags]
    return float(lags[int(np.argmax(corr))] * 0.1), float(max(corr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("log")
    ap.add_argument("--step-deg", type=float, default=22.5)
    ap.add_argument("--window-deg", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.log.rstrip("/\\") + "_dense"
    meta = json.load(open(os.path.join(a.log, "meta.json"), encoding="utf-8"))
    mission = json.load(open(os.path.join(a.run, "mission.json"), encoding="utf-8"))
    marks = json.load(open(os.path.join(a.run, "marks.json"), encoding="utf-8"))
    caps = [m for m in marks if m["kind"] == "capture"]
    base = list(csv.DictReader(open(os.path.join(a.log, "telemetry.csv"), encoding="utf-8")))
    assert len(base) == len(caps), "кількість кадрів журналу не збігається з мітками capture"
    st = pp.load_state(a.run)
    lag, corr = video_lag(a.run)
    fr = list(csv.DictReader(open(os.path.join(a.run, "frames.csv"), encoding="utf-8")))
    fr_t = np.array([float(r["t"]) for r in fr])
    fr_s = np.array([float(r["sharpness"]) for r in fr])
    yaw_frame = np.interp(fr_t - lag, st["t"], st["yaw_unwrapped"])       # рискання Tello в момент знімання
    yaw_sign, heading = float(meta["yaw_sign"]), float(meta.get("heading_deg", 0.0))
    yaw_ref = float(pp.interp_state(st, "yaw_unwrapped", next(m["t"] for m in marks if m["kind"] == "yaw_ref")))

    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "pred"), exist_ok=True)
    rows, want = [], {}
    for rec in base:
        rows.append(dict(rec))
        for sub, name in (("frames", f"frame_{int(rec['index']):04d}.png"), ("pred", f"pred_{int(rec['index']):04d}.npz")):
            src = os.path.join(a.log, sub, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, sub, name))
    nxt = len(base)
    for c0, c1, rec in zip(caps, caps[1:], base):
        if c0["waypoint"] != c1["waypoint"]:
            continue
        w0 = pp.state_window(st, "yaw_unwrapped", c0["t"], c0["t"] + c0.get("hold_s", 0.6))
        w1 = pp.state_window(st, "yaw_unwrapped", c1["t"], c1["t"] + c1.get("hold_s", 0.6))
        if not len(w0) or not len(w1):
            continue
        y0, y1 = float(np.median(w0)), float(np.median(w1))
        turn = y1 - y0
        n = int(round(abs(turn) / a.step_deg))
        for k in range(1, n):
            target = y0 + turn * k / n
            sel = np.where((fr_t > c0["t"]) & (fr_t < c1["t"]) & (np.abs(yaw_frame - target) < a.window_deg))[0]
            if not len(sel):
                continue
            best = int(sel[np.argmax(fr_s[sel])])
            tof = pp.state_window(st, "tof", fr_t[best] - 0.2, fr_t[best] + 0.2)
            z = float(np.median(tof)) / 100.0 if len(tof) else float(rec["z_tof"])
            yaw_ac1 = (heading + yaw_sign * (float(yaw_frame[best]) - yaw_ref) + 180.0) % 360.0 - 180.0
            want[best] = {"index": nxt, "x": rec["x"], "y": rec["y"], "z_tof": f"{z:.3f}", "yaw_deg": f"{yaw_ac1:.1f}",
                          "waypoint": rec["waypoint"]}
            nxt += 1
    import cv2
    K = dist = None
    calib = meta.get("calib")
    if calib and os.path.exists(calib):
        cal = json.load(open(calib, encoding="utf-8"))
        K, dist = np.asarray(cal["K"], float), np.asarray(cal["dist"], float).ravel()
    got = 0
    for i, _t, frame in pp.iter_frames(a.run):
        if i in want:
            if K is not None:
                frame = cv2.undistort(frame, K, dist, None, K)
            cv2.imwrite(os.path.join(out, "frames", f"frame_{want[i]['index']:04d}.png"), frame)
            rows.append(want[i])
            got += 1
            if got == len(want):
                break
    rows.sort(key=lambda r: int(r["index"]))
    with open(os.path.join(out, "telemetry.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "x", "y", "z_tof", "yaw_deg", "waypoint"])
        w.writeheader()
        w.writerows(rows)
    meta = dict(meta)
    meta.update({"dense": {"step_deg": a.step_deg, "video_lag_s": lag, "lag_corr": round(corr, 2),
                           "extra_frames": got, "base_frames": len(base)}})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{os.path.basename(out)}: {len(base)} + {got} кадрів, запізнення відео {lag:+.1f} с (кореляція {corr:.2f})")


if __name__ == "__main__":
    main()
